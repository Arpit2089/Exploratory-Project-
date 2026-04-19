# enas/weight_pool.py
"""
Lamarckian Elite Weight Pool  —  Tier-2 inheritance store.

Motivation (from the integration doc):
  "Maintain a Lamarckian elite pool storing the best-performing architectures'
  weights.  When a new candidate is generated: if parent exists → inherit
  parent weights; if similar architecture exists → inherit from elite pool;
  else → use supernet weights."

  This creates the 3-tier inheritance system described in Strategy B.

Pool mechanics
--------------
- Stores the top-MAX_SIZE architectures (by Pareto quality) with their
  serialised state dicts in temp files.
- When a new child model needs weights, the pool finds the *structurally
  most similar* stored architecture and loads its weights (strict=False).
- Similarity is measured by a weighted combination of:
    1. Op-type LCS sequence similarity  (weight 0.50)
    2. Channel profile cosine similarity (weight 0.30)
    3. Log-scale parameter count proximity (weight 0.20)
- A minimum similarity threshold prevents inheritance from architectures
  that are too dissimilar (which would be worse than random init).
- Eviction policy: remove dominated / highest-val-error entries first.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import tempfile
from typing import List, Optional, Tuple
from utils.logger import get_logger

logger = get_logger("weight_pool", logfile="logs/weight_pool.log")

_MAX_POOL_SIZE          = 20
_MIN_SIMILARITY         = 0.30   # below this threshold, skip inheritance
_SEQUENCE_WEIGHT        = 0.50
_CHANNEL_WEIGHT         = 0.30
_PARAM_WEIGHT           = 0.20


# =============================================================================
# Structural feature extraction helpers
# =============================================================================

def _op_sequence(graph) -> List[str]:
    try:
        topo = graph.topological_sort()
        return [graph.nodes[n].op_type for n in topo]
    except Exception:
        return []


def _channel_profile(graph) -> np.ndarray:
    """
    Sorted, L2-normalised vector of all channel/feature dimensions found in
    conv and linear layers.  Used for cosine similarity matching.
    """
    vals = []
    for node in graph.nodes.values():
        p  = node.params
        op = node.op_type
        if op in ("conv", "sep_conv", "separableconv2d"):
            vals.append(p.get("in_channels",  0))
            vals.append(p.get("out_channels", 0))
        elif op in ("fc", "linear"):
            vals.append(p.get("in_features",  0))
            vals.append(p.get("out_features", 0))
    if not vals:
        return np.zeros(4)
    arr  = np.array(sorted(vals), dtype=float)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def _lcs_similarity(a: List[str], b: List[str]) -> float:
    """
    Normalised longest-common-subsequence similarity ∈ [0, 1].
    Fast dynamic programming implementation.
    """
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    # Use 1-row rolling DP to save memory
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for ai in a:
        for j, bj in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if ai == bj else max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    lcs = prev[n]
    return (2 * lcs) / (m + n)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two arrays, zero-padded to equal length."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    l  = max(len(a), len(b))
    pa = np.pad(a, (0, l - len(a)))
    pb = np.pad(b, (0, l - len(b)))
    d  = np.linalg.norm(pa) * np.linalg.norm(pb)
    return float(np.dot(pa, pb) / d) if d > 0 else 0.0


def _param_similarity(p1: int, p2: int) -> float:
    """Log-scale parameter proximity ∈ [0, 1].  0 if >100× apart."""
    if p1 <= 0 or p2 <= 0:
        return 0.5
    log_ratio = abs(np.log10(max(p1, 1)) - np.log10(max(p2, 1)))
    return max(0.0, 1.0 - log_ratio / 2.0)


# =============================================================================
# Pool entry
# =============================================================================

class _PoolEntry:
    """Lightweight record stored in the elite pool (no tensors in RAM)."""
    __slots__ = ("ind_id", "val_error", "params", "sd_path",
                 "op_seq", "chan_profile")

    def __init__(self, ind, sd_path: str):
        self.ind_id      = ind.id
        self.val_error   = (ind.f_exp   or {}).get("val_error", 1.0)
        self.params      = (ind.f_cheap or {}).get("params",    0)
        self.sd_path     = sd_path
        self.op_seq      = _op_sequence(ind.graph)
        self.chan_profile = _channel_profile(ind.graph)

    def similarity_to(self, candidate_ind) -> float:
        """Composite similarity between this entry and a candidate."""
        tgt_seq  = _op_sequence(candidate_ind.graph)
        tgt_chan = _channel_profile(candidate_ind.graph)
        tgt_p    = (candidate_ind.f_cheap or {}).get("params", 0)

        seq_sim  = _lcs_similarity(self.op_seq, tgt_seq)
        chan_sim  = _cosine_similarity(self.chan_profile, tgt_chan)
        par_sim  = _param_similarity(self.params, tgt_p)

        return (_SEQUENCE_WEIGHT * seq_sim
                + _CHANNEL_WEIGHT  * chan_sim
                + _PARAM_WEIGHT    * par_sim)


# =============================================================================
# Public class
# =============================================================================

class LamarcikianElitePool:
    """
    Maintains a bounded collection of best-performing architectures with
    their trained weights.  Provides Tier-2 weight inheritance.

    Parameters
    ----------
    max_size : maximum number of entries to keep
    temp_dir : directory for serialised state-dict files
               (defaults to system temp dir)
    """

    def __init__(self,
                 max_size: int = _MAX_POOL_SIZE,
                 temp_dir: Optional[str] = None):
        self._pool:     List[_PoolEntry] = []
        self._max_size  = max_size
        self._temp_dir  = temp_dir or tempfile.gettempdir()
        # Track all written paths for cleanup
        self._all_paths: List[str] = []

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def update(self, trained_individuals: list):
        """
        Add newly trained individuals to the pool.
        Evict dominated / worst-error entries to stay within max_size.

        Call once per generation with the list of children that were
        successfully trained.
        """
        for ind in trained_individuals:
            if ind.model is None or ind.f_exp is None:
                continue
            try:
                sd_path = os.path.join(
                    self._temp_dir,
                    f"elite_{ind.id}.pt"
                )
                torch.save(ind.model.state_dict(), sd_path)
                entry = _PoolEntry(ind, sd_path)
                self._pool.append(entry)
                self._all_paths.append(sd_path)
            except Exception as exc:
                logger.warning("Pool: failed to store ind %s: %s", ind.id, exc)

        self._evict()
        logger.info("Elite pool: %d entries after update", len(self._pool))

    # ------------------------------------------------------------------
    # Tier-2 inheritance
    # ------------------------------------------------------------------

    def inherit_weights(self,
                        target_model: nn.Module,
                        target_ind) -> Tuple[bool, float]:
        """
        Find the most similar pool entry and load its weights into target_model.

        Uses strict=False so only matching-shape parameters are loaded.
        Non-matching layers remain at their current initialisation (from Tier-1
        or random).

        Returns
        -------
        (success, similarity_score)
        """
        if not self._pool:
            return False, 0.0

        best_entry: Optional[_PoolEntry] = None
        best_score = -1.0
        for entry in self._pool:
            score = entry.similarity_to(target_ind)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is None or best_score < _MIN_SIMILARITY:
            logger.debug("Pool: no match above threshold (best=%.3f < %.3f)",
                         best_score, _MIN_SIMILARITY)
            return False, float(best_score)

        if not os.path.exists(best_entry.sd_path):
            logger.warning("Pool: weight file missing for %s", best_entry.ind_id)
            self._pool = [e for e in self._pool if e is not best_entry]
            return False, 0.0

        try:
            sd = torch.load(best_entry.sd_path, map_location="cpu",
                            weights_only=True)
            # Strip thop artefacts
            sd = {k: v for k, v in sd.items()
                  if not any(t in k for t in ("total_ops", "total_params", "profile"))}
            target_model.load_state_dict(sd, strict=False)
            logger.info("Pool: Tier-2 inheritance from %s (sim=%.3f params=%d → %d)",
                        best_entry.ind_id, best_score, best_entry.params,
                        (target_ind.f_cheap or {}).get("params", 0))
            return True, float(best_score)
        except Exception as exc:
            logger.warning("Pool: load failed for %s: %s", best_entry.ind_id, exc)
            return False, 0.0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._pool)

    def cleanup(self):
        """Delete all temp weight files and clear the pool."""
        for path in self._all_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        self._all_paths.clear()
        self._pool.clear()
        logger.info("Elite pool cleaned up.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict(self):
        """
        Remove excess entries. Priority: keep low-error, Pareto-non-dominated ones.
        Sort ascending by (val_error, params) and keep top max_size.
        """
        if len(self._pool) <= self._max_size:
            return

        self._pool.sort(key=lambda e: (e.val_error, e.params))
        evicted = self._pool[self._max_size:]
        self._pool = self._pool[:self._max_size]

        for e in evicted:
            try:
                if os.path.exists(e.sd_path):
                    os.remove(e.sd_path)
            except OSError:
                pass
        logger.debug("Pool: evicted %d entries, keeping %d",
                     len(evicted), len(self._pool))