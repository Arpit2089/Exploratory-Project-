# enas/inheritance.py
"""
Three-Tier Lamarckian Inheritance Orchestrator.

Motivation (from the integration doc, Strategy B):
  "When a new candidate is generated:
    - If parent architecture exists → inherit parent weights.
    - Else if similar architecture exists → inherit from elite pool.
    - Else → use supernet weights."
  → This creates a 3-tier inheritance system.

  Strategy A augmentation: Even when Tier-1 succeeds, Tiers 2/3 can still
  fill in layers that the morphism ADDED (new layers have no parent weights).

Tier summary
------------
Tier 1 — Direct parent Lamarckian
    Uses morphisms/weights.py::transfer_weights().
    Copies matching-key layers from parent model; initialises new/modified
    layers according to the specific morphism type.

Tier 2 — Elite pool Lamarckian
    Finds architecturally most similar entry in LamarcikianElitePool and
    loads its weights via strict=False.

Tier 3 — Supernet EMA bank
    Seeds any remaining uninitialised layers from the supernet's
    exponential-moving-average weight bank.

All tiers use strict=False, so they never crash due to shape mismatches.
The tiers compose — e.g. Tier 1 initialises 80% of layers, then Tier 3
fills the new conv layer that the net2deeper morphism inserted.
"""

import torch.nn as nn
from typing import Optional, Dict
from utils.logger import get_logger

logger = get_logger("inheritance", logfile="logs/inheritance.log")

TierResult = str   # "parent" | "elite_pool" | "supernet" | "random"


class ThreeTierInheritance:
    """
    Orchestrates the 3-tier weight initialisation strategy.

    Parameters
    ----------
    supernet   : SupernetWeightBank  — Tier-3 fallback
    elite_pool : LamarcikianElitePool — Tier-2 fallback

    Usage
    -----
    ::
        inherit = ThreeTierInheritance(supernet, elite_pool)

        tier = inherit.initialize_child(
            child_model=child_model,
            child_graph=child.graph,
            child_ind=child,
            parent_model=parent_model,    # None for gen-0
            parent_graph=parent.graph,
            op_name=op_name,
            target_info=target_info,
        )
        # tier ∈ {"parent", "elite_pool", "supernet", "random"}
    """

    def __init__(self, supernet, elite_pool):
        self.supernet   = supernet
        self.elite_pool = elite_pool

        # Cumulative tier usage counters (for logging)
        self._stats: Dict[str, int] = {
            "parent":     0,
            "elite_pool": 0,
            "supernet":   0,
            "random":     0,
        }

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def initialize_child(self,
                         child_model:  nn.Module,
                         child_graph,
                         child_ind,
                         parent_model:  Optional[nn.Module] = None,
                         parent_graph=None,
                         op_name:       Optional[str] = None,
                         target_info:   Optional[dict] = None,
                         ) -> TierResult:
        """
        Initialise child_model weights using the best available tier.

        Decision logic:
            1. Try Tier 1 (parent morphism inheritance).
               If successful, ALSO apply Tier 3 to fill new layers.
               Return "parent".

            2. Try Tier 2 (elite pool).
               If successful, ALSO apply Tier 3 to fill remaining new layers.
               Return "elite_pool".

            3. Try Tier 3 (supernet bank).
               If at least 1 layer seeded, return "supernet".

            4. Else return "random" (PyTorch default init).

        Returns
        -------
        TierResult — the primary tier that succeeded.
        """
        tier: TierResult = "random"

        # ── Tier 1: Direct Lamarckian parent inheritance ───────────────
        if parent_model is not None and op_name is not None:
            t1_ok = self._tier1(child_model, child_graph,
                                parent_model, parent_graph,
                                op_name, target_info)
            if t1_ok:
                tier = "parent"
                # Still apply Tier 3 to seed any NEW layers the morphism added
                self._tier3(child_model, child_graph)
                self._stats["parent"] += 1
                return tier

        # ── Tier 2: Elite pool Lamarckian inheritance ──────────────────
        t2_ok, score = self.elite_pool.inherit_weights(child_model, child_ind)
        if t2_ok:
            tier = "elite_pool"
            # Tier 3 fills remaining layers not covered by pool weights
            self._tier3(child_model, child_graph)
            self._stats["elite_pool"] += 1
            return tier

        # ── Tier 3: Supernet EMA bank ──────────────────────────────────
        n_seeded = self._tier3(child_model, child_graph)
        if n_seeded > 0:
            tier = "supernet"
            self._stats["supernet"] += 1
        else:
            self._stats["random"] += 1

        return tier

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_stats(self) -> Dict[str, int]:
        """Log and return cumulative tier usage counts."""
        total = sum(self._stats.values()) or 1
        logger.info(
            "3-Tier inheritance stats: "
            "parent=%.1f%%(%d) elite=%.1f%%(%d) "
            "supernet=%.1f%%(%d) random=%.1f%%(%d)",
            100 * self._stats["parent"]     / total, self._stats["parent"],
            100 * self._stats["elite_pool"] / total, self._stats["elite_pool"],
            100 * self._stats["supernet"]   / total, self._stats["supernet"],
            100 * self._stats["random"]     / total, self._stats["random"],
        )
        return dict(self._stats)

    def reset_stats(self):
        for k in self._stats:
            self._stats[k] = 0

    # ------------------------------------------------------------------
    # Per-tier implementation
    # ------------------------------------------------------------------

    def _tier1(self,
               child_model, child_graph,
               parent_model, parent_graph,
               op_name: str, target_info: Optional[dict]) -> bool:
        """
        Tier 1: existing Lamarckian transfer via morphisms/weights.py.
        Returns True on success, False on any exception.
        """
        try:
            from morphisms.weights import transfer_weights
            transfer_weights(parent_model, child_model, child_graph,
                             op_name, target_info or {})
            return True
        except Exception as exc:
            logger.debug("Tier-1 failed: %s", exc)
            return False

    def _tier3(self, child_model, child_graph) -> int:
        """
        Tier 3: supernet EMA bank seeding.
        Returns number of layers initialised (0 means bank is empty).
        """
        try:
            return self.supernet.initialize_model(child_model, child_graph)
        except Exception as exc:
            logger.debug("Tier-3 supernet init failed: %s", exc)
            return 0