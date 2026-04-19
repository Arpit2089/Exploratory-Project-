# enas/supernet.py
"""
Supernet Weight Bank  —  ENAS Tier-3 shared weight store.

Motivation (from the integration doc):
  ENAS uses a *supernet* where all candidate architectures share a common
  parameter set. We adapt this idea for graph-based NAS: instead of a single
  over-parameterised network we maintain a *weight dictionary* keyed by
  (op_type, in_channels, out_channels, kernel_size).  After every training
  run, each layer's weights are averaged into the bank via EMA.  When a new
  child architecture is built, any layer whose signature already exists in the
  bank receives supernet weights instead of random initialisation.

  This is "Tier 3" in the 3-tier hierarchy:
      parent weights  >  elite pool  >  supernet  >  random

Design:
  - EMA decay alpha=0.1  →  bank smoothly tracks the "average good init"
  - Keys are exact shape-matched, so there is never a tensor size mismatch
  - Only learnable layers (conv, sep_conv, bn, linear) are stored
  - Thread/process safe: the bank lives in the main process only
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from utils.logger import get_logger

logger = get_logger("supernet", logfile="logs/supernet.log")

# Exponential Moving Average decay for weight bank updates.
# Lower value → slower but more stable averaging.
_EMA_ALPHA = 0.1


class SupernetWeightBank:
    """
    Shared weight dictionary for ENAS-style Tier-3 initialisation.

    Key format: "<op>_<in>_<out>_<k>"
      e.g.  "conv_64_128_3"  |  "sepconv_32_64_3"  |  "bn_128"  |  "linear_512_200"

    Usage
    -----
    After training any model::

        bank.update_from_model(trained_model, individual.graph)

    Before training a new child::

        n = bank.initialize_model(child_model, child.graph)
        # n = number of layers successfully seeded from the bank
    """

    def __init__(self):
        # {key: {param_name: Tensor}}  — CPU tensors only
        self._bank:   Dict[str, Dict[str, torch.Tensor]] = {}
        # How many models contributed to each key
        self._counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_from_model(self, model: nn.Module, graph) -> int:
        """
        Absorb a freshly trained model into the weight bank (EMA update).
        Call this after every child finishes training.

        Returns the number of layer keys updated.
        """
        if model is None:
            return 0

        updated = 0
        for node_id, node in graph.nodes.items():
            key = self._node_key(node)
            if key is None:
                continue

            layer_key = str(node_id)
            if not hasattr(model, "layers") or layer_key not in model.layers:
                continue

            try:
                # Grab current layer weights, strip thop artefacts
                raw_sd = model.layers[layer_key].state_dict()
                sd = {
                    k: v.detach().cpu().float().clone()
                    for k, v in raw_sd.items()
                    if not any(t in k for t in ("total_ops", "total_params", "profile"))
                    and isinstance(v, torch.Tensor)
                }
                if not sd:
                    continue

                if key not in self._bank:
                    self._bank[key]   = sd
                    self._counts[key] = 1
                else:
                    # EMA update — only update params whose shape still matches
                    existing = self._bank[key]
                    for pname, new_val in sd.items():
                        if pname in existing and existing[pname].shape == new_val.shape:
                            existing[pname] = (
                                (1.0 - _EMA_ALPHA) * existing[pname]
                                + _EMA_ALPHA        * new_val
                            )
                        else:
                            # New shape (e.g. after widen) → just store it
                            existing[pname] = new_val
                    self._counts[key] += 1
                updated += 1
            except Exception as exc:
                logger.debug("Supernet update skip node %s: %s", layer_key, exc)

        logger.debug("Supernet: updated %d layers (bank size=%d unique keys)",
                     updated, len(self._bank))
        return updated

    def initialize_model(self, model: nn.Module, graph) -> int:
        """
        Seed a freshly-built model from the weight bank (Tier-3 fallback).

        Only layers whose (op_type, channels) signature exists in the bank
        AND whose tensor shapes match are initialised.  All others stay at
        the default PyTorch random init — no errors are ever raised.

        Returns the number of layers seeded from the bank.
        """
        if not self._bank or model is None:
            return 0

        seeded = 0
        for node_id, node in graph.nodes.items():
            key = self._node_key(node)
            if key is None or key not in self._bank:
                continue

            layer_key = str(node_id)
            if not hasattr(model, "layers") or layer_key not in model.layers:
                continue

            try:
                stored  = self._bank[key]
                cur_sd  = model.layers[layer_key].state_dict()
                to_load = {
                    k: v.clone()
                    for k, v in stored.items()
                    if k in cur_sd and cur_sd[k].shape == v.shape
                }
                if to_load:
                    model.layers[layer_key].load_state_dict(to_load, strict=False)
                    seeded += 1
            except Exception as exc:
                logger.debug("Supernet init skip node %s: %s", layer_key, exc)

        logger.debug("Supernet: seeded %d layers from bank", seeded)
        return seeded

    def coverage(self) -> int:
        """Number of unique operation signatures in the bank."""
        return len(self._bank)

    def stats(self) -> dict:
        """Return summary statistics for logging/checkpointing."""
        return {
            "n_keys":   len(self._bank),
            "counts":   dict(self._counts),
        }

    def state_dict(self) -> dict:
        """Serialise for checkpointing alongside the LEMONADE run."""
        return {"bank": self._bank, "counts": self._counts}

    def load_state_dict(self, d: dict):
        """Restore from checkpoint."""
        self._bank   = d.get("bank",   {})
        self._counts = d.get("counts", {})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_key(node) -> Optional[str]:
        """
        Build a string key from a graph Node.
        Returns None for ops without learnable parameters
        (relu, maxpool, flatten, add, concat, identity).
        """
        op = node.op_type
        p  = node.params
        try:
            if op == "conv":
                ic = p["in_channels"]
                oc = p["out_channels"]
                k  = p.get("kernel_size", 3)
                return f"conv_{ic}_{oc}_{k}"
            elif op in ("sep_conv", "separableconv2d"):
                ic = p["in_channels"]
                oc = p["out_channels"]
                k  = p.get("kernel_size", 3)
                return f"sepconv_{ic}_{oc}_{k}"
            elif op == "bn":
                f = p.get("num_features", 0)
                return f"bn_{f}"
            elif op in ("fc", "linear"):
                i = p.get("in_features", 0)
                o = p.get("out_features", 0)
                return f"linear_{i}_{o}"
        except (KeyError, TypeError):
            pass
        return None