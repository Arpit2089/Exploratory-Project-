# enas/__init__.py
"""
ENAS-Lamarckian integration package for LEMONADE NAS.

New methodology: 3-Tier Lamarckian Inheritance + ENAS Controller

Tier 1 — Direct parent inheritance  (existing Lamarckian, morphisms/weights.py)
Tier 2 — Elite pool inheritance     (new: LamarcikianElitePool)
Tier 3 — Supernet weight bank       (new: SupernetWeightBank, ENAS-style)

Controller — REINFORCE-trained MLP that learns to select the best
             operator for each parent architecture (new: ENASController)
"""
from enas.supernet    import SupernetWeightBank
from enas.controller  import ENASController, OPERATOR_NAMES
from enas.weight_pool import LamarcikianElitePool
from enas.inheritance import ThreeTierInheritance

__all__ = [
    "SupernetWeightBank",
    "ENASController",
    "OPERATOR_NAMES",
    "LamarcikianElitePool",
    "ThreeTierInheritance",
]