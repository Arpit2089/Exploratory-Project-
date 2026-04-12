# config.py
# =============================================================================
# ALL USER-FACING HYPERPARAMETERS ARE IN THIS ONE FILE.
# Change values here; nothing else needs to be touched for normal experiments.
# =============================================================================

from dataclasses import dataclass, field
from typing import List


@dataclass
class NASConfig:

    # -------------------------------------------------------------------------
    # Dataset
    # Options: "CIFAR-10" | "CIFAR-100" | "IMAGENET"
    # -------------------------------------------------------------------------
    TARGET_DATASET: str = "CIFAR-10"

    # True  → small subset of data (fast iteration / debugging)
    # False → full dataset (production runs)
    FAST_DEV_MODE: bool = False

    BATCH_SIZE: int = 128

    # -------------------------------------------------------------------------
    # Cheap Objectives
    # Controls what enters KDE sampling + Pareto dominance.
    # Options: any subset of ["params", "flops"]
    #   ["params", "flops"]  → full multi-objective (slower, needs thop)
    #   ["params"]           → params-only (fast, no model build for cheap eval)
    #   []                   → single-objective NAS (val_error only)
    # -------------------------------------------------------------------------
    CHEAP_OBJECTIVES: List[str] = field(default_factory=lambda: ["params"])
    # CHEAP_OBJECTIVES: List[str] = field(default_factory=lambda: [])
    # CHEAP_OBJECTIVES: List[str] = field(default_factory=lambda: ["params", "flops"])

    # -------------------------------------------------------------------------
    # Evolution
    # -------------------------------------------------------------------------
    GENERATIONS: int = 6
    NUM_SEEDS: int = 16

    # npc: candidates generated per generation.  Must be > N_ACCEPT.
    N_CHILDREN: int = 20

    # nac: how many pass the KDE filter and enter expensive training.
    # Ratio N_CHILDREN / N_ACCEPT should be at least 3:1 for meaningful filtering.
    N_ACCEPT: int = 12

    # Hard parameter cap; children exceeding this are discarded before training.
    MAX_PARAMS: int = 10_000_000

    # Minimum Pareto population size.  If Pareto front shrinks below this,
    # diverse dominated individuals are added to maintain breadth.
    MIN_POP: int = 3

    # -------------------------------------------------------------------------
    # Training Epochs  (THREE distinct phases — do NOT conflate them)
    #
    #   INIT_EPOCHS   : Gen 0 — training the seed population from scratch
    #   CHILD_EPOCHS  : Gen 1+ — standard training for every accepted child
    #   DISTILL_EPOCHS: ANM children only — distillation alignment BEFORE
    #                   the standard CHILD_EPOCHS training phase
    # -------------------------------------------------------------------------
    INIT_EPOCHS: int = 6
    CHILD_EPOCHS: int = 4
    DISTILL_EPOCHS: int = 2

    # Set True to add +1 epoch every 5 generations (slow convergence boost)
    EPOCH_PROGRESSION: bool = True

    # -------------------------------------------------------------------------
    # Optimizer & Regularisation
    # OPTIMIZER options: "sgd" | "adam" | "adamw"
    # -------------------------------------------------------------------------
    OPTIMIZER: str = "sgd"

    # Learning rates — separate per phase
    INIT_LR: float = 0.025        # Gen 0
    CHILD_LR: float = 0.01      # Gen 1+ training
    DISTILL_LR: float = 0.005    # Distillation alignment
    # INIT_LR: float = 0.01        # Gen 0
    # CHILD_LR: float = 0.005      # Gen 1+ training
    # DISTILL_LR: float = 0.001    # Distillation alignment

    WEIGHT_DECAY: float = 3e-4

    # Distillation settings
    DISTILL_TEMPERATURE: float = 4.0
    # 0.0 = pure teacher-mimicking, 1.0 = pure cross-entropy on hard labels
    DISTILL_ALPHA: float = 0.5

    # -------------------------------------------------------------------------
    # KDE Sampler
    # Higher bandwidth → smoother density → more exploration (good early on).
    # Lower bandwidth → sharper density → more exploitation (good later).
    # -------------------------------------------------------------------------
    KDE_BANDWIDTH: float = 0.3

    # -------------------------------------------------------------------------
    # UX / Debugging
    # -------------------------------------------------------------------------
    # Set False for non-interactive environments (CI, HPC) or when logs are
    # preferred over animated bars.  Workers always suppress their own bars.
    SHOW_PROGRESS_BAR: bool = False


# Singleton — import this everywhere
CFG = NASConfig()