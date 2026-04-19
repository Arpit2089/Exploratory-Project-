# config.py
# =============================================================================
# ALL USER-FACING HYPERPARAMETERS ARE IN THIS ONE FILE.
# Change values here; nothing else needs to be touched.
#
# ── KAGGLE GPU NOTES ─────────────────────────────────────────────────────────
#   Kaggle provides T4 (15 GB) or P100 (16 GB) GPU instances.
#
#   Key Kaggle settings:
#     - NUM_DATALOADER_WORKERS = 2   ← Kaggle CPU quota is 2 cores on GPU tier
#     - BATCH_SIZE = 256             ← T4/P100 have 15-16 GB, 256 is safe & fast
#     - USE_AMP = True               ← ~2x speedup, always enable
#     - CUDNN_BENCHMARK = True       ← ~2-4x speedup for fixed-size inputs
#     - USE_TF32 = True              ← Free speedup on A100, ignored on T4/P100
#     - PIN_MEMORY = True            ← Fast CPU→GPU transfers
#     - SHOW_PROGRESS_BAR = True     ← Useful in Kaggle notebooks
#
# ── BEST tiny_net CONFIG FOR KAGGLE ──────────────────────────────────────────
#   Recommended settings for a full NAS run that completes within Kaggle's
#   9-hour session limit:
#
#     GENERATIONS    = 5      # 5 gens gives good diversity
#     NUM_SEEDS      = 6      # 6 seeds: enough diversity, not too slow
#     N_CHILDREN     = 16     # 16 candidates per generation (3:1 filter ratio)
#     N_ACCEPT       = 6      # 6 pass KDE filter → train
#     INIT_EPOCHS    = 15     # 15 for from-scratch, 8 if USE_PRETRAINED_BACKBONE
#     CHILD_EPOCHS   = 5      # Quick child training
#     DISTILL_EPOCHS = 2      # Short distillation
#
#   With USE_PRETRAINED_BACKBONE=True:
#     INIT_EPOCHS    = 8
#     CHILD_EPOCHS   = 5
#     INIT_LR        = 0.001
#     CHILD_LR       = 0.0005
#
# ── WINDOWS / LOCAL GPU NOTES ────────────────────────────────────────────────
#   - NUM_DATALOADER_WORKERS = 4   ← for laptops with 6-12 cores
#   - BATCH_SIZE = 128             ← safe for 4-6 GB VRAM
#   - USE_MULTI_GPU = True         ← enable for multi-GPU desktops
# =============================================================================

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class NASConfig:

    # -------------------------------------------------------------------------
    # Dataset
    # Options: "CIFAR-10" | "CIFAR-100" | "TINY-IMAGENET" | "IMAGENET"
    # -------------------------------------------------------------------------
    TARGET_DATASET: str = "TINY-IMAGENET"

    # True  → small subset of data (fast iteration / debugging)
    # False → full dataset (production runs)
    FAST_DEV_MODE: bool = False

    # Kaggle T4/P100 (15-16 GB): use 256 for maximum throughput.
    # Windows laptop (4-6 GB): use 128.
    BATCH_SIZE: int = 256

    # -------------------------------------------------------------------------
    # Pretrained Backbone
    # -------------------------------------------------------------------------
    # True  → ResNet18 ImageNet weights. Converges in fewer epochs.
    #         Also lower INIT_LR=0.001, CHILD_LR=0.0005.
    # False → BaseNet from scratch (higher LR, more epochs needed).
    USE_PRETRAINED_BACKBONE: bool = True

    # -------------------------------------------------------------------------
    # Cheap Objectives
    # -------------------------------------------------------------------------
    CHEAP_OBJECTIVES: List[str] = field(default_factory=lambda: ["params", "flops"])

    # -------------------------------------------------------------------------
    # Evolution  (Kaggle-optimised for 9-hour session)
    # -------------------------------------------------------------------------
    GENERATIONS: int = 5
    NUM_SEEDS:   int = 8

    # npc: candidates generated per generation. Must be > N_ACCEPT.
    N_CHILDREN: int = 18

    # nac: how many pass the KDE filter and enter expensive training.
    # Ratio N_CHILDREN / N_ACCEPT should be ≥ 3:1.
    N_ACCEPT: int = 6

    MAX_PARAMS: int = 30_000_000
    MIN_POP:    int = 3

    # -------------------------------------------------------------------------
    # Training Epochs  (THREE distinct phases)
    #
    #   USE_PRETRAINED_BACKBONE=True  →  INIT_EPOCHS=8,  CHILD_EPOCHS=5
    #   USE_PRETRAINED_BACKBONE=False →  INIT_EPOCHS=15, CHILD_EPOCHS=5
    # -------------------------------------------------------------------------
    INIT_EPOCHS:    int = 15     # Gen-0 seed training (8 for pretrained, 15 for scratch)
    CHILD_EPOCHS:   int = 6     # Gen 1+ child training
    DISTILL_EPOCHS: int = 2     # ANM distillation phase

    # Add +1 epoch every 5 generations for progressive difficulty
    EPOCH_PROGRESSION: bool = True

    # -------------------------------------------------------------------------
    # Optimizer & Regularisation
    # -------------------------------------------------------------------------
    OPTIMIZER: str = "sgd"

    # Fine-tuning LRs (for USE_PRETRAINED_BACKBONE=True)
    INIT_LR:    float = 0.001
    CHILD_LR:   float = 0.0005
    DISTILL_LR: float = 0.001

    # From-scratch LRs — change these if USE_PRETRAINED_BACKBONE=False:
    # INIT_LR    = 0.05
    # CHILD_LR   = 0.01

    WEIGHT_DECAY: float = 1e-4

    DISTILL_TEMPERATURE: float = 3.0
    DISTILL_ALPHA:       float = 0.1   # 0.0=pure KD, 1.0=pure CE

    # -------------------------------------------------------------------------
    # KDE Sampler bandwidth
    # -------------------------------------------------------------------------
    KDE_BANDWIDTH: float = 0.3

    # -------------------------------------------------------------------------
    # GPU / Performance
    # -------------------------------------------------------------------------
    # AMP: ~2x speedup on any NVIDIA GPU. Auto-disabled on CPU.
    USE_AMP: bool = True

    # Multi-GPU: DataParallel across all available GPUs.
    # Kaggle only has 1 GPU, so this is a no-op there.
    USE_MULTI_GPU: bool = True

    # cuDNN benchmark: picks fastest conv kernel for your GPU & input size.
    # Fixed 64×64 input → permanent 2-4x speedup after first ~10 batches.
    CUDNN_BENCHMARK: bool = True

    # TF32: free 2-3x matmul speedup on Ampere/Ada GPUs (A100, RTX 30/40).
    # T4/P100 ignore this safely.
    USE_TF32: bool = True

    # Kaggle GPU tier = 2 CPU cores → use 2.
    # Windows 6-12 core laptop → use 4.
    NUM_DATALOADER_WORKERS: int = 2

    # Pin memory for fast CPU→GPU transfers. Auto-disabled without CUDA.
    PIN_MEMORY: bool = True

    # -------------------------------------------------------------------------
    # UX / Debugging
    # -------------------------------------------------------------------------
    SHOW_PROGRESS_BAR: bool = True

    # -------------------------------------------------------------------------
    # Derived helpers — do NOT edit these
    # -------------------------------------------------------------------------
    def get_input_size(self) -> Tuple[int, int, int, int]:
        if self.TARGET_DATASET == "IMAGENET":
            return (1, 3, 224, 224)
        elif self.TARGET_DATASET == "TINY-IMAGENET":
            return (1, 3, 64, 64)
        else:
            return (1, 3, 32, 32)

    def get_num_classes(self) -> int:
        _map = {
            "CIFAR-10":      10,
            "CIFAR-100":    100,
            "TINY-IMAGENET": 200,
            "IMAGENET":     1000,
        }
        if self.TARGET_DATASET not in _map:
            raise ValueError(
                f"Unknown TARGET_DATASET '{self.TARGET_DATASET}'. "
                f"Valid choices: {list(_map.keys())}"
            )
        return _map[self.TARGET_DATASET]


# Singleton — import this everywhere
CFG = NASConfig()