# config.py
# =============================================================================
# ALL USER-FACING HYPERPARAMETERS ARE IN THIS ONE FILE.
# Change values here; nothing else needs to be touched.
#
# ── Laptop GPU (NVIDIA, CUDA) ────────────────────────────────────────────────
#   - USE_AMP = True          → fp16 training, 1.5-2x speedup on any NVIDIA GPU
#   - CUDNN_BENCHMARK = True  → cuDNN auto-tunes kernels for your exact GPU.
#                               BIG speedup when input sizes never change.
#                               CIFAR always 32x32 → always enable this.
#   - NUM_DATALOADER_WORKERS  → Windows: MUST be 0 (Windows+CUDA workers crash)
#                               The code auto-sets 0 on Windows for you.
#                               Linux/Mac: 2-4 is good.
#   - BATCH_SIZE              → Tune to your GPU VRAM:
#                                 4 GB  →  64
#                                 6 GB  → 128 (sweet spot for most laptops)
#                                 8 GB  → 256
#                                12 GB  → 512
#   - PIN_MEMORY = True       → fast CPU→GPU transfers (auto-disabled on CPU)
#
# ── Google Colab ─────────────────────────────────────────────────────────────
#   - NUM_DATALOADER_WORKERS = 2  (Colab Linux has 2 prefetch-CPU cores)
#   - BATCH_SIZE = 128            (Colab T4 = 15 GB VRAM)
#   - Everything else is the same.
# =============================================================================

from dataclasses import dataclass, field
from typing import List


@dataclass
class NASConfig:

    # -------------------------------------------------------------------------
    # Dataset
    # Options: "CIFAR-10" | "CIFAR-100" | "IMAGENET"
    # -------------------------------------------------------------------------
    TARGET_DATASET: str = "CIFAR-100"

    # True  → small subset of data (fast iteration / debugging)
    # False → full dataset (production runs)
    FAST_DEV_MODE: bool = False

    # Tune to your GPU VRAM (see table above)
    BATCH_SIZE: int = 128

    # -------------------------------------------------------------------------
    # Cheap Objectives
    # Options: any subset of ["params", "flops"]
    #   ["params", "flops"]  → full multi-objective (needs thop)
    #   ["params"]           → params-only (fastest, recommended)
    #   []                   → single-objective NAS (val_error only)
    # -------------------------------------------------------------------------
    CHEAP_OBJECTIVES: List[str] = field(default_factory=lambda: ["params"])

    # -------------------------------------------------------------------------
    # Evolution
    # -------------------------------------------------------------------------
    GENERATIONS: int = 5
    NUM_SEEDS: int = 12

    # npc: candidates generated per generation.  Must be > N_ACCEPT.
    N_CHILDREN: int = 24

    # nac: how many pass the KDE filter and enter expensive training.
    # Ratio N_CHILDREN / N_ACCEPT should be at least 3:1.
    N_ACCEPT: int = 8

    # Hard parameter cap; children exceeding this are discarded before training.
    MAX_PARAMS: int = 10_000_000

    # Minimum Pareto population size.
    MIN_POP: int = 3

    # -------------------------------------------------------------------------
    # Training Epochs  (THREE distinct phases — do NOT conflate them)
    #
    #   INIT_EPOCHS   : Gen 0 — train seed population from scratch
    #   CHILD_EPOCHS  : Gen 1+ — standard training for accepted children
    #   DISTILL_EPOCHS: ANM children only — distillation BEFORE CHILD_EPOCHS
    # -------------------------------------------------------------------------
    INIT_EPOCHS: int = 20
    CHILD_EPOCHS: int = 12
    DISTILL_EPOCHS: int = 5

    # Set True to add +1 epoch every 5 generations
    EPOCH_PROGRESSION: bool = True

    # -------------------------------------------------------------------------
    # Optimizer & Regularisation
    # OPTIMIZER options: "sgd" | "adam" | "adamw"
    # -------------------------------------------------------------------------
    OPTIMIZER: str = "sgd"

    INIT_LR: float = 0.01
    CHILD_LR: float = 0.005
    DISTILL_LR: float = 0.001

    WEIGHT_DECAY: float = 1e-4

    DISTILL_TEMPERATURE: float = 3.0
    DISTILL_ALPHA: float = 0.0   # 0.0 = pure KD alignment, 1.0 = pure CE

    # -------------------------------------------------------------------------
    # KDE Sampler
    # -------------------------------------------------------------------------
    KDE_BANDWIDTH: float = 0.3

    # -------------------------------------------------------------------------
    # GPU / Performance
    # -------------------------------------------------------------------------

    # Automatic Mixed Precision (fp16 forward + fp32 gradients).
    # 1.5-2x speedup on NVIDIA GPUs. Auto-disabled on CPU — safe to leave True.
    USE_AMP: bool = True

    # cuDNN benchmark mode: lets cuDNN profile and choose the fastest
    # convolution algorithm for your specific GPU and input shape.
    # CIFAR images are always 32x32 so shapes never change → always a win.
    # Set False only if your input sizes vary significantly between batches.
    CUDNN_BENCHMARK: bool = True

    # DataLoader prefetch worker processes.
    # ⚠ Windows + CUDA: set 0. The code auto-detects Windows and overrides
    #   this to 0, so you don't have to change it manually.
    # Linux/Mac with GPU: 2-4 works well.
    # CPU-only (no GPU): 0 (subprocess overhead cancels any gain).
    NUM_DATALOADER_WORKERS: int = 2

    # Pin CPU memory for fast zero-copy CPU→GPU transfers.
    # Auto-disabled when CUDA is not available — safe to leave True always.
    PIN_MEMORY: bool = True

    # -------------------------------------------------------------------------
    # UX / Debugging
    # -------------------------------------------------------------------------
    SHOW_PROGRESS_BAR: bool = True


# Singleton — import this everywhere
CFG = NASConfig()