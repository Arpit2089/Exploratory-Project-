# config.py
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class NASConfig:

    TARGET_DATASET: str  = "CIFAR-10"
    FAST_DEV_MODE:  bool = False
    BATCH_SIZE:     int  = 256

    USE_PRETRAINED_BACKBONE: bool = True

    # ["params"] = fast NAS (no model builds for filtering)
    # ["params","flops"] = post-hoc analysis only
    CHEAP_OBJECTIVES: List[str] = field(default_factory=lambda: ["params","flops"])

    GENERATIONS: int = 5
    NUM_SEEDS:   int = 6
    N_CHILDREN:  int = 16
    N_ACCEPT:    int = 4
    MAX_PARAMS:  int = 30_000_000
    MIN_POP:     int = 3

    INIT_EPOCHS:       int  = 8
    CHILD_EPOCHS:      int  = 5
    DISTILL_EPOCHS:    int  = 2
    EPOCH_PROGRESSION: bool = True

    OPTIMIZER:           str   = "sgd"
    INIT_LR:             float = 0.001
    CHILD_LR:            float = 0.0005
    DISTILL_LR:          float = 0.001
    WEIGHT_DECAY:        float = 1e-4
    DISTILL_TEMPERATURE: float = 3.0
    DISTILL_ALPHA:       float = 0.1

    KDE_BANDWIDTH: float = 0.3

    USE_AMP:         bool = True
    USE_MULTI_GPU:   bool = True
    CUDNN_BENCHMARK: bool = True
    USE_TF32:        bool = True
    NUM_DATALOADER_WORKERS: int  = 2
    PIN_MEMORY:             bool = True

    # ── ENAS 3-Tier Lamarckian integration (NEW) ──────────────────────
    # REINFORCE controller for guided operator selection (GPU path only).
    # Set False to use original random OP_WEIGHTS selection.
    USE_ENAS_CONTROLLER:     bool  = True
    CONTROLLER_LR:           float = 3e-4
    CONTROLLER_ENTROPY:      float = 0.05   # exploration regularisation
    CONTROLLER_UPDATE_EVERY: int   = 8      # REINFORCE gradient step every N outcomes

    # Tier-2 elite pool: stores top-N architectures' weights for inheritance
    ELITE_POOL_SIZE: int = 20

    SHOW_PROGRESS_BAR: bool = True

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
            raise ValueError(f"Unknown TARGET_DATASET '{self.TARGET_DATASET}'.")
        return _map[self.TARGET_DATASET]


CFG = NASConfig()