import torch
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class NASConfig:

    TARGET_DATASET: str  = "CIFAR-10"
    FAST_DEV_MODE:  bool = False
    BATCH_SIZE:     int  = 64  # Reduced to prevent CPU cache bottlenecks

    USE_PRETRAINED_BACKBONE: bool = False

    # ["params"] = fast NAS (no model builds for filtering)
    # ["params","flops"] = post-hoc analysis only
    CHEAP_OBJECTIVES: List[str] = field(default_factory=lambda: ["params", "flops"])

    GENERATIONS: int = 15       
    NUM_SEEDS:   int = 7
    N_CHILDREN:  int = 21
    N_ACCEPT:    int = 7
    MAX_PARAMS:  int = 5_000_000 
    MIN_POP:     int = 3

    INIT_EPOCHS:       int  = 2
    CHILD_EPOCHS:      int  = 1 
    DISTILL_EPOCHS:    int  = 1
    EPOCH_PROGRESSION: bool = True

    OPTIMIZER:           str   = "sgd"
    INIT_LR:             float = 0.025  
    CHILD_LR:            float = 0.01   
    DISTILL_LR:          float = 0.01
    WEIGHT_DECAY:        float = 3e-4   
    DISTILL_TEMPERATURE: float = 3.0
    DISTILL_ALPHA:       float = 0.1

    KDE_BANDWIDTH: float = 0.3

    # ── Hardware & Compilation (CPU OPTIMIZED) ────────────────────
    USE_AMP:         bool = False 
    USE_MULTI_GPU:   bool = False 
    CUDNN_BENCHMARK: bool = False 
    USE_TF32:        bool = False 
    
    # CRITICAL FIX FOR WINDOWS OOM: 
    # Must be 0 to prevent multiprocessing memory duplication on spawn
    NUM_DATALOADER_WORKERS: int = 0     
    PIN_MEMORY:             bool = False 
    
    COMPILE_MODEL: bool = False 
    COMPILE_MODE:  str  = "default"  

    # ── ENAS 3-Tier Lamarckian integration ─────────────────────────
    USE_ENAS_CONTROLLER:     bool  = False # Set to false per your logs
    CONTROLLER_LR:           float = 3e-4
    CONTROLLER_ENTROPY:      float = 0.01   
    CONTROLLER_UPDATE_EVERY: int   = 8      

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