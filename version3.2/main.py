# main.py
# =============================================================================
# Entry point for LEMONADE NAS.
# ALL hyperparameters live in config.py — edit them there.
#
# ── Running on a laptop GPU (Windows) ────────────────────────────────────────
#   python main.py
#   (NUM_DATALOADER_WORKERS is auto-set to 0 on Windows — no action needed)
#
# ── Running on a laptop GPU (Linux) ──────────────────────────────────────────
#   python main.py
#
# ── Running on Google Colab ───────────────────────────────────────────────────
#   !python main.py
#   OR: from main import main; main()
# =============================================================================

import os
import sys
import platform
import multiprocessing

# ---- Thread-count lock (must happen before any torch import) ----------------
# Prevents CPU thread explosion when multiprocessing is used on CPU path.
os.environ.setdefault("OMP_NUM_THREADS",        "1")
os.environ.setdefault("MKL_NUM_THREADS",        "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",   "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",    "1")

import warnings
warnings.filterwarnings("ignore")

import copy
import pickle
import datetime
import torch

from config import CFG
from evolution.lemonade_full import run_lemonade
from evolution.operators import random_operator
from evolution.individual import Individual
from models.basenet import build_basenet_graph
from utils.logger import get_logger

logger = get_logger("main", logfile="logs/main.log")

# =============================================================================
# Platform-aware safety fixes
# =============================================================================

_IS_WINDOWS = platform.system() == "Windows"

if _IS_WINDOWS and CFG.NUM_DATALOADER_WORKERS != 0:
    # Windows + CUDA DataLoader workers is a known crash.
    # Silently override — user does not need to change config.py.
    CFG.NUM_DATALOADER_WORKERS = 0

# =============================================================================
# Dataset + num_classes wiring
# =============================================================================

_DATASET_META = {
    "CIFAR-10":  (10,   128),
    "CIFAR-100": (100,  128),
    "IMAGENET":  (1000,  32),
}

NUM_CLASSES = _DATASET_META[CFG.TARGET_DATASET][0]
if CFG.TARGET_DATASET == "IMAGENET" and CFG.BATCH_SIZE == 128:
    CFG.BATCH_SIZE = 32   # ImageNet needs smaller batch even on good GPUs


# =============================================================================
# Seed population
# =============================================================================

def create_seed_population(num_seeds: int) -> list:
    logger.info("Building %d seed architectures for %s (%d classes)",
                num_seeds, CFG.TARGET_DATASET, NUM_CLASSES)

    base_graph = build_basenet_graph(
        num_classes=NUM_CLASSES,
        dataset_type=CFG.TARGET_DATASET,
    )
    graphs = [base_graph]

    for _ in range(num_seeds - 1):
        for _ in range(15):
            tmp = Individual(copy.deepcopy(base_graph))
            new_graph, _, _ = random_operator(tmp)
            if new_graph is None:
                continue
            new_ind = Individual(new_graph)
            try:
                cheap = new_ind.evaluate_cheap(
                    objective_keys=CFG.CHEAP_OBJECTIVES,
                    input_size=(1, 3, 32, 32),
                )
                if cheap.get("params", 0) <= CFG.MAX_PARAMS:
                    graphs.append(new_graph)
                    break
            except Exception:
                continue
        else:
            graphs.append(copy.deepcopy(base_graph))

    logger.info("Seed population ready: %d architectures", len(graphs))
    return graphs


# =============================================================================
# GPU setup helper
# =============================================================================

def _configure_gpu(cfg) -> str:
    """
    Detect GPU, apply cuDNN settings, auto-adjust batch size if VRAM is small.
    Returns the device string ("cuda" or "cpu").
    """
    if not torch.cuda.is_available():
        cfg.USE_AMP          = False
        cfg.CUDNN_BENCHMARK  = False
        cfg.PIN_MEMORY       = False
        n_cpu = os.cpu_count() or 1
        print(f"\n  No GPU detected — using CPU ({n_cpu} cores, parallel training)\n")
        return "cpu"

    device   = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    vram_gb  = gpu_props.total_memory / 1e9
    cuda_ver = torch.version.cuda or "unknown"

    # ── cuDNN benchmark ────────────────────────────────────────────────────
    # This is the single biggest laptop-GPU optimisation for fixed-size inputs.
    # It runs once at startup and caches the best algorithm for each op shape.
    torch.backends.cudnn.benchmark    = cfg.CUDNN_BENCHMARK
    torch.backends.cudnn.deterministic = False   # required for benchmark mode

    # ── AMP safety ────────────────────────────────────────────────────────
    # AMP requires compute capability >= 7.0 (Volta+). Older cards (GTX 900/
    # 1000 series, compute 6.x) don't have Tensor Cores; AMP still works but
    # gives no speedup and can cause NaN on some old drivers.
    cc_major = gpu_props.major
    if cc_major < 7:
        cfg.USE_AMP = False
        amp_note = f"DISABLED (GPU compute capability {cc_major}.x < 7.0)"
    else:
        amp_note = "ENABLED" if cfg.USE_AMP else "DISABLED by config"

    # ── VRAM-aware batch-size guard ────────────────────────────────────────
    # Prevent OOM on small laptop GPUs by automatically capping BATCH_SIZE.
    # The user can always increase it manually in config.py.
    if cfg.TARGET_DATASET in ("CIFAR-10", "CIFAR-100"):
        vram_to_max_batch = {4: 64, 6: 128, 8: 256, 12: 512}
        recommended = 64
        for vram_thresh, bs in sorted(vram_to_max_batch.items()):
            if vram_gb >= vram_thresh:
                recommended = bs
        if cfg.BATCH_SIZE > recommended:
            old_bs = cfg.BATCH_SIZE
            cfg.BATCH_SIZE = recommended
            print(f"  [Auto] BATCH_SIZE reduced {old_bs} → {recommended} "
                  f"for {vram_gb:.1f} GB VRAM.")

    print(f"\n{'='*58}")
    print(f"  GPU : {gpu_name}")
    print(f"  VRAM: {vram_gb:.1f} GB   |   CUDA {cuda_ver}")
    print(f"  Compute cap: {gpu_props.major}.{gpu_props.minor}   |   "
          f"SM count: {gpu_props.multi_processor_count}")
    print(f"  AMP (fp16)       : {amp_note}")
    print(f"  cuDNN benchmark  : {'ON' if cfg.CUDNN_BENCHMARK else 'OFF'}")
    print(f"  Batch size       : {cfg.BATCH_SIZE}")
    print(f"  DL workers       : {cfg.NUM_DATALOADER_WORKERS}"
          f"{'  (auto=0 on Windows)' if _IS_WINDOWS else ''}")
    print(f"  Pin memory       : {cfg.PIN_MEMORY}")
    print(f"  Training strategy: Sequential on GPU")
    print(f"{'='*58}\n")

    logger.info(
        "GPU=%s VRAM=%.1fGB CUDA=%s cc=%d.%d AMP=%s benchmark=%s",
        gpu_name, vram_gb, cuda_ver,
        gpu_props.major, gpu_props.minor,
        cfg.USE_AMP, cfg.CUDNN_BENCHMARK,
    )

    return device


# =============================================================================
# Main
# =============================================================================

def main():
    # ------------------------------------------------------------------
    # GPU detection + cuDNN configuration
    # ------------------------------------------------------------------
    device = _configure_gpu(CFG)

    logger.info("Starting LEMONADE NAS | dataset=%s device=%s",
                CFG.TARGET_DATASET, device)
    logger.info("Config: %s", CFG)

    # ------------------------------------------------------------------
    # Timestamped output directory
    # ------------------------------------------------------------------
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"results_{CFG.TARGET_DATASET}_{ts}"
    os.makedirs(run_dir, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    # ------------------------------------------------------------------
    # Data loaders for main process
    # ------------------------------------------------------------------
    from data.loader_factory import get_loaders
    loaders      = get_loaders(CFG, split_test=True)
    train_loader = loaders[0]
    val_loader   = loaders[1]
    test_loader  = loaders[2] if len(loaders) > 2 else None

    # ------------------------------------------------------------------
    # Seed population
    # ------------------------------------------------------------------
    init_graphs = create_seed_population(CFG.NUM_SEEDS)

    # ------------------------------------------------------------------
    # Run LEMONADE
    # ------------------------------------------------------------------
    final_population, history = run_lemonade(
        init_graphs  = init_graphs,
        cfg          = CFG,
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = device,
        run_dir      = run_dir,
    )

    # ------------------------------------------------------------------
    # Save history + config
    # ------------------------------------------------------------------
    history_dir = os.path.join(run_dir, "history")
    os.makedirs(history_dir, exist_ok=True)
    with open(os.path.join(history_dir, "history.pkl"), "wb") as f:
        pickle.dump(history, f)
    with open(os.path.join(run_dir, "config.pkl"), "wb") as f:
        pickle.dump(CFG, f)
    logger.info("History + config saved to %s", run_dir)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    from utils.plot import plot_all_pairs, plot_3d_pareto, plot_convergence
    plot_all_pairs(history, cheap_objectives=CFG.CHEAP_OBJECTIVES,
                   save_dir=plot_dir)
    plot_3d_pareto(history,  save_dir=plot_dir)
    plot_convergence(history, save_dir=plot_dir)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  LEMONADE COMPLETE — {CFG.TARGET_DATASET}")
    print(f"  Results → {run_dir}")
    print("=" * 60)

    for i, ind in enumerate(final_population):
        ve = ind.f_exp.get("val_error")  if ind.f_exp   else None
        p  = ind.f_cheap.get("params")   if ind.f_cheap else None
        fl = ind.f_cheap.get("flops")    if ind.f_cheap else None

        te = None
        if test_loader is not None and ind.model is not None:
            from train.evaluate import evaluate_accuracy
            te = evaluate_accuracy(ind.model, test_loader,
                                   device=device, use_amp=CFG.USE_AMP)

        p_str  = f"{p:>12,}"       if p  is not None else f"{'?':>12}"
        fl_str = f"{int(fl):>14,}" if fl is not None else f"{'?':>14}"
        ve_str = f"{ve:.4f}"       if ve is not None else "?"
        te_str = f"{te:.4f}"       if te is not None else "N/A"
        print(f"  [{i}] params={p_str}  flops={fl_str}  "
              f"val_err={ve_str}  test_err={te_str}")
        logger.info(
            "Final model %d | params=%s flops=%s val_err=%s test_err=%s",
            i, p_str.strip(), fl_str.strip(), ve_str, te_str,
        )

    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Windows requires "spawn"; Linux/Mac default is "fork" which is fine.
    # force=False means: only set if not already set (safe to call multiple times).
    try:
        multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass   # already set — ignore

    main()