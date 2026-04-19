# main.py
# =============================================================================
# Entry point for LEMONADE NAS.
# ALL hyperparameters live in config.py — edit them there.
#
# Windows / Local GPU usage:
#   python main.py
#   (Make sure you installed PyTorch with the correct CUDA version first —
#    see requirements.txt for instructions)
# =============================================================================

import os
import multiprocessing

# ---- Thread-count lock (must happen before any torch import) ----
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
# GPU speed settings
# =============================================================================

def _apply_gpu_speed_settings():
    """
    Apply global GPU performance flags ONCE at startup.
    cudnn.benchmark  — picks the fastest conv kernel for your GPU/input size.
    allow_tf32       — uses TF32 matmuls on Ampere/Ada GPUs (free 2-3x speedup).
    """
    if not torch.cuda.is_available():
        return
    if CFG.CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True
    if CFG.USE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True


# =============================================================================
# Seed population
# =============================================================================

def create_seed_population(num_seeds: int) -> list:
    """
    Build the initial seed population of architectures.

    When USE_PRETRAINED_BACKBONE=False (default):
        Returns a list of ArchitectureGraph objects (BaseNet architecture).
        The LEMONADE loop wraps them in Individual objects.

    When USE_PRETRAINED_BACKBONE=True:
        Returns a list of Individual objects whose .model attribute already
        has torchvision ResNet18 ImageNet weights loaded into it.
        The LEMONADE loop detects Individual objects and uses them directly,
        so the pretrained weights are serialised by _build_payloads and
        passed to the training workers before gen-0 fine-tuning starts.
    """
    num_classes = CFG.get_num_classes()
    input_size  = CFG.get_input_size()

    # ------------------------------------------------------------------
    # Choose base architecture
    # ------------------------------------------------------------------
    if CFG.USE_PRETRAINED_BACKBONE:
        from models.resnet_graph import build_resnet18_graph
        logger.info(
            "USE_PRETRAINED_BACKBONE=True: building ResNet18 seed graphs "
            "for %s (%d classes)", CFG.TARGET_DATASET, num_classes
        )
        base_graph, _ = build_resnet18_graph(
            num_classes  = num_classes,
            dataset_type = CFG.TARGET_DATASET,
        )
    else:
        logger.info(
            "Building %d BaseNet seed architectures for %s (%d classes)",
            num_seeds, CFG.TARGET_DATASET, num_classes,
        )
        base_graph = build_basenet_graph(
            num_classes  = num_classes,
            dataset_type = CFG.TARGET_DATASET,
        )

    # ------------------------------------------------------------------
    # Generate seed graphs (apply random morphisms for diversity)
    # ------------------------------------------------------------------
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
                    input_size=input_size,
                )
                if cheap.get("params", 0) <= CFG.MAX_PARAMS:
                    graphs.append(new_graph)
                    break
            except Exception:
                continue
        else:
            graphs.append(copy.deepcopy(base_graph))

    logger.info("Seed graphs ready: %d architectures", len(graphs))

    # ------------------------------------------------------------------
    # Pretrained backbone: load ImageNet weights into each seed
    # ------------------------------------------------------------------
    if CFG.USE_PRETRAINED_BACKBONE:
        from models.resnet_graph import load_pretrained_resnet18_weights
        from architectures.compiler import CompiledModel

        inds = []
        for i, g in enumerate(graphs):
            ind   = Individual(g)
            model = ind.build_model(input_shape=input_size)

            # The node_map is attached to each graph by build_resnet18_graph
            # and propagated to child graphs by deepcopy in random_operator.
            node_map = getattr(g, '_pretrained_node_map', None)
            if node_map is not None:
                load_pretrained_resnet18_weights(model, node_map, num_classes)
            else:
                logger.warning(
                    "Seed %d has no _pretrained_node_map — "
                    "using random initialisation", i
                )

            # ind.model is now set with pretrained weights.
            # _build_payloads in lemonade_full.py will save this state dict
            # to a temp file and pass it to the training worker.
            inds.append(ind)

        logger.info(
            "Pretrained ResNet18 weights loaded into %d seed models", len(inds)
        )
        # Return Individual objects (not just graphs).
        # run_lemonade detects this and uses them directly.
        return inds

    # Standard path: return graphs only
    return graphs


# =============================================================================
# Main
# =============================================================================

def main():
    # ------------------------------------------------------------------
    # Validate dataset
    # ------------------------------------------------------------------
    valid_datasets = {"CIFAR-10", "CIFAR-100", "TINY-IMAGENET", "IMAGENET"}
    if CFG.TARGET_DATASET not in valid_datasets:
        raise ValueError(
            f"Unknown TARGET_DATASET '{CFG.TARGET_DATASET}'. "
            f"Valid choices: {valid_datasets}"
        )

    num_classes = CFG.get_num_classes()
    input_size  = CFG.get_input_size()

    # ------------------------------------------------------------------
    # Auto-adjust batch size for large images
    # ------------------------------------------------------------------
    if CFG.TARGET_DATASET == "IMAGENET" and CFG.BATCH_SIZE > 32:
        CFG.BATCH_SIZE = 32
        print("  [INFO] IMAGENET → BATCH_SIZE reduced to 32")
    elif CFG.TARGET_DATASET == "TINY-IMAGENET" and CFG.BATCH_SIZE > 256:
        CFG.BATCH_SIZE = 256

    # ------------------------------------------------------------------
    # Warn if LR is too high for fine-tuning
    # ------------------------------------------------------------------
    if CFG.USE_PRETRAINED_BACKBONE and CFG.INIT_LR > 0.01:
        print(
            f"\n  [WARN] USE_PRETRAINED_BACKBONE=True but INIT_LR={CFG.INIT_LR:.4f} "
            f"seems high for fine-tuning.\n"
            f"         Consider setting INIT_LR=0.001 and CHILD_LR=0.0005 in config.py\n"
            f"         Also consider INIT_EPOCHS=8 and CHILD_EPOCHS=5 for faster runs.\n"
        )

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device   = "cuda"
        num_gpus = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9

        print(f"\n  GPU detected  : {gpu_name}  ({gpu_mem:.1f} GB VRAM)")
        if num_gpus > 1 and CFG.USE_MULTI_GPU:
            gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
            print(f"  Multi-GPU     : {num_gpus} GPUs → DataParallel")
            for i, name in enumerate(gpu_names):
                print(f"                  GPU {i}: {name}")
        elif num_gpus > 1:
            print(f"  Multi-GPU     : {num_gpus} GPUs available (USE_MULTI_GPU=False)")
        else:
            print(f"  Multi-GPU     : disabled (only 1 GPU)")

        print(f"  AMP (fp16)    : {'ENABLED'  if CFG.USE_AMP          else 'DISABLED'}")
        print(f"  cuDNN bench   : {'ENABLED'  if CFG.CUDNN_BENCHMARK   else 'DISABLED'}")
        print(f"  TF32          : {'ENABLED'  if CFG.USE_TF32          else 'DISABLED'}")
        print(f"  DL workers    : {CFG.NUM_DATALOADER_WORKERS}")
        print(f"  Pretrained    : {'ResNet18 (ImageNet)' if CFG.USE_PRETRAINED_BACKBONE else 'No (BaseNet from scratch)'}")

        _apply_gpu_speed_settings()
    else:
        device = "cpu"
        n_cpu  = os.cpu_count() or 1
        print(f"\n  No GPU — using CPU ({n_cpu} cores, parallel training)")
        CFG.USE_AMP          = False
        CFG.CUDNN_BENCHMARK  = False
        CFG.USE_TF32         = False
        CFG.USE_MULTI_GPU    = False

    print(f"  Dataset       : {CFG.TARGET_DATASET}")
    print(f"  Classes       : {num_classes}")
    print(f"  Input size    : {input_size}")
    print(f"  Batch size    : {CFG.BATCH_SIZE}")
    print(f"  Init LR       : {CFG.INIT_LR}  |  Child LR: {CFG.CHILD_LR}")
    print(f"  Init epochs   : {CFG.INIT_EPOCHS}  |  Child epochs: {CFG.CHILD_EPOCHS}\n")

    logger.info(
        "Starting LEMONADE NAS | dataset=%s | device=%s | pretrained=%s",
        CFG.TARGET_DATASET, device, CFG.USE_PRETRAINED_BACKBONE,
    )

    # ------------------------------------------------------------------
    # Timestamped output directory
    # ------------------------------------------------------------------
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"results_{CFG.TARGET_DATASET}_{ts}"
    os.makedirs(run_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    from data.loader_factory import get_loaders
    loaders      = get_loaders(CFG, split_test=True)
    train_loader = loaders[0]
    val_loader   = loaders[1]
    test_loader  = loaders[2] if len(loaders) > 2 else None

    # ------------------------------------------------------------------
    # Seed population
    # ------------------------------------------------------------------
    init_population = create_seed_population(CFG.NUM_SEEDS)

    # ------------------------------------------------------------------
    # Run LEMONADE
    # ------------------------------------------------------------------
    final_population, history = run_lemonade(
        init_graphs  = init_population,   # list of graphs OR list of Individuals
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
    try:
        from utils.plot import plot_all_pairs, plot_3d_pareto, plot_convergence
        plot_all_pairs(history, cheap_objectives=CFG.CHEAP_OBJECTIVES,
                       save_dir=plot_dir)
        plot_3d_pareto(history, save_dir=plot_dir)
        plot_convergence(history, save_dir=plot_dir)
    except Exception as e:
        logger.warning("Plotting failed (non-fatal): %s", e)

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
            try:
                from train.evaluate import evaluate_accuracy
                te = evaluate_accuracy(ind.model, test_loader, device=device)
            except Exception:
                pass

        p_str  = f"{p:>12,}"       if p  is not None else f"{'?':>12}"
        fl_str = f"{int(fl):>14,}" if fl is not None else f"{'?':>14}"
        ve_str = f"{ve:.4f}"       if ve is not None else "?"
        te_str = f"{te:.4f}"       if te is not None else "N/A"
        print(f"  [{i}] params={p_str}  flops={fl_str}  "
              f"val_err={ve_str}  test_err={te_str}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Windows requires "spawn" start method for safe multiprocessing.
    try:
        multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    main()
    
# # main.py
# # =============================================================================
# # Entry point for LEMONADE NAS.
# # ALL hyperparameters live in config.py — edit them there.
# #
# # Windows / Local GPU usage:
# #   Run:  python main.py
# #   (Make sure you installed PyTorch with the correct CUDA version first —
# #    see requirements.txt for instructions)
# #
# # Google Colab usage:
# #   Run: !python main.py  (after uploading the project folder)
# # =============================================================================

# import os
# import multiprocessing

# # ---- Thread-count lock (must happen before any torch import) ----
# # We lock these to 1 so that PyTorch's internal OpenMP/MKL threads don't
# # fight with DataLoader workers for CPU cores.
# os.environ.setdefault("OMP_NUM_THREADS",        "1")
# os.environ.setdefault("MKL_NUM_THREADS",        "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS",   "1")
# os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
# os.environ.setdefault("NUMEXPR_NUM_THREADS",    "1")

# import warnings
# warnings.filterwarnings("ignore")

# import copy
# import pickle
# import datetime
# import torch

# from config import CFG
# from evolution.lemonade_full import run_lemonade
# from evolution.operators import random_operator
# from evolution.individual import Individual
# from models.basenet import build_basenet_graph
# from utils.logger import get_logger

# logger = get_logger("main", logfile="logs/main.log")


# # =============================================================================
# # GPU speed settings — applied once before any training starts
# # =============================================================================

# def _apply_gpu_speed_settings():
#     """
#     Apply global GPU performance flags.
#     Call this ONCE at startup, before any model is built or trained.

#     cudnn.benchmark  — auto-selects the fastest conv kernel for your GPU
#                        and input size.  2-4x speedup on fixed-size inputs
#                        (Tiny ImageNet 64x64 is always fixed).

#     allow_tf32       — on Ampere/Ada GPUs (RTX 30/40 series) this uses
#                        TF32 matmuls which are 2-3x faster than FP32 with
#                        negligible accuracy loss.  Older GPUs ignore it.
#     """
#     if not torch.cuda.is_available():
#         return

#     if CFG.CUDNN_BENCHMARK:
#         torch.backends.cudnn.benchmark = True

#     if CFG.USE_TF32:
#         # Matmul (Linear / Attention layers)
#         torch.backends.cuda.matmul.allow_tf32 = True
#         # Convolutions
#         torch.backends.cudnn.allow_tf32 = True


# # =============================================================================
# # Seed population
# # =============================================================================

# def create_seed_population(num_seeds: int) -> list:
#     num_classes = CFG.get_num_classes()
#     input_size  = CFG.get_input_size()

#     logger.info("Building %d seed architectures for %s (%d classes, input=%s)",
#                 num_seeds, CFG.TARGET_DATASET, num_classes, input_size)

#     base_graph = build_basenet_graph(
#         num_classes=num_classes,
#         dataset_type=CFG.TARGET_DATASET,
#     )
#     graphs = [base_graph]

#     for _ in range(num_seeds - 1):
#         for _ in range(15):
#             tmp = Individual(copy.deepcopy(base_graph))
#             new_graph, _, _ = random_operator(tmp)
#             if new_graph is None:
#                 continue
#             new_ind = Individual(new_graph)
#             try:
#                 cheap = new_ind.evaluate_cheap(
#                     objective_keys=CFG.CHEAP_OBJECTIVES,
#                     input_size=input_size,
#                 )
#                 if cheap.get("params", 0) <= CFG.MAX_PARAMS:
#                     graphs.append(new_graph)
#                     break
#             except Exception:
#                 continue
#         else:
#             graphs.append(copy.deepcopy(base_graph))

#     logger.info("Seed population ready: %d architectures", len(graphs))
#     return graphs


# # =============================================================================
# # Main
# # =============================================================================

# def main():
#     # ------------------------------------------------------------------
#     # Validate dataset choice early
#     # ------------------------------------------------------------------
#     valid_datasets = {"CIFAR-10", "CIFAR-100", "TINY-IMAGENET", "IMAGENET"}
#     if CFG.TARGET_DATASET not in valid_datasets:
#         raise ValueError(
#             f"Unknown TARGET_DATASET '{CFG.TARGET_DATASET}'. "
#             f"Valid choices: {valid_datasets}"
#         )

#     num_classes = CFG.get_num_classes()
#     input_size  = CFG.get_input_size()

#     # ------------------------------------------------------------------
#     # Auto-adjust batch size for large images
#     # ------------------------------------------------------------------
#     if CFG.TARGET_DATASET == "IMAGENET" and CFG.BATCH_SIZE > 32:
#         CFG.BATCH_SIZE = 32
#         print("  [INFO] IMAGENET detected — BATCH_SIZE reduced to 32 for GPU safety")
#     elif CFG.TARGET_DATASET == "TINY-IMAGENET" and CFG.BATCH_SIZE > 256:
#         CFG.BATCH_SIZE = 256

#     # ------------------------------------------------------------------
#     # Device detection with detailed info
#     # ------------------------------------------------------------------
#     if torch.cuda.is_available():
#         device   = "cuda"
#         gpu_name = torch.cuda.get_device_name(0)
#         gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
#         print(f"\n  GPU detected : {gpu_name}  ({gpu_mem:.1f} GB VRAM)")
#         print(f"  AMP (fp16)   : {'ENABLED'  if CFG.USE_AMP          else 'DISABLED'}")
#         print(f"  cuDNN bench  : {'ENABLED'  if CFG.CUDNN_BENCHMARK   else 'DISABLED'}")
#         print(f"  TF32         : {'ENABLED'  if CFG.USE_TF32          else 'DISABLED'}")
#         print(f"  DL workers   : {CFG.NUM_DATALOADER_WORKERS}")
#         print(f"  Training     : SEQUENTIAL on GPU (faster than multiprocessing)")

#         # Apply global GPU speed settings
#         _apply_gpu_speed_settings()

#     else:
#         device = "cpu"
#         n_cpu  = os.cpu_count() or 1
#         print(f"\n  No GPU found — using CPU ({n_cpu} cores, parallel training)")
#         CFG.USE_AMP          = False   # AMP has no effect on CPU
#         CFG.CUDNN_BENCHMARK  = False
#         CFG.USE_TF32         = False

#     print(f"  Dataset      : {CFG.TARGET_DATASET}")
#     print(f"  Classes      : {num_classes}")
#     print(f"  Input size   : {input_size}")
#     print(f"  Batch size   : {CFG.BATCH_SIZE}")
#     print(f"  Init LR      : {CFG.INIT_LR}")
#     print(f"  Init epochs  : {CFG.INIT_EPOCHS}")
#     print(f"  Child epochs : {CFG.CHILD_EPOCHS}\n")

#     logger.info("Starting LEMONADE NAS | dataset=%s | device=%s | input_size=%s",
#                 CFG.TARGET_DATASET, device, input_size)
#     logger.info("Config: %s", CFG)

#     # ------------------------------------------------------------------
#     # Timestamped output directory
#     # ------------------------------------------------------------------
#     ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_dir = f"results_{CFG.TARGET_DATASET}_{ts}"
#     os.makedirs(run_dir, exist_ok=True)
#     logger.info("Run directory: %s", run_dir)

#     # ------------------------------------------------------------------
#     # Data loaders for main process
#     # ------------------------------------------------------------------
#     from data.loader_factory import get_loaders
#     loaders      = get_loaders(CFG, split_test=True)
#     train_loader = loaders[0]
#     val_loader   = loaders[1]
#     test_loader  = loaders[2] if len(loaders) > 2 else None

#     # ------------------------------------------------------------------
#     # Seed population
#     # ------------------------------------------------------------------
#     init_graphs = create_seed_population(CFG.NUM_SEEDS)

#     # ------------------------------------------------------------------
#     # Run LEMONADE
#     # ------------------------------------------------------------------
#     final_population, history = run_lemonade(
#         init_graphs  = init_graphs,
#         cfg          = CFG,
#         train_loader = train_loader,
#         val_loader   = val_loader,
#         device       = device,
#         run_dir      = run_dir,
#     )

#     # ------------------------------------------------------------------
#     # Save history + config
#     # ------------------------------------------------------------------
#     history_dir = os.path.join(run_dir, "history")
#     os.makedirs(history_dir, exist_ok=True)
#     with open(os.path.join(history_dir, "history.pkl"), "wb") as f:
#         pickle.dump(history, f)
#     with open(os.path.join(run_dir, "config.pkl"), "wb") as f:
#         pickle.dump(CFG, f)
#     logger.info("History + config saved to %s", run_dir)

#     # ------------------------------------------------------------------
#     # Plots
#     # ------------------------------------------------------------------
#     plot_dir = os.path.join(run_dir, "plots")
#     os.makedirs(plot_dir, exist_ok=True)
#     try:
#         from utils.plot import plot_all_pairs, plot_3d_pareto, plot_convergence
#         plot_all_pairs(history, cheap_objectives=CFG.CHEAP_OBJECTIVES,
#                        save_dir=plot_dir)
#         plot_3d_pareto(history, save_dir=plot_dir)
#         plot_convergence(history, save_dir=plot_dir)
#     except Exception as e:
#         logger.warning("Plotting failed (non-fatal): %s", e)

#     # ------------------------------------------------------------------
#     # Final summary
#     # ------------------------------------------------------------------
#     print("\n" + "=" * 60)
#     print(f"  LEMONADE COMPLETE — {CFG.TARGET_DATASET}")
#     print(f"  Results → {run_dir}")
#     print("=" * 60)

#     for i, ind in enumerate(final_population):
#         ve = ind.f_exp.get("val_error")  if ind.f_exp   else None
#         p  = ind.f_cheap.get("params")   if ind.f_cheap else None
#         fl = ind.f_cheap.get("flops")    if ind.f_cheap else None

#         te = None
#         if test_loader is not None and ind.model is not None:
#             try:
#                 from train.evaluate import evaluate_accuracy
#                 te = evaluate_accuracy(ind.model, test_loader, device=device)
#             except Exception:
#                 pass

#         p_str  = f"{p:>12,}"       if p  is not None else f"{'?':>12}"
#         fl_str = f"{int(fl):>14,}" if fl is not None else f"{'?':>14}"
#         ve_str = f"{ve:.4f}"       if ve is not None else "?"
#         te_str = f"{te:.4f}"       if te is not None else "N/A"
#         print(f"  [{i}] params={p_str}  flops={fl_str}  "
#               f"val_err={ve_str}  test_err={te_str}")
#         logger.info("Final model %d | params=%s flops=%s val_err=%s test_err=%s",
#                     i, p_str.strip(), fl_str.strip(), ve_str, te_str)

#     print("=" * 60 + "\n")


# if __name__ == "__main__":
#     # Windows requires "spawn" start method.
#     # Linux/Mac use "fork" by default (fine for CPU workers).
#     # We use force=False so it doesn't crash if already set.
#     try:
#         multiprocessing.set_start_method("spawn", force=False)
#     except RuntimeError:
#         pass  # already set — ignore

#     main()