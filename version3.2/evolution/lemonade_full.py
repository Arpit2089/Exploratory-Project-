# evolution/lemonade_full.py
# =============================================================================
# LEMONADE loop — GPU/CPU adaptive execution.
#
# GPU strategy (CUDA available):
#   Models are trained ONE AT A TIME on the GPU in the main process.
#   ┌─────────────────────────────────────────────────────────┐
#   │  Why sequential on GPU beats parallel on CPU:           │
#   │  • A single GPU forward/backward is 20-100x faster      │
#   │    than a CPU pass — no parallelism needed.             │
#   │  • No pickle/IPC overhead (models stay in main proc).   │
#   │  • Dataset is loaded ONCE per generation, not once       │
#   │    per model per worker.                                 │
#   │  • CUDA cannot be forked to child processes safely.      │
#   └─────────────────────────────────────────────────────────┘
#
# CPU strategy (no CUDA):
#   Models are trained IN PARALLEL using ProcessPoolExecutor
#   (same as before, one worker per CPU core).
#
# The loop logic (correct LEMONADE Algorithm 1 order) is unchanged.
# =============================================================================

import os
import gc
import pickle
import time
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from evolution.individual import Individual
from evolution.pareto import pareto_front, fill_with_diversity
from evolution.sampling import KDESampler
from evolution.operators import random_operator, is_approx_op
from utils.logger import get_logger

logger       = get_logger("lemonade",        logfile="logs/lemonade.log")
error_logger = get_logger("lemonade_errors", logfile="logs/lemonade_errors.log")


# =============================================================================
# GPU path — train one child at a time in the main process
# =============================================================================

def _train_one_child_gpu(pc, cfg, train_epochs, train_lr,
                         device, train_loader, val_loader):
    """
    Train a single child on the GPU (main process, no pickling).

    Parameters
    ----------
    pc           : bytes — pickled payload tuple
    cfg          : NASConfig
    train_epochs : int
    train_lr     : float
    device       : "cuda" or "cuda:N"
    train_loader : DataLoader — shared across all children this generation
    val_loader   : DataLoader — shared across all children this generation

    Returns
    -------
    dict with keys: status, ind (Individual), error
    """
    import torch
    from objectives.cheap import clean_state_dict

    try:
        child_graph, child_sd_path, parent_graph, parent_sd_path, is_approx = \
            pickle.loads(pc)

        # ------------------------------------------------------------------
        # 1. Build child model ONCE (stays in main process — no CUDA fork)
        # ------------------------------------------------------------------
        child       = Individual(child_graph)
        child_model = child.build_model()

        # ------------------------------------------------------------------
        # 2. Load Lamarckian inherited weights
        # ------------------------------------------------------------------
        if child_sd_path and os.path.exists(child_sd_path):
            try:
                sd = torch.load(child_sd_path, map_location="cpu",
                                weights_only=True)
                child_model.load_state_dict(clean_state_dict(sd), strict=False)
            except Exception:
                pass  # random init is fine
            finally:
                try:
                    os.remove(child_sd_path)
                except OSError:
                    pass

        # ------------------------------------------------------------------
        # 3. Build teacher if ANM child (for distillation)
        # ------------------------------------------------------------------
        teacher_model = None
        if is_approx and parent_graph is not None:
            try:
                teacher       = Individual(parent_graph)
                teacher_model = teacher.build_model()
                if parent_sd_path and os.path.exists(parent_sd_path):
                    sd = torch.load(parent_sd_path, map_location="cpu",
                                    weights_only=True)
                    teacher_model.load_state_dict(clean_state_dict(sd),
                                                  strict=False)
                teacher_model.eval()
                for p in teacher_model.parameters():
                    p.requires_grad = False
            except Exception:
                teacher_model = None

        # ------------------------------------------------------------------
        # 4. Evaluate cheap objectives (graph-based, no GPU needed)
        # ------------------------------------------------------------------
        try:
            child.evaluate_cheap(
                objective_keys=cfg.CHEAP_OBJECTIVES,
                input_size=(1, 3, 32, 32),
            )
        except Exception:
            child.f_cheap = {k: 0 for k in cfg.CHEAP_OBJECTIVES}

        # ------------------------------------------------------------------
        # 5a. Distillation phase (ANM only)
        #     Use DISTILL_EPOCHS + DISTILL_LR — separate from training.
        # ------------------------------------------------------------------
        if is_approx and teacher_model is not None:
            try:
                from train.distill import train_with_distillation
                train_with_distillation(
                    student_model  = child_model,
                    teacher_model  = teacher_model,
                    train_loader   = train_loader,
                    device         = device,
                    epochs         = cfg.DISTILL_EPOCHS,
                    lr             = cfg.DISTILL_LR,
                    temperature    = cfg.DISTILL_TEMPERATURE,
                    alpha          = cfg.DISTILL_ALPHA,
                    weight_decay   = cfg.WEIGHT_DECAY,
                    optimizer_name = cfg.OPTIMIZER,
                    show_progress  = False,   # no nested bars
                    use_amp        = cfg.USE_AMP,
                )
            except Exception as e:
                error_logger.error("Distillation failed %s: %s", child.id, e)
            finally:
                # Free teacher from GPU immediately after distillation
                teacher_model.cpu()
                del teacher_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        # ------------------------------------------------------------------
        # 5b. Standard training (all children)
        # ------------------------------------------------------------------
        try:
            from train.trainer import train_model
            train_model(
                model          = child_model,
                train_loader   = train_loader,
                device         = device,
                epochs         = train_epochs,
                lr             = train_lr,
                weight_decay   = cfg.WEIGHT_DECAY,
                optimizer_name = cfg.OPTIMIZER,
                show_progress  = False,
                use_amp        = cfg.USE_AMP,
            )
        except Exception as e:
            error_logger.error("Training failed %s: %s", child.id, e)

        # ------------------------------------------------------------------
        # 6. Evaluate (AMP inference)
        # ------------------------------------------------------------------
        try:
            from train.evaluate import evaluate_accuracy
            val_error = evaluate_accuracy(child_model, val_loader,
                                          device=device,
                                          use_amp=cfg.USE_AMP)
        except Exception:
            val_error = 1.0

        child.f_exp = {"val_error": val_error}

        # ------------------------------------------------------------------
        # 7. Move model back to CPU to free GPU memory for the next child
        # ------------------------------------------------------------------
        child_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {"status": "ok", "ind": child}

    except Exception as exc:
        return {
            "status":    "error",
            "error":     str(exc),
            "traceback": traceback.format_exc(),
        }


def _sequential_train_gpu(payloads, cfg, train_epochs, train_lr,
                           device, desc):
    """
    Train each child sequentially on the GPU.
    Data loaders are created ONCE and shared across all children.
    """
    import torch
    from data.loader_factory import get_loaders

    # Create data loaders once for the whole batch
    # Use full num_workers + pin_memory for GPU
    loaders      = get_loaders(cfg, split_test=True)
    train_loader = loaders[0]
    val_loader   = loaders[1]

    trained = []
    n       = len(payloads)

    if cfg.SHOW_PROGRESS_BAR:
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(payloads), total=n,
                            desc=desc, unit="model")
        except ImportError:
            iterator = enumerate(payloads)
    else:
        iterator = enumerate(payloads)

    for i, (idx, pc) in iterator:
        result = _train_one_child_gpu(
            pc, cfg, train_epochs, train_lr,
            device, train_loader, val_loader,
        )
        if result["status"] == "ok":
            trained.append(result["ind"])
        else:
            error_logger.error("%s child %d/%d error: %s\n%s",
                               desc, i + 1, n,
                               result.get("error"),
                               result.get("traceback", ""))

    return trained


# =============================================================================
# CPU path — parallel training via ProcessPoolExecutor (unchanged from before)
# =============================================================================

def _worker_train_child(idx, pickled_payload, cfg, train_epochs, train_lr):
    """
    Worker function for CPU-parallel training.
    Runs in a subprocess — must not use CUDA.
    """
    import os, gc, pickle, time, traceback
    import torch

    os.environ["TQDM_DISABLE"] = "1"
    torch.set_num_threads(1)

    try:
        start = time.time()
        child_graph, child_sd_path, parent_graph, parent_sd_path, is_approx = \
            pickle.loads(pickled_payload)

        from objectives.cheap import clean_state_dict
        from evolution.individual import Individual as _Ind

        child       = _Ind(child_graph)
        child_model = child.build_model()

        if child_sd_path and os.path.exists(child_sd_path):
            try:
                sd = torch.load(child_sd_path, map_location="cpu",
                                weights_only=True)
                child_model.load_state_dict(clean_state_dict(sd), strict=False)
            except Exception:
                pass
            finally:
                try:
                    os.remove(child_sd_path)
                except OSError:
                    pass

        teacher_model = None
        if is_approx and parent_graph is not None:
            try:
                teacher       = _Ind(parent_graph)
                teacher_model = teacher.build_model()
                if parent_sd_path and os.path.exists(parent_sd_path):
                    sd = torch.load(parent_sd_path, map_location="cpu",
                                    weights_only=True)
                    teacher_model.load_state_dict(clean_state_dict(sd),
                                                  strict=False)
                teacher_model.eval()
                for p in teacher_model.parameters():
                    p.requires_grad = False
            except Exception:
                teacher_model = None

        try:
            child.evaluate_cheap(
                objective_keys=cfg.CHEAP_OBJECTIVES,
                input_size=(1, 3, 32, 32),
            )
        except Exception:
            child.f_cheap = {k: 0 for k in cfg.CHEAP_OBJECTIVES}

        from data.loader_factory import get_loaders_for_worker
        train_loader_w, val_loader_w = get_loaders_for_worker(cfg)

        device = "cpu"  # workers MUST use CPU

        if is_approx and teacher_model is not None:
            try:
                from train.distill import train_with_distillation
                train_with_distillation(
                    student_model  = child_model,
                    teacher_model  = teacher_model,
                    train_loader   = train_loader_w,
                    device         = device,
                    epochs         = cfg.DISTILL_EPOCHS,
                    lr             = cfg.DISTILL_LR,
                    temperature    = cfg.DISTILL_TEMPERATURE,
                    alpha          = cfg.DISTILL_ALPHA,
                    weight_decay   = cfg.WEIGHT_DECAY,
                    optimizer_name = cfg.OPTIMIZER,
                    show_progress  = False,
                    use_amp        = False,  # no AMP on CPU
                )
            except Exception as e:
                error_logger.error("Distillation failed %s: %s", child.id, e)
            finally:
                del teacher_model
                gc.collect()

        try:
            from train.trainer import train_model
            train_model(
                model          = child_model,
                train_loader   = train_loader_w,
                device         = device,
                epochs         = train_epochs,
                lr             = train_lr,
                weight_decay   = cfg.WEIGHT_DECAY,
                optimizer_name = cfg.OPTIMIZER,
                show_progress  = False,
                use_amp        = False,
            )
        except Exception as e:
            error_logger.error("Training failed %s: %s", child.id, e)

        try:
            from train.evaluate import evaluate_accuracy
            val_error = evaluate_accuracy(child_model, val_loader_w,
                                          device=device, use_amp=False)
        except Exception:
            val_error = 1.0

        child.f_exp = {"val_error": val_error}

        # Save trained weights to disk for IPC back to main process
        sd_path = os.path.join(
            tempfile.gettempdir(),
            f"nas_cpu_{child.id}_{int(time.time() * 1000)}.pt",
        )
        torch.save(child_model.state_dict(), sd_path)

        child.model = None
        del child_model
        gc.collect()

        return {
            "idx":             idx,
            "status":          "ok",
            "pickled_child":   pickle.dumps(child),
            "trained_sd_path": sd_path,
            "duration":        time.time() - start,
        }

    except Exception as exc:
        return {
            "idx":       idx,
            "status":    "error",
            "error":     str(exc),
            "traceback": traceback.format_exc(),
        }


def _parallel_train_cpu(payloads, cfg, train_epochs, train_lr, desc):
    """
    Train children in parallel on CPU cores (used when no GPU is available).
    """
    import torch
    cpu_count   = max(1, (os.cpu_count() or 2) - 1)
    max_workers = min(cpu_count, len(payloads))

    trained = []

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(_worker_train_child, idx, pc, cfg,
                       train_epochs, train_lr): idx
            for idx, pc in payloads
        }

        if cfg.SHOW_PROGRESS_BAR:
            try:
                from tqdm import tqdm
                iterator = tqdm(as_completed(futures), total=len(futures),
                                desc=desc, unit="model")
            except ImportError:
                iterator = as_completed(futures)
        else:
            iterator = as_completed(futures)

        for fut in iterator:
            try:
                result = fut.result()
            except Exception as e:
                error_logger.error("%s future exception: %s", desc, e)
                continue

            if result.get("status") != "ok":
                error_logger.error("%s worker error [idx=%s]: %s\n%s",
                                   desc, result.get("idx"),
                                   result.get("error"),
                                   result.get("traceback", ""))
                continue

            try:
                ind     = pickle.loads(result["pickled_child"])
                sd_path = result.get("trained_sd_path")
                if sd_path and os.path.exists(sd_path):
                    model = ind.build_model()
                    model.load_state_dict(
                        torch.load(sd_path, map_location="cpu",
                                   weights_only=True),
                        strict=False,
                    )
                    try:
                        os.remove(sd_path)
                    except OSError:
                        pass
                trained.append(ind)
            except Exception as e:
                error_logger.error("Deserialise error: %s", e)

    return trained


# =============================================================================
# Unified dispatcher — chooses GPU or CPU path automatically
# =============================================================================

def _train_all(payloads, cfg, train_epochs, train_lr, desc, device):
    """
    Route to GPU-sequential or CPU-parallel training based on device.
    """
    if device.startswith("cuda"):
        return _sequential_train_gpu(
            payloads, cfg, train_epochs, train_lr, device, desc
        )
    else:
        return _parallel_train_cpu(
            payloads, cfg, train_epochs, train_lr, desc
        )


# =============================================================================
# Shared helpers
# =============================================================================

def _build_payloads(items, temp_dir, gen):
    """
    Serialise (child, parent, is_approx) triples into pickled payloads.
    Returns (payloads, paths_to_clean_later).
    """
    import torch
    payloads          = []
    parent_temp_paths = []

    for idx, (child, parent, approx) in enumerate(items):
        try:
            child_sd_path  = None
            parent_sd_path = None

            if child.model is not None:
                child_sd_path = os.path.join(
                    temp_dir, f"child_{child.id}_{gen}.pt"
                )
                torch.save(child.model.state_dict(), child_sd_path)

            if parent is not None and parent.model is not None:
                parent_sd_path = os.path.join(
                    temp_dir, f"parent_{parent.id}_for_{child.id}_{gen}.pt"
                )
                torch.save(parent.model.state_dict(), parent_sd_path)
                parent_temp_paths.append(parent_sd_path)

            pc = pickle.dumps((
                child.graph, child_sd_path,
                parent.graph if parent is not None else None,
                parent_sd_path,
                approx,
            ))
            payloads.append((idx, pc))

        except Exception as e:
            error_logger.error("Serialise error for child %s: %s", child.id, e)

    return payloads, parent_temp_paths


def _save_generation_models(population, gen, models_dir):
    import torch
    gen_dir = os.path.join(models_dir, f"gen_{gen:03d}")
    os.makedirs(gen_dir, exist_ok=True)

    history_entry = []
    for ind in population:
        record = {
            "id":         ind.id,
            "params":     ind.f_cheap.get("params")    if ind.f_cheap else None,
            "flops":      ind.f_cheap.get("flops")     if ind.f_cheap else None,
            "val_error":  ind.f_exp.get("val_error")   if ind.f_exp   else None,
            "model_path": None,
            "graph_path": None,
        }
        if ind.model is not None:
            try:
                wpath = os.path.join(gen_dir, f"{ind.id}_weights.pt")
                torch.save(ind.model.state_dict(), wpath)
                record["model_path"] = wpath
            except Exception:
                pass
        try:
            gpath = os.path.join(gen_dir, f"{ind.id}_graph.pkl")
            with open(gpath, "wb") as f:
                pickle.dump(ind.graph, f)
            record["graph_path"] = gpath
        except Exception:
            pass
        history_entry.append(record)

    return history_entry


def _print_summary(gen, population):
    rows = []
    for ind in population:
        p = ind.f_cheap.get("params")    if ind.f_cheap else None
        f = ind.f_cheap.get("flops")     if ind.f_cheap else None
        v = ind.f_exp.get("val_error")   if ind.f_exp   else None
        rows.append((p, f, v))
    rows.sort(key=lambda r: (
        r[2] if r[2] is not None else 1.0,
        r[0] if r[0] is not None else float("inf"),
    ))
    print(f"\n{'='*65}")
    print(f"  Generation {gen}  |  Pareto population: {len(rows)} models")
    print(f"  {'params':>10}  {'flops':>12}  {'val_error':>10}")
    for p, f, v in rows[:8]:
        ps = f"{p:>10,}"      if p is not None else f"{'?':>10}"
        fs = f"{int(f):>12,}" if f is not None else f"{'?':>12}"
        vs = f"{v:.4f}"       if v is not None else "?"
        print(f"  {ps}  {fs}  {vs:>10}")
    print(f"{'='*65}\n")


# =============================================================================
# Main LEMONADE entry point
# =============================================================================

def run_lemonade(init_graphs, cfg, train_loader, val_loader, device, run_dir):
    """
    Run the LEMONADE NAS algorithm.

    Parameters
    ----------
    init_graphs  : list[ArchitectureGraph]
    cfg          : NASConfig
    train_loader : DataLoader (main process — GPU path ignores this internally,
                   creates its own with proper settings per generation)
    val_loader   : DataLoader
    device       : "cpu" | "cuda" | "cuda:N"
    run_dir      : output directory

    Returns
    -------
    (final_population: list[Individual], history: dict)
    """
    import torch

    use_gpu = device.startswith("cuda") and torch.cuda.is_available()
    logger.info(
        "LEMONADE start: gens=%d N_pc=%d N_ac=%d "
        "init_ep=%d child_ep=%d distill_ep=%d "
        "device=%s amp=%s",
        cfg.GENERATIONS, cfg.N_CHILDREN, cfg.N_ACCEPT,
        cfg.INIT_EPOCHS, cfg.CHILD_EPOCHS, cfg.DISTILL_EPOCHS,
        device, cfg.USE_AMP and use_gpu,
    )

    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    temp_dir = tempfile.gettempdir()
    history  = {}
    sampler  = KDESampler(base_bandwidth=cfg.KDE_BANDWIDTH)

    # ------------------------------------------------------------------
    # Initialise population with cheap objectives
    # ------------------------------------------------------------------
    population = [Individual(g) for g in init_graphs]
    for ind in population:
        try:
            ind.evaluate_cheap(
                objective_keys=cfg.CHEAP_OBJECTIVES,
                input_size=(1, 3, 32, 32),
            )
        except Exception as e:
            ind.f_cheap = {k: 0 for k in cfg.CHEAP_OBJECTIVES}
            error_logger.error("Cheap eval failed for seed %s: %s", ind.id, e)

    # ------------------------------------------------------------------
    # Generation 0 — train seed population
    # ------------------------------------------------------------------
    logger.info("=== Generation 0: Training %d seed models ===", len(population))

    # Build payloads: each seed trains from scratch (no parent)
    seed_triples = [(ind, None, False) for ind in population]
    payloads_g0, paths_g0 = _build_payloads(seed_triples, temp_dir, gen=0)

    if payloads_g0:
        trained_seeds = _train_all(
            payloads_g0, cfg,
            train_epochs=cfg.INIT_EPOCHS,
            train_lr=cfg.INIT_LR,
            desc="Gen 0 | Seed Training",
            device=device,
        )
        if trained_seeds:
            population = trained_seeds

    # Clean up seed temp files
    for path in paths_g0:
        try:
            os.remove(path)
        except OSError:
            pass

    population = pareto_front(population)
    population = fill_with_diversity(population, population, cfg.MIN_POP)
    history[0] = _save_generation_models(population, 0, models_dir)
    _print_summary(0, population)

    # ------------------------------------------------------------------
    # Generations 1 .. GENERATIONS
    # ------------------------------------------------------------------
    for gen in range(1, cfg.GENERATIONS + 1):
        gen_start = time.time()
        logger.info("===== Generation %d =====", gen)

        child_epochs = (
            cfg.CHILD_EPOCHS + int(gen / 5)
            if cfg.EPOCH_PROGRESSION
            else cfg.CHILD_EPOCHS
        )

        parent_temp_paths = []

        try:
            # --------------------------------------------------------------
            # STEP 1: KDE fit on current population (cheap objectives only)
            # --------------------------------------------------------------
            sampler.fit(population,
                        objective_keys=cfg.CHEAP_OBJECTIVES,
                        generation=gen)

            # --------------------------------------------------------------
            # STEP 2: Sample npc parents (inverse KDE density)
            # --------------------------------------------------------------
            parents = sampler.sample(
                population, cfg.N_CHILDREN, allow_repeats=True
            )
            if not parents:
                logger.warning("Gen %d: no parents sampled — skipping", gen)
                history[gen] = history[gen - 1]
                continue

            # --------------------------------------------------------------
            # STEP 3: Generate npc candidate children (NO training yet)
            # --------------------------------------------------------------
            candidates = []   # (child_Individual, parent_Individual, is_approx)

            for p in parents:
                try:
                    new_graph, op_name, target_info = random_operator(p)
                    if new_graph is None:
                        continue

                    approx      = is_approx_op(op_name)
                    child       = Individual(new_graph)
                    child.op_name = op_name

                    # Lamarckian weight transfer (CPU, no GPU needed)
                    try:
                        from morphisms.weights import transfer_weights
                        parent_model = p.build_model()
                        child_model  = child.build_model()
                        transfer_weights(parent_model, child_model,
                                         child.graph, op_name, target_info)
                    except Exception as e:
                        error_logger.error(
                            "Weight transfer %s→%s (%s): %s",
                            p.id, child.id, op_name, e,
                        )
                        # child has random init — still valid

                    candidates.append((child, p, approx))

                except Exception as e:
                    error_logger.error(
                        "Child gen error from %s: %s", p.id, e
                    )

            if not candidates:
                logger.warning("Gen %d: no candidates generated", gen)
                history[gen] = history[gen - 1]
                continue

            # --------------------------------------------------------------
            # STEP 4: Evaluate CHEAP objectives (free — no training)
            # --------------------------------------------------------------
            valid_candidates = []
            for child, parent, approx in candidates:
                try:
                    cheap = child.evaluate_cheap(
                        objective_keys=cfg.CHEAP_OBJECTIVES,
                        input_size=(1, 3, 32, 32),
                    )
                    if cheap.get("params", 0) <= cfg.MAX_PARAMS:
                        valid_candidates.append((child, parent, approx))
                    else:
                        logger.debug(
                            "Candidate %s exceeds MAX_PARAMS", child.id
                        )
                except Exception as e:
                    error_logger.error(
                        "Cheap eval failed for %s: %s", child.id, e
                    )

            if not valid_candidates:
                logger.warning("Gen %d: all candidates exceed MAX_PARAMS", gen)
                history[gen] = history[gen - 1]
                continue

            # --------------------------------------------------------------
            # STEP 5: KDE filter — sample nac accepted children
            # --------------------------------------------------------------
            candidate_inds = [c for c, _, _ in valid_candidates]
            sampler.fit(candidate_inds,
                        objective_keys=cfg.CHEAP_OBJECTIVES,
                        generation=gen)
            n_accept      = min(cfg.N_ACCEPT, len(valid_candidates))
            accepted_inds = sampler.sample(
                candidate_inds, n_accept, allow_repeats=False
            )
            accepted_set = {id(ind) for ind in accepted_inds}

            accepted_triples = [
                (c, p, a)
                for c, p, a in valid_candidates
                if id(c) in accepted_set
            ]

            logger.info(
                "Gen %d: %d candidates → %d accepted for training",
                gen, len(valid_candidates), len(accepted_triples),
            )

            # --------------------------------------------------------------
            # STEP 6: Train only accepted children (expensive)
            # --------------------------------------------------------------
            payloads, parent_temp_paths = _build_payloads(
                accepted_triples, temp_dir, gen
            )

            trained_children = []
            if payloads:
                trained_children = _train_all(
                    payloads, cfg,
                    train_epochs=child_epochs,
                    train_lr=cfg.CHILD_LR,
                    desc=f"Gen {gen} | Training {len(payloads)} children",
                    device=device,
                )

            # --------------------------------------------------------------
            # STEP 7: Pareto update + diversity fill
            # --------------------------------------------------------------
            if trained_children:
                combined   = population + trained_children
                new_front  = pareto_front(combined)
                population = fill_with_diversity(new_front, combined,
                                                 cfg.MIN_POP)
            else:
                logger.warning("Gen %d: no children trained successfully", gen)

        except Exception as e:
            error_logger.error(
                "Unhandled error in gen %d: %s\n%s",
                gen, e, traceback.format_exc(),
            )
        finally:
            for path in parent_temp_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass
            elapsed = time.time() - gen_start
            logger.info("Generation %d complete in %.1fs | pop=%d",
                        gen, elapsed, len(population))

        # ------------------------------------------------------------------
        # Save history + model weights for this generation
        # ------------------------------------------------------------------
        history[gen] = _save_generation_models(population, gen, models_dir)
        _print_summary(gen, population)

    return population, history