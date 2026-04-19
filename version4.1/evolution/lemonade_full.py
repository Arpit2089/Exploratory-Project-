# evolution/lemonade_full.py
# =============================================================================
# LEMONADE + ENAS  —  3-Tier Lamarckian Inheritance + REINFORCE Controller
# =============================================================================
#
# NEW METHODOLOGY (integrated from Work_to_be_done.docx + LEMONADE paper):
# ─────────────────────────────────────────────────────────────────────────
#
# 1. ENASController  (enas/controller.py)
#    REINFORCE-trained 2-layer MLP.  State = [params, flops, val_error,
#    gen/20, pop/20].  Output = softmax over 6 operators.
#    Reward ∈ {-1, -0.5, 0, +0.5, +1} based on Pareto improvement.
#    Only active on GPU path (main process); CPU workers use OP_WEIGHTS.
#
# 2. Three-Tier Lamarckian Inheritance  (enas/inheritance.py)
#
#    Tier 1 — Direct parent morphism inheritance  (existing Lamarckian)
#              morphisms/weights.py::transfer_weights()
#
#    Tier 2 — Elite pool inheritance  (NEW: enas/weight_pool.py)
#              Find structurally-similar archived individual, load strict=False.
#              Uses LCS op-sequence + cosine channel profile + log-param proximity.
#
#    Tier 3 — Supernet EMA weight bank  (NEW: enas/supernet.py)
#              All trained models contribute to a running EMA dictionary
#              keyed by (op_type, channels).  Seeds any new layer whose
#              signature exists in the bank.
#
# 3. Post-training pipeline
#    After each child trains:
#      supernet.update_from_model()   → absorb weights into bank
#      elite_pool.update()            → add to best-known set
#      controller.record_outcome()    → REINFORCE bookkeeping
#      controller.update()            → gradient step every N children
#
# GPU strategy: sequential in main process; all ENAS components live here.
# CPU strategy: parallel ProcessPoolExecutor; ENAS updated post-hoc.
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
from evolution.operators import (
    random_operator, apply_specific_operator, is_approx_op
)
from utils.logger import get_logger

logger       = get_logger("lemonade",        logfile="logs/lemonade.log")
error_logger = get_logger("lemonade_errors", logfile="logs/lemonade_errors.log")


# =============================================================================
# Shared helpers
# =============================================================================

def _maybe_load_pretrained(child_model, child_graph, child_sd_path, cfg):
    if not getattr(cfg, "USE_PRETRAINED_BACKBONE", False):
        return
    if child_sd_path is not None:
        return
    node_map = getattr(child_graph, "_pretrained_node_map", None)
    if node_map is None:
        return
    try:
        from models.resnet_graph import load_pretrained_resnet18_weights
        load_pretrained_resnet18_weights(child_model, node_map, cfg.get_num_classes())
    except Exception as e:
        logger.warning("Pretrained load failed: %s", e)


def _wrap_for_multi_gpu(model, cfg):
    import torch, torch.nn as nn
    if not getattr(cfg, "USE_MULTI_GPU", False):
        return model, False
    if not torch.cuda.is_available():
        return model, False
    if torch.cuda.device_count() <= 1:
        return model, False
    return nn.DataParallel(model), True


# =============================================================================
# GPU path — sequential, ENAS-aware
# =============================================================================

def _train_one_child_gpu(pc, cfg, train_epochs, train_lr,
                         device, train_loader, val_loader):
    import torch
    from objectives.cheap import clean_state_dict
    input_size = cfg.get_input_size()

    try:
        child_graph, child_sd_path, parent_graph, parent_sd_path, is_approx = \
            pickle.loads(pc)

        child       = Individual(child_graph)
        child_model = child.build_model(input_shape=input_size)

        _maybe_load_pretrained(child_model, child_graph, child_sd_path, cfg)

        if child_sd_path and os.path.exists(child_sd_path):
            try:
                sd = torch.load(child_sd_path, map_location="cpu", weights_only=True)
                child_model.load_state_dict(clean_state_dict(sd), strict=False)
            except Exception:
                pass
            finally:
                try: os.remove(child_sd_path)
                except OSError: pass

        teacher_model = None
        if is_approx and parent_graph is not None:
            try:
                teacher       = Individual(parent_graph)
                teacher_model = teacher.build_model(input_shape=input_size)
                if parent_sd_path and os.path.exists(parent_sd_path):
                    sd = torch.load(parent_sd_path, map_location="cpu", weights_only=True)
                    teacher_model.load_state_dict(clean_state_dict(sd), strict=False)
                teacher_model.eval()
                for p in teacher_model.parameters(): p.requires_grad = False
            except Exception:
                teacher_model = None

        try:
            child.evaluate_cheap(objective_keys=cfg.CHEAP_OBJECTIVES,
                                 input_size=input_size)
        except Exception:
            child.f_cheap = {k: 0 for k in cfg.CHEAP_OBJECTIVES}

        if is_approx and teacher_model is not None:
            try:
                from train.distill import train_with_distillation
                train_with_distillation(
                    student_model=child_model, teacher_model=teacher_model,
                    train_loader=train_loader, device=device,
                    epochs=cfg.DISTILL_EPOCHS, lr=cfg.DISTILL_LR,
                    temperature=cfg.DISTILL_TEMPERATURE, alpha=cfg.DISTILL_ALPHA,
                    weight_decay=cfg.WEIGHT_DECAY, optimizer_name=cfg.OPTIMIZER,
                    show_progress=False, use_amp=cfg.USE_AMP,
                )
            except Exception as e:
                error_logger.error("Distillation failed %s: %s", child.id, e)
            finally:
                teacher_model.cpu(); del teacher_model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()

        model_to_train, is_dp = _wrap_for_multi_gpu(child_model, cfg)
        try:
            from train.trainer import train_model
            train_model(
                model=model_to_train, train_loader=train_loader, device=device,
                epochs=train_epochs, lr=train_lr, weight_decay=cfg.WEIGHT_DECAY,
                optimizer_name=cfg.OPTIMIZER, show_progress=False, use_amp=cfg.USE_AMP,
            )
        except Exception as e:
            error_logger.error("Training failed %s: %s", child.id, e)

        try:
            from train.evaluate import evaluate_accuracy
            val_error = evaluate_accuracy(model_to_train, val_loader,
                                          device=device, use_amp=cfg.USE_AMP)
        except Exception:
            val_error = 1.0

        child.f_exp = {"val_error": val_error}

        if is_dp: del model_to_train
        child_model.cpu()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        # Keep CPU model attached for supernet/pool update in caller
        child.model = child_model
        return {"status": "ok", "ind": child}

    except Exception as exc:
        return {"status": "error", "error": str(exc),
                "traceback": traceback.format_exc()}


def _sequential_train_gpu(payloads, cfg, train_epochs, train_lr,
                          device, desc, train_loader, val_loader,
                          supernet, elite_pool, controller,
                          population, log_prob_map):
    """
    Sequential GPU training with post-train ENAS updates.

    After each child completes:
      - supernet updated (EMA bank)
      - elite pool updated
      - controller reward recorded and periodic gradient step run
    """
    trained = []
    n       = len(payloads)

    if cfg.SHOW_PROGRESS_BAR:
        try:
            from tqdm import tqdm
            it = tqdm(enumerate(payloads), total=n, desc=desc, unit="model")
        except ImportError:
            it = enumerate(payloads)
    else:
        it = enumerate(payloads)

    for i, (idx, pc) in it:
        result = _train_one_child_gpu(pc, cfg, train_epochs, train_lr,
                                      device, train_loader, val_loader)
        if result["status"] != "ok":
            error_logger.error("%s child %d/%d error: %s\n%s",
                               desc, i + 1, n,
                               result.get("error"), result.get("traceback", ""))
            continue

        ind = result["ind"]
        trained.append(ind)

        # ── Post-training ENAS pipeline ────────────────────────────────
        if supernet is not None and ind.model is not None:
            supernet.update_from_model(ind.model, ind.graph)

        if elite_pool is not None:
            elite_pool.update([ind])

        if controller is not None:
            log_prob = log_prob_map.get(ind.id)
            if log_prob is not None:
                reward = controller.compute_reward(ind, population, cfg.MAX_PARAMS)
                controller.record_outcome(log_prob, reward)
                logger.debug("Controller: ind=%s reward=%.2f tier=%s",
                             ind.id, reward,
                             getattr(ind, "_inheritance_tier", "?"))
            controller.update()   # gradient step when buffer full

    if controller is not None:
        controller.force_update()   # flush remaining buffer at gen end

    return trained


# =============================================================================
# CPU path — parallel workers (no controller; supernet/pool updated post-hoc)
# =============================================================================

def _worker_train_child(idx, pickled_payload, cfg, train_epochs, train_lr):
    import os, gc, pickle, time, traceback
    import torch
    os.environ["TQDM_DISABLE"] = "1"
    torch.set_num_threads(1)
    input_size = cfg.get_input_size()

    try:
        start = time.time()
        child_graph, child_sd_path, parent_graph, parent_sd_path, is_approx = \
            pickle.loads(pickled_payload)

        from objectives.cheap import clean_state_dict
        from evolution.individual import Individual as _Ind

        child       = _Ind(child_graph)
        child_model = child.build_model(input_shape=input_size)

        _maybe_load_pretrained(child_model, child_graph, child_sd_path, cfg)

        if child_sd_path and os.path.exists(child_sd_path):
            try:
                sd = torch.load(child_sd_path, map_location="cpu", weights_only=True)
                child_model.load_state_dict(clean_state_dict(sd), strict=False)
            except Exception: pass
            finally:
                try: os.remove(child_sd_path)
                except OSError: pass

        teacher_model = None
        if is_approx and parent_graph is not None:
            try:
                teacher       = _Ind(parent_graph)
                teacher_model = teacher.build_model(input_shape=input_size)
                if parent_sd_path and os.path.exists(parent_sd_path):
                    sd = torch.load(parent_sd_path, map_location="cpu", weights_only=True)
                    teacher_model.load_state_dict(clean_state_dict(sd), strict=False)
                teacher_model.eval()
                for p in teacher_model.parameters(): p.requires_grad = False
            except Exception: teacher_model = None

        try:
            child.evaluate_cheap(objective_keys=cfg.CHEAP_OBJECTIVES,
                                 input_size=input_size)
        except Exception:
            child.f_cheap = {k: 0 for k in cfg.CHEAP_OBJECTIVES}

        from data.loader_factory import get_loaders_for_worker
        train_loader_w, val_loader_w = get_loaders_for_worker(cfg)

        if is_approx and teacher_model is not None:
            try:
                from train.distill import train_with_distillation
                train_with_distillation(
                    student_model=child_model, teacher_model=teacher_model,
                    train_loader=train_loader_w, device="cpu",
                    epochs=cfg.DISTILL_EPOCHS, lr=cfg.DISTILL_LR,
                    temperature=cfg.DISTILL_TEMPERATURE, alpha=cfg.DISTILL_ALPHA,
                    weight_decay=cfg.WEIGHT_DECAY, optimizer_name=cfg.OPTIMIZER,
                    show_progress=False, use_amp=False,
                )
            except Exception as e:
                error_logger.error("CPU distill failed %s: %s", child.id, e)
            finally: del teacher_model; gc.collect()

        try:
            from train.trainer import train_model
            train_model(
                model=child_model, train_loader=train_loader_w, device="cpu",
                epochs=train_epochs, lr=train_lr, weight_decay=cfg.WEIGHT_DECAY,
                optimizer_name=cfg.OPTIMIZER, show_progress=False, use_amp=False,
            )
        except Exception as e:
            error_logger.error("CPU train failed %s: %s", child.id, e)

        try:
            from train.evaluate import evaluate_accuracy
            val_error = evaluate_accuracy(child_model, val_loader_w,
                                          device="cpu", use_amp=False)
        except Exception: val_error = 1.0

        child.f_exp = {"val_error": val_error}

        sd_path = os.path.join(tempfile.gettempdir(),
                               f"nas_cpu_{child.id}_{int(time.time()*1000)}.pt")
        torch.save(child_model.state_dict(), sd_path)
        child.model = None; del child_model; gc.collect()

        return {"idx": idx, "status": "ok",
                "pickled_child":   pickle.dumps(child),
                "trained_sd_path": sd_path,
                "duration":        time.time() - start}
    except Exception as exc:
        return {"idx": idx, "status": "error",
                "error": str(exc), "traceback": traceback.format_exc()}


def _parallel_train_cpu(payloads, cfg, train_epochs, train_lr, desc,
                        supernet, elite_pool):
    import torch
    cpu_count   = max(1, (os.cpu_count() or 2) - 1)
    max_workers = min(cpu_count, len(payloads))
    input_size  = cfg.get_input_size()
    trained     = []

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(_worker_train_child, idx, pc, cfg, train_epochs, train_lr): idx
            for idx, pc in payloads
        }
        if cfg.SHOW_PROGRESS_BAR:
            try:
                from tqdm import tqdm
                it = tqdm(as_completed(futures), total=len(futures), desc=desc)
            except ImportError: it = as_completed(futures)
        else: it = as_completed(futures)

        for fut in it:
            try: result = fut.result()
            except Exception as e:
                error_logger.error("%s future exception: %s", desc, e); continue

            if result.get("status") != "ok":
                error_logger.error("%s worker error [idx=%s]: %s\n%s",
                                   desc, result.get("idx"),
                                   result.get("error"), result.get("traceback", ""))
                continue
            try:
                ind     = pickle.loads(result["pickled_child"])
                sd_path = result.get("trained_sd_path")
                if sd_path and os.path.exists(sd_path):
                    model = ind.build_model(input_shape=input_size)
                    model.load_state_dict(
                        torch.load(sd_path, map_location="cpu", weights_only=True),
                        strict=False)
                    ind.model = model
                    try: os.remove(sd_path)
                    except OSError: pass
                trained.append(ind)
            except Exception as e:
                error_logger.error("Deserialise error: %s", e)

    # Update supernet + pool post-hoc from CPU-trained models
    for ind in trained:
        if supernet is not None and ind.model is not None:
            supernet.update_from_model(ind.model, ind.graph)
    if elite_pool is not None:
        elite_pool.update(trained)

    return trained


# =============================================================================
# Dispatcher
# =============================================================================

def _train_all(payloads, cfg, train_epochs, train_lr, desc, device,
               train_loader, val_loader,
               supernet, elite_pool, controller, population, log_prob_map):
    if device.startswith("cuda"):
        return _sequential_train_gpu(
            payloads, cfg, train_epochs, train_lr, device, desc,
            train_loader, val_loader,
            supernet, elite_pool, controller, population, log_prob_map,
        )
    else:
        return _parallel_train_cpu(
            payloads, cfg, train_epochs, train_lr, desc, supernet, elite_pool,
        )


# =============================================================================
# Payload helpers
# =============================================================================

def _build_payloads(items, temp_dir, gen):
    import torch
    payloads, parent_temp_paths = [], []
    for idx, (child, parent, approx) in enumerate(items):
        try:
            child_sd_path  = None
            parent_sd_path = None
            if child.model is not None:
                child_sd_path = os.path.join(temp_dir, f"child_{child.id}_{gen}.pt")
                torch.save(child.model.state_dict(), child_sd_path)
            if parent is not None and parent.model is not None:
                parent_sd_path = os.path.join(
                    temp_dir, f"parent_{parent.id}_for_{child.id}_{gen}.pt")
                torch.save(parent.model.state_dict(), parent_sd_path)
                parent_temp_paths.append(parent_sd_path)
            pc = pickle.dumps((child.graph, child_sd_path,
                               parent.graph if parent is not None else None,
                               parent_sd_path, approx))
            payloads.append((idx, pc))
        except Exception as e:
            error_logger.error("Serialise error child %s: %s", child.id, e)
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
            "model_path": None, "graph_path": None,
        }
        if ind.model is not None:
            try:
                wpath = os.path.join(gen_dir, f"{ind.id}_weights.pt")
                torch.save(ind.model.state_dict(), wpath)
                record["model_path"] = wpath
            except Exception: pass
        try:
            gpath = os.path.join(gen_dir, f"{ind.id}_graph.pkl")
            with open(gpath, "wb") as f: pickle.dump(ind.graph, f)
            record["graph_path"] = gpath
        except Exception: pass
        history_entry.append(record)
    return history_entry


def _print_summary(gen, population, inheritance_stats=None):
    rows = []
    for ind in population:
        p = ind.f_cheap.get("params")  if ind.f_cheap else None
        f = ind.f_cheap.get("flops")   if ind.f_cheap else None
        v = ind.f_exp.get("val_error") if ind.f_exp   else None
        rows.append((p, f, v))
    rows.sort(key=lambda r: (r[2] or 1.0, r[0] or float("inf")))
    print(f"\n{'='*65}")
    print(f"  Generation {gen}  |  Pareto population: {len(rows)} models")
    print(f"  {'params':>10}  {'flops':>12}  {'val_error':>10}")
    for p, f, v in rows[:8]:
        ps = f"{p:>10,}"      if p is not None else f"{'?':>10}"
        fs = f"{int(f):>12,}" if f is not None else f"{'?':>12}"
        vs = f"{v:.4f}"       if v is not None else "?"
        print(f"  {ps}  {fs}  {vs:>10}")
    if inheritance_stats:
        total = sum(inheritance_stats.values()) or 1
        parts = " | ".join(
            f"{k}={100*v/total:.0f}%({v})"
            for k, v in inheritance_stats.items()
        )
        print(f"\n  Inheritance: {parts}")
    print(f"{'='*65}\n")


# =============================================================================
# Main entry point
# =============================================================================

def run_lemonade(init_graphs, cfg, train_loader, val_loader, device, run_dir):
    """
    Run LEMONADE NAS with ENAS 3-Tier Lamarckian Inheritance.

    New vs original:
      - ENASController guides operator selection (GPU path)
      - ThreeTierInheritance replaces single-tier transfer_weights
      - SupernetWeightBank accumulates EMA weights across all generations
      - LamarcikianElitePool stores top-20 architectures for Tier-2 fallback
      - Post-training pipeline updates all ENAS components

    Parameters
    ----------
    init_graphs  : list[ArchitectureGraph] | list[Individual]
    cfg          : NASConfig
    train_loader : DataLoader  (created once, reused every generation)
    val_loader   : DataLoader
    device       : str
    run_dir      : str

    Returns
    -------
    (population, history)
    """
    import torch

    use_gpu    = device.startswith("cuda") and torch.cuda.is_available()
    input_size = cfg.get_input_size()

    # ── Initialise ENAS components ─────────────────────────────────────────
    from enas.supernet    import SupernetWeightBank
    from enas.weight_pool import LamarcikianElitePool
    from enas.inheritance import ThreeTierInheritance

    supernet    = SupernetWeightBank()
    elite_pool  = LamarcikianElitePool(
        max_size=getattr(cfg, "ELITE_POOL_SIZE", 20)
    )
    inheritance  = ThreeTierInheritance(supernet, elite_pool)

    use_enas_ctrl = getattr(cfg, "USE_ENAS_CONTROLLER", True) and use_gpu
    controller    = None
    if use_enas_ctrl:
        from enas.controller import ENASController
        controller = ENASController(
            lr=getattr(cfg, "CONTROLLER_LR",           3e-4),
            entropy_coef=getattr(cfg, "CONTROLLER_ENTROPY", 0.05),
            update_every=getattr(cfg, "CONTROLLER_UPDATE_EVERY", 8),
        )
        print("  [ENAS] ENASController enabled — controller-guided operator selection")
    else:
        print("  [ENAS] ENASController disabled — using random operator selection")

    print(f"  [ENAS] Supernet weight bank initialised (EMA alpha=0.1)")
    print(f"  [ENAS] Elite pool capacity={getattr(cfg, 'ELITE_POOL_SIZE', 20)}")
    print(f"  [ENAS] 3-Tier inheritance: parent > elite_pool > supernet > random\n")

    logger.info(
        "LEMONADE+ENAS start: gens=%d N_pc=%d N_ac=%d "
        "init_ep=%d child_ep=%d distill_ep=%d device=%s "
        "controller=%s pool_size=%d",
        cfg.GENERATIONS, cfg.N_CHILDREN, cfg.N_ACCEPT,
        cfg.INIT_EPOCHS, cfg.CHILD_EPOCHS, cfg.DISTILL_EPOCHS,
        device, use_enas_ctrl, getattr(cfg, "ELITE_POOL_SIZE", 20),
    )

    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    temp_dir = tempfile.gettempdir()
    history  = {}
    sampler  = KDESampler(base_bandwidth=cfg.KDE_BANDWIDTH)

    # ── Population init ────────────────────────────────────────────────────
    if init_graphs and isinstance(init_graphs[0], Individual):
        population = list(init_graphs)
    else:
        population = [Individual(g) for g in init_graphs]

    for ind in population:
        try:
            ind.evaluate_cheap(objective_keys=cfg.CHEAP_OBJECTIVES,
                               input_size=input_size)
        except Exception as e:
            ind.f_cheap = {k: 0 for k in cfg.CHEAP_OBJECTIVES}
            error_logger.error("Cheap eval failed seed %s: %s", ind.id, e)

    # ── Generation 0 — seed training ──────────────────────────────────────
    logger.info("=== Generation 0: Training %d seeds ===", len(population))

    seed_triples = [(ind, None, False) for ind in population]
    payloads_g0, paths_g0 = _build_payloads(seed_triples, temp_dir, gen=0)

    # Free memory before training
    for ind in population:
        if ind.model is not None:
            ind.model.cpu(); ind.model = None
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

    if payloads_g0:
        trained_seeds = _train_all(
            payloads_g0, cfg,
            train_epochs=cfg.INIT_EPOCHS, train_lr=cfg.INIT_LR,
            desc="Gen 0 | Seed Training", device=device,
            train_loader=train_loader, val_loader=val_loader,
            supernet=supernet, elite_pool=elite_pool,
            controller=None,          # no controller reward signal for gen-0
            population=population, log_prob_map={},
        )
        if trained_seeds:
            population = trained_seeds

    for path in paths_g0:
        try: os.remove(path)
        except OSError: pass

    population = pareto_front(population)
    population = fill_with_diversity(population, population, cfg.MIN_POP)
    history[0] = _save_generation_models(population, 0, models_dir)
    _print_summary(0, population)

    # ── Generations 1+ ────────────────────────────────────────────────────
    for gen in range(1, cfg.GENERATIONS + 1):
        gen_start = time.time()
        logger.info("===== Generation %d =====", gen)

        child_epochs = (
            cfg.CHILD_EPOCHS + int(gen / 5)
            if cfg.EPOCH_PROGRESSION else cfg.CHILD_EPOCHS
        )
        parent_temp_paths = []
        inheritance.reset_stats()

        try:
            # Step 1: KDE fit
            sampler.fit(population, objective_keys=cfg.CHEAP_OBJECTIVES,
                        generation=gen)

            # Step 2: Sample parents
            parents = sampler.sample(population, cfg.N_CHILDREN, allow_repeats=True)
            if not parents:
                logger.warning("Gen %d: no parents sampled", gen)
                history[gen] = history[gen - 1]; continue

            # Log controller distribution if active
            if controller is not None and population:
                probs = controller.operator_probs(population[0], gen, len(population))
                logger.info("Controller probs gen %d: %s", gen,
                            {k: f"{v:.2f}" for k, v in probs.items()})

            # Step 3: Generate candidates with 3-tier inheritance
            candidates   = []
            log_prob_map = {}   # {child.id: log_prob_tensor}

            for p in parents:
                try:
                    log_prob = None

                    # Operator selection: controller or random
                    if controller is not None and use_gpu:
                        chosen_op, log_prob = controller.select_operator(
                            p, gen, len(population)
                        )
                        new_graph, op_name, target_info = apply_specific_operator(
                            p, chosen_op
                        )
                        if new_graph is None:   # fallback if chosen op failed
                            new_graph, op_name, target_info = random_operator(p)
                            log_prob = None
                    else:
                        new_graph, op_name, target_info = random_operator(p)

                    if new_graph is None:
                        continue

                    approx        = is_approx_op(op_name)
                    child         = Individual(new_graph)
                    child.op_name = op_name

                    # ── 3-Tier Inheritance ────────────────────────────────
                    parent_model = p.build_model(input_shape=input_size)
                    child_model  = child.build_model(input_shape=input_size)

                    tier = inheritance.initialize_child(
                        child_model=child_model, child_graph=child.graph,
                        child_ind=child,
                        parent_model=parent_model, parent_graph=p.graph,
                        op_name=op_name, target_info=target_info,
                    )
                    child._inheritance_tier = tier

                    if log_prob is not None:
                        log_prob_map[child.id] = log_prob

                    candidates.append((child, p, approx))

                except Exception as e:
                    error_logger.error("Child gen error from %s: %s", p.id, e)

            if not candidates:
                logger.warning("Gen %d: no candidates", gen)
                history[gen] = history[gen - 1]; continue

            # Step 4: Cheap objectives filter
            valid_candidates = []
            for child, parent, approx in candidates:
                try:
                    cheap = child.evaluate_cheap(
                        objective_keys=cfg.CHEAP_OBJECTIVES, input_size=input_size)
                    if cheap.get("params", 0) <= cfg.MAX_PARAMS:
                        valid_candidates.append((child, parent, approx))
                except Exception as e:
                    error_logger.error("Cheap eval failed %s: %s", child.id, e)

            if not valid_candidates:
                logger.warning("Gen %d: all candidates exceed MAX_PARAMS", gen)
                history[gen] = history[gen - 1]; continue

            # Step 5: KDE acceptance filter
            candidate_inds = [c for c, _, _ in valid_candidates]
            sampler.fit(candidate_inds, objective_keys=cfg.CHEAP_OBJECTIVES,
                        generation=gen)
            n_accept      = min(cfg.N_ACCEPT, len(valid_candidates))
            accepted_inds = sampler.sample(candidate_inds, n_accept,
                                           allow_repeats=False)
            accepted_set  = {id(ind) for ind in accepted_inds}
            accepted_triples = [
                (c, p, a) for c, p, a in valid_candidates
                if id(c) in accepted_set
            ]
            logger.info("Gen %d: %d → %d accepted (controller=%s)",
                        gen, len(valid_candidates), len(accepted_triples),
                        use_enas_ctrl)

            # Log inheritance tier distribution for accepted children
            tier_counts = {}
            for c, _, _ in accepted_triples:
                t = getattr(c, "_inheritance_tier", "?")
                tier_counts[t] = tier_counts.get(t, 0) + 1
            logger.info("Gen %d accepted tier dist: %s", gen, tier_counts)

            # Step 6: Train accepted children
            payloads, parent_temp_paths = _build_payloads(
                accepted_triples, temp_dir, gen)

            trained_children = []
            if payloads:
                trained_children = _train_all(
                    payloads, cfg,
                    train_epochs=child_epochs, train_lr=cfg.CHILD_LR,
                    desc=f"Gen {gen} | Training {len(payloads)} children",
                    device=device,
                    train_loader=train_loader, val_loader=val_loader,
                    supernet=supernet, elite_pool=elite_pool,
                    controller=controller,
                    population=population, log_prob_map=log_prob_map,
                )

            # Step 7: Pareto update + diversity fill
            if trained_children:
                combined   = population + trained_children
                new_front  = pareto_front(combined)
                population = fill_with_diversity(new_front, combined, cfg.MIN_POP)
            else:
                logger.warning("Gen %d: no children trained", gen)

        except Exception as e:
            error_logger.error("Unhandled error gen %d: %s\n%s",
                               gen, e, traceback.format_exc())
        finally:
            for path in parent_temp_paths:
                try: os.remove(path)
                except OSError: pass
            elapsed = time.time() - gen_start
            logger.info(
                "Gen %d done in %.1fs | pop=%d | supernet=%d keys | pool=%d entries",
                gen, elapsed, len(population),
                supernet.coverage(), len(elite_pool),
            )

        history[gen] = _save_generation_models(population, gen, models_dir)
        _print_summary(gen, population, inheritance.log_stats())

    # ── Final summary + save ENAS state ───────────────────────────────────
    print(f"\n  [ENAS] Final supernet bank: {supernet.coverage()} unique op signatures")
    print(f"  [ENAS] Final elite pool: {len(elite_pool)} architectures")
    inheritance.log_stats()

    try:
        enas_dir = os.path.join(run_dir, "enas_state")
        os.makedirs(enas_dir, exist_ok=True)
        with open(os.path.join(enas_dir, "supernet.pkl"), "wb") as f:
            pickle.dump(supernet.state_dict(), f)
        if controller is not None:
            with open(os.path.join(enas_dir, "controller.pkl"), "wb") as f:
                pickle.dump(controller.state_dict(), f)
        logger.info("ENAS state checkpointed to %s", enas_dir)
    except Exception as e:
        logger.warning("ENAS checkpoint failed (non-fatal): %s", e)

    elite_pool.cleanup()
    return population, history