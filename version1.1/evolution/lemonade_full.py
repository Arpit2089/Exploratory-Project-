# evolution/lemonade_full.py
"""
CPU-parallelized LEMONADE run loop (ProcessPoolExecutor).
- Accepts `device` argument so main.py can call run_lemonade(..., device=...)
- Worker processes rebuild an Individual from a pickled ArchitectureGraph,
  run cheap & expensive eval (expensive on CPU), and return a pickled trained Individual.
- Workers limit torch threads and DataLoader workers to avoid CPU oversubscription.
- Workers set TQDM_DISABLE=1 so progress bars from workers don't collide with main.
"""

import os
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from evolution.individual import Individual
from evolution.pareto import pareto_front
from evolution.sampling import KDESampler
from evolution.operators import random_operator
from utils.logger import get_logger

logger = get_logger("lemonade", logfile="logs/lemonade.log")
error_logger = get_logger("lemonade_errors", logfile="logs/lemonade_errors.log")


def _worker_train_child(idx, pickled_graph, epochs, batch_size, num_workers_loader, requested_device):
    """
    Worker function executed in a separate process.

    Returns a dict describing success or error:
      {"idx": idx, "status": "ok", "pickled_child": <bytes>, "duration": <float>}
    or {"idx": idx, "status": "error", "error": <str>, "traceback": <str>}
    """
    try:
        import os
        import pickle
        import time
        import traceback
        # local import of torch & data loader factory
        import torch

        # reduce noisy progress bars in worker
        os.environ["TQDM_DISABLE"] = "1"

        # Limit BLAS / torch threads inside worker to avoid oversubscription
        # (important for CPU parallel runs)
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(v, "1")

        start = time.time()

        # Rebuild Individual from architecture graph (graph is smaller IPC)
        graph = pickle.loads(pickled_graph)
        child = Individual(graph)

        # Recompute cheap objectives inside worker to ensure consistency
        try:
            child.evaluate_cheap()
        except Exception as e:
            tb = traceback.format_exc()
            return {"idx": idx, "status": "error", "error": f"cheap_eval_failed: {e}", "traceback": tb}

        # Recreate dataset loaders inside worker process to avoid sending DataLoader objects
        from data.cifar10 import get_cifar_loaders
        train_loader_w, val_loader_w = get_cifar_loaders(batch_size=batch_size, num_workers=num_workers_loader)

        # Force CPU training in worker (safe default for CPU-parallel)
        device_to_use = "cpu"
        if requested_device != "cpu":
            # if user asked for cuda but we're in CPU-parallel mode, we force CPU and return a note
            # (parent still receives trained model)
            pass

        # Run expensive evaluation (training) on CPU inside worker
        child.evaluate_expensive(train_loader_w, val_loader_w, device=device_to_use, epochs=epochs)

        duration = time.time() - start

        # Return the pickled trained child
        return {"idx": idx, "status": "ok", "pickled_child": pickle.dumps(child), "duration": duration}

    except Exception as exc:
        tb = traceback.format_exc()
        return {"idx": idx, "status": "error", "error": str(exc), "traceback": tb}


def _print_generation_summary(gen, population, max_rows=6):
    """
    Simple terminal-friendly summary of current Pareto population.
    Sort by cheap flops ascending, then val_error ascending (if available).
    """
    # prepare list of tuples (params, flops, val_error)
    rows = []
    for ind in population:
        params = ind.f_cheap.get("params") if ind.f_cheap else None
        flops = ind.f_cheap.get("flops") if ind.f_cheap else None
        val_error = None
        if getattr(ind, "f_exp", None):
            val_error = ind.f_exp.get("val_error")
        rows.append((params, flops, val_error, ind))

    # sort: prefer low flops, then low val_error (None -> treated as big)
    def sort_key(t):
        params, flops, val_error, _ = t
        return (flops if flops is not None else float("inf"),
                val_error if val_error is not None else float("inf"),
                params if params is not None else float("inf"))

    rows = sorted(rows, key=sort_key)[:max_rows]

    # print compact table
    print("\n" + "=" * 60)
    print(f"Generation {gen} summary (top {len(rows)} Pareto models):")
    print(f"{'idx':>3}  {'params':>10}  {'flops':>10}  {'val_err':>8}")
    for i, (params, flops, val_error, ind) in enumerate(rows):
        pe = str(params) if params is not None else "-"
        fe = str(int(flops)) if flops is not None else "-"
        ve = f"{val_error:.4f}" if val_error is not None else "-"
        print(f"{i:>3}  {pe:>10}  {fe:>10}  {ve:>8}")
    print("=" * 60 + "\n")


def run_lemonade(
    init_graphs,
    generations=5,
    n_children=6,
    n_accept=3,
    epochs=1,
    train_loader=None,
    val_loader=None,
    device="cpu",
    # parallel tuning
    num_workers_loader=0
):
    """
    FULL LEMONADE LOOP (CPU-parallel child expensive evaluation)

    Arguments:
      - init_graphs: list of ArchitectureGraph to initialize population
      - generations, n_children, n_accept, epochs: usual LEMONADE params
      - train_loader, val_loader: if provided, used for expensive evaluation (training)
      - device: requested device string (main accepts it); worker processes will run on CPU
      - max_worker_cap: upper bound for ProcessPoolExecutor max_workers (default 8)
      - num_workers_loader: DataLoader num_workers used inside each worker process (recommended 0 on CPU)
    """

    logger.info("Starting LEMONADE (CPU-parallel child training). gens=%d n_children=%d n_accept=%d epochs=%d device=%s",
                generations, n_children, n_accept, epochs, device)

    # ------------------------------
    # Initialize population
    # ------------------------------
    population = [Individual(g) for g in init_graphs]

    # Evaluate cheap & initial expensive (serial) for initial population
    for idx, ind in enumerate(population):
        try:
            ind.evaluate_cheap()
            logger.debug("Initial population member %d cheap eval: params=%s flops=%s", idx,
                         ind.f_cheap.get('params'), ind.f_cheap.get('flops'))
            if train_loader is not None:
                logger.info("Initial expensive eval for population member %d (serial, device=%s).", idx, device)
                start = time.time()
                # run initial evaluations on requested device (parent process) to keep logs readable
                ind.evaluate_expensive(train_loader, val_loader, device=device, epochs=2)
                duration = time.time() - start
                logger.info("Initial expensive eval done for member %d in %.2fs", idx, duration)
        except Exception as e:
            tb = traceback.format_exc()
            error_logger.error("Error evaluating initial population member %d: %s\n%s", idx, e, tb)

    population = pareto_front(population)
    sampler = KDESampler()

    # ------------------------------
    # Evolution Loop
    # ------------------------------
    for gen in range(generations):
        gen_start = time.time()
        try:
            logger.info("===== Generation %d =====", gen)
            sampler.fit(population)

            children = []
            parents = sampler.sample(population, n_children)
            logger.info("Sampled %d parents to produce up to %d children.", len(parents), n_children)

            # Create candidate children (cheap evaluation only)
            for p_i, p in enumerate(parents):
                try:
                    new_graph = random_operator(p)
                    if new_graph is None:
                        logger.debug("Parent #%d produced no child (operator returned None).", p_i)
                        continue

                    child = Individual(new_graph)
                    child.evaluate_cheap()
                    logger.debug("Child created from parent #%d cheap eval: params=%s flops=%s", p_i,
                                 child.f_cheap.get('params'), child.f_cheap.get('flops'))
                    children.append(child)
                except Exception as e:
                    tb = traceback.format_exc()
                    error_logger.error("Error creating/evaluating cheap child from parent #%d: %s\n%s", p_i, e, tb)

            if len(children) == 0:
                logger.warning("No children generated this generation.")
                _print_generation_summary(gen, population)
                continue

            # ------------------------------
            # Parallel expensive evaluation for children (if loaders provided)
            # ------------------------------
            if train_loader is not None and len(children) > 0:
                cpu_count = os.cpu_count() or 1
                max_workers = min(cpu_count, len(children))
                # try to infer batch_size from provided train_loader; fallback to 128
                batch_size = getattr(train_loader, "batch_size", None) or 128

                logger.info("Dispatching %d children to up to %d worker(s) for expensive eval "
                            "(batch_size=%d, loader_workers=%d).",
                            len(children), max_workers, batch_size, num_workers_loader)

                # Prepare picklable payloads: send only graphs (lighter IPC)
                pickled_children = []
                fallback_serial_children = []  # list of (idx_into_children, child)
                for idx, ch in enumerate(children):
                    try:
                        pc = pickle.dumps(ch.graph)
                        pickled_children.append((idx, pc))
                    except Exception as e:
                        tb = traceback.format_exc()
                        logger.warning("Child #%d graph not pickleable (fallback to serial). Error: %s", idx, e)
                        error_logger.error("Pickle failure for child graph #%d: %s\n%s", idx, e, tb)
                        fallback_serial_children.append((idx, ch))

                trained_children = []
                training_errors = []

                # Parallel dispatch for pickleable children
                if pickled_children:
                    with ProcessPoolExecutor(max_workers=max_workers) as exe:
                        futures_map = {}
                        for idx, pc in pickled_children:
                            fut = exe.submit(_worker_train_child, idx, pc, epochs, batch_size, num_workers_loader, device)
                            futures_map[fut] = idx

                        # Collect results as they complete
                        for fut in as_completed(futures_map):
                            origin_idx = futures_map[fut]
                            try:
                                result = fut.result()
                            except Exception as e:
                                tb = traceback.format_exc()
                                error_logger.error("Executor failure for child #%d: %s\n%s", origin_idx, e, tb)
                                training_errors.append((origin_idx, str(e)))
                                continue

                            if not isinstance(result, dict):
                                error_logger.error("Worker returned unexpected result type for child #%d: %r", origin_idx, type(result))
                                training_errors.append((origin_idx, "unexpected result type"))
                                continue

                            if result.get("status") == "ok":
                                try:
                                    trained_child = pickle.loads(result["pickled_child"])
                                    trained_children.append(trained_child)
                                    logger.info("Worker finished child #%d successfully in %.2fs | params=%s flops=%s",
                                                result.get("idx"),
                                                result.get("duration", -1.0),
                                                trained_child.f_cheap.get("params"),
                                                trained_child.f_cheap.get("flops"))
                                except Exception as e:
                                    tb = traceback.format_exc()
                                    error_logger.error("Failed to unpickle trained child #%d: %s\n%s", result.get("idx"), e, tb)
                                    training_errors.append((result.get("idx"), str(e)))
                            else:
                                worker_idx = result.get("idx")
                                worker_err = result.get("error")
                                worker_tb = result.get("traceback")
                                error_logger.error("Worker error while training child #%s: %s\n%s", worker_idx, worker_err, worker_tb)
                                training_errors.append((worker_idx, worker_err))

                # Serial fallback training for children that couldn't be pickled
                for idx, ch in fallback_serial_children:
                    try:
                        logger.info("Serial-training child #%d (fallback)", idx)
                        start = time.time()
                        ch.evaluate_expensive(train_loader, val_loader, device=device, epochs=epochs)
                        dur = time.time() - start
                        trained_children.append(ch)
                        logger.info("Serial training finished for fallback child #%d in %.2fs", idx, dur)
                    except Exception as e:
                        tb = traceback.format_exc()
                        error_logger.error("Serial training failed for fallback child #%d: %s\n%s", idx, e, tb)
                        training_errors.append((idx, str(e)))

                if training_errors:
                    logger.warning("There were %d training errors this generation. See logs for details.", len(training_errors))

                # Replace children list with successfully trained children
                children = trained_children

                if len(children) == 0:
                    logger.warning("After training, no children remained (all trainings failed). Continuing to next generation.")
                    _print_generation_summary(gen, population)
                    continue

            else:
                logger.info("No train_loader provided; skipping expensive evaluation for children this generation.")

            # ------------------------------
            # Selection: accept best children via KDE
            # ------------------------------
            sampler.fit(children)
            accepted = sampler.sample(children, min(n_accept, len(children)))

            # Merge + keep Pareto front
            # population = pareto_front(population + accepted)
            population = pareto_front(population + accepted)
            MIN_POP = 4
            if len(population) < MIN_POP:
                logger.warning("Population collapsed, refilling with accepted children")
                population = (population + accepted)[:MIN_POP]

            # remove duplicates by (params, flops)
            unique = {}
            for ind in population:
                try:
                    key = (ind.f_cheap["params"], ind.f_cheap["flops"])
                    unique[key] = ind
                except Exception:
                    tb = traceback.format_exc()
                    error_logger.error("Failed reading f_cheap for an individual during dedupe: %s\n%s", ind, tb)
                    unique[id(ind)] = ind

            population = list(unique.values())

            logger.info("Population size after selection: %d", len(population))
            # Terminal-friendly summary
            _print_generation_summary(gen, population)

        except Exception as e:
            tb = traceback.format_exc()
            error_logger.error("Unhandled error in generation %d: %s\n%s", gen, e, tb)
            logger.warning("Generation %d encountered an unexpected error; continuing to next generation.", gen)

        finally:
            gen_dur = time.time() - gen_start
            logger.info("Generation %d completed in %.2fs", gen, gen_dur)

    logger.info("LEMONADE finished. Final population size=%d", len(population))
    return population