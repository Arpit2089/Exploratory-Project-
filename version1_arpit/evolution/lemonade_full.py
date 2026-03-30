# evolution/lemonade_full.py

import os
import pickle
import time
import traceback
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  

from evolution.individual import Individual
from evolution.pareto import pareto_front
from evolution.sampling import KDESampler
from evolution.operators import random_operator
from utils.logger import get_logger

logger = get_logger("lemonade", logfile="logs/lemonade.log")
error_logger = get_logger("lemonade_errors", logfile="logs/lemonade_errors.log")


def _worker_train_child(idx, pickled_payload, epochs, batch_size, num_workers_loader, requested_device):
    """
    Worker function executed in a separate process.
    Accepts graph data and file paths to load weights directly from disk, bypassing IPC bottlenecks.
    """
    try:
        import os
        import pickle
        import time
        import traceback
        import torch

        os.environ["TQDM_DISABLE"] = "1"

        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(v, "1")

        start = time.time()

        # Unpack the lightweight payload
        child_graph, child_sd_path, parent_graph, parent_sd_path = pickle.loads(pickled_payload)
        
        # Helper to aggressively strip thop garbage from dictionaries
        def clean_sd(sd):
            return {k: v for k, v in sd.items() if "total_ops" not in k and "total_params" not in k and "profile" not in k}

        # 1. Build the Student and load Lamarckian weights from disk
        child = Individual(child_graph)
        student_model = child.build_model()
        if child_sd_path and os.path.exists(child_sd_path):
            sd = torch.load(child_sd_path, map_location="cpu", weights_only=True)
            student_model.load_state_dict(clean_sd(sd), strict=False)
            try: os.remove(child_sd_path) 
            except: pass

        # 2. Build the Teacher for Distillation and load from disk (If parent exists)
        teacher_model = None
        if parent_graph is not None:
            teacher = Individual(parent_graph)
            teacher_model = teacher.build_model()
            if parent_sd_path and os.path.exists(parent_sd_path):
                sd = torch.load(parent_sd_path, map_location="cpu", weights_only=True)
                teacher_model.load_state_dict(clean_sd(sd), strict=False)
            teacher_model.eval() 

        try:
            child.evaluate_cheap()
        except Exception as e:
            tb = traceback.format_exc()
            return {"idx": idx, "status": "error", "error": f"cheap_eval_failed: {e}", "traceback": tb}

        from data.cifar10 import get_cifar_loaders
        train_loader_w, val_loader_w, _ = get_cifar_loaders(
            batch_size=batch_size, 
            num_workers=num_workers_loader, 
            split_test=True
        )

        device_to_use = "cpu"

        child.evaluate_expensive(
            train_loader_w, 
            val_loader_w, 
            device=device_to_use, 
            epochs=epochs,
            teacher_model=teacher_model  
        )

        duration = time.time() - start

        return {"idx": idx, "status": "ok", "pickled_child": pickle.dumps(child), "duration": duration}

    except Exception as exc:
        tb = traceback.format_exc()
        return {"idx": idx, "status": "error", "error": str(exc), "traceback": tb}


def _print_generation_summary(gen, population, max_rows=6):
    rows = []
    for ind in population:
        params = ind.f_cheap.get("params") if ind.f_cheap else None
        flops = ind.f_cheap.get("flops") if ind.f_cheap else None
        val_error = getattr(ind, "f_exp", {}).get("val_error") if getattr(ind, "f_exp", None) else None
        rows.append((params, flops, val_error, ind))

    def sort_key(t):
        params, flops, val_error, _ = t
        return (flops if flops is not None else float("inf"),
                val_error if val_error is not None else float("inf"),
                params if params is not None else float("inf"))

    rows = sorted(rows, key=sort_key)[:max_rows]

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
    num_workers_loader=0
):
    logger.info("Starting LEMONADE (CPU-parallel child training). gens=%d n_children=%d n_accept=%d epochs=%d device=%s",
                generations, n_children, n_accept, epochs, device)

    population = [Individual(g) for g in init_graphs]
    temp_dir = tempfile.gettempdir()

    # 1. Evaluate cheap objectives (serial) for initial population
    for idx, ind in enumerate(population):
        try:
            ind.evaluate_cheap()
        except Exception as e:
            tb = traceback.format_exc()
            error_logger.error("Error evaluating initial population member %d cheap: %s\n%s", idx, e, tb)

    # 2. Parallel expensive evaluation for initial population
    if train_loader is not None and len(population) > 0:
        logger.info("Running parallel expensive evaluation for Generation 0.")
        import torch
        cpu_count = os.cpu_count() or 1
        max_workers = min(cpu_count, len(population))
        batch_size = getattr(train_loader, "batch_size", None) or 128

        pickled_initials = []
        fallback_serial_initials = [] 
        
        for idx, ind in enumerate(population):
            try:
                ind_sd_path = None
                if ind.model is None:
                    ind.build_model()
                if ind.model is not None:
                    ind_sd_path = os.path.join(temp_dir, f"init_{ind.id}.pt")
                    torch.save(ind.model.state_dict(), ind_sd_path)
                
                # Payload: (child_graph, child_sd_path, parent_graph, parent_sd_path)
                pc = pickle.dumps((ind.graph, ind_sd_path, None, None))
                pickled_initials.append((idx, pc))
            except Exception as e:
                fallback_serial_initials.append((idx, ind))

        trained_initials = []
        training_errors = []

        if pickled_initials:
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                futures_map = {}
                for idx, pc in pickled_initials:
                    fut = exe.submit(_worker_train_child, idx, pc, epochs, batch_size, num_workers_loader, device)
                    futures_map[fut] = idx

                for fut in tqdm(as_completed(futures_map), total=len(pickled_initials), desc="Gen 0 Parallel Training", unit="model"):
                    origin_idx = futures_map[fut]
                    try:
                        result = fut.result()
                        if result.get("status") == "ok":
                            trained_ind = pickle.loads(result["pickled_child"])
                            trained_initials.append(trained_ind)
                        else:
                            training_errors.append((origin_idx, result.get("error")))
                            error_logger.error("Worker error for initial member %d: %s", origin_idx, result.get("error"))
                    except Exception as e:
                        training_errors.append((origin_idx, str(e)))
                        error_logger.error("Future exception for initial member %d: %s", origin_idx, str(e))

        # Serial fallback training
        if fallback_serial_initials:
            logger.info("Running serial fallback training for %d initial models", len(fallback_serial_initials))
            for idx, ind in tqdm(fallback_serial_initials, desc="Gen 0 Serial Fallback", unit="model"):
                try:
                    ind.evaluate_expensive(train_loader, val_loader, device=device, epochs=epochs)
                    trained_initials.append(ind)
                except Exception as e:
                    training_errors.append((idx, str(e)))

        if training_errors:
            logger.warning("There were %d training errors in initial population.", len(training_errors))

        population = trained_initials

    population = pareto_front(population)
    sampler = KDESampler()

    for gen in range(1, generations + 1):
        gen_start = time.time()
        parent_paths_to_clean = []
        try:
            logger.info("===== Generation %d =====", gen)
            sampler.fit(population)

            children = []
            successful_parents = []
            parents = sampler.sample(population, n_children)

            for p_i, p in enumerate(parents):
                try:
                    new_graph, op_name, target_info = random_operator(p)
                    if new_graph is None:
                        continue

                    child = Individual(new_graph)
                    
                    # LAMARCKIAN INHERITANCE HOOK
                    parent_model = p.build_model()
                    child_model = child.build_model()
                    
                    try:
                        from morphisms.weights import transfer_weights
                        transfer_weights(parent_model, child_model, op_name, target_info)
                    except ImportError:
                        pass
                    except Exception as e:
                        error_logger.error("Weight transfer failed for child %s: %s", child.id, e)

                    child.evaluate_cheap()
                    children.append(child)
                    successful_parents.append(p)
                except Exception as e:
                    tb = traceback.format_exc()
                    error_logger.error("Error creating/evaluating child from parent #%d: %s\n%s", p_i, e, tb)

            if len(children) == 0:
                logger.warning("No children generated this generation.")
                _print_generation_summary(gen, population)
                continue

            # Parallel Distillation & Evaluation
            if train_loader is not None and len(children) > 0:
                import torch 
                cpu_count = os.cpu_count() or 1
                max_workers = min(cpu_count, len(children))
                batch_size = getattr(train_loader, "batch_size", None) or 128

                pickled_children = []
                fallback_serial_children = [] 
                
                # Pair the successful children with their parents and save weights to disk
                for idx, (ch, parent) in enumerate(zip(children, successful_parents)):
                    try:
                        child_sd_path = None
                        parent_sd_path = None
                        
                        if ch.model is not None:
                            child_sd_path = os.path.join(temp_dir, f"child_{ch.id}_{gen}.pt")
                            torch.save(ch.model.state_dict(), child_sd_path)
                            
                        if parent.model is not None:
                            # Unique name preventing workers from deleting a shared parent's weights
                            parent_sd_path = os.path.join(temp_dir, f"parent_{parent.id}_for_{ch.id}_{gen}.pt")
                            torch.save(parent.model.state_dict(), parent_sd_path)
                            parent_paths_to_clean.append(parent_sd_path)
                        
                        pc = pickle.dumps((ch.graph, child_sd_path, parent.graph, parent_sd_path))
                        pickled_children.append((idx, pc))
                    except Exception as e:
                        fallback_serial_children.append((idx, ch, parent))

                trained_children = []
                training_errors = []

                if pickled_children:
                    with ProcessPoolExecutor(max_workers=max_workers) as exe:
                        futures_map = {}
                        for idx, pc in pickled_children:
                            fut = exe.submit(_worker_train_child, idx, pc, epochs, batch_size, num_workers_loader, device)
                            futures_map[fut] = idx

                        for fut in tqdm(as_completed(futures_map), total=len(pickled_children), desc=f"Gen {gen} Parallel Training", unit="child"):
                            origin_idx = futures_map[fut]
                            try:
                                result = fut.result()
                                if result.get("status") == "ok":
                                    trained_child = pickle.loads(result["pickled_child"])
                                    trained_children.append(trained_child)
                                else:
                                    training_errors.append((origin_idx, result.get("error")))
                                    error_logger.error("Worker error for child %d: %s", origin_idx, result.get("error"))
                            except Exception as e:
                                training_errors.append((origin_idx, str(e)))
                                error_logger.error("Future exception for child %d: %s", origin_idx, str(e))

                # Serial fallback training
                if fallback_serial_children:
                    logger.info("Running serial fallback training for %d children", len(fallback_serial_children))
                    for idx, ch, parent in tqdm(fallback_serial_children, desc=f"Gen {gen} Serial Fallback", unit="child"):
                        try:
                            teacher_model = parent.build_model()
                            teacher_model.eval()
                            ch.evaluate_expensive(train_loader, val_loader, device=device, epochs=epochs, teacher_model=teacher_model)
                            trained_children.append(ch)
                        except Exception as e:
                            training_errors.append((idx, str(e)))

                if training_errors:
                    logger.warning("There were %d training errors this generation.", len(training_errors))

                children = trained_children

            # Selection & Pareto Update
            if len(children) > 0:
                sampler.fit(children)
                accepted = sampler.sample(children, min(n_accept, len(children)))

                combined_pop = population + accepted
                new_population = pareto_front(combined_pop)
                
                MIN_POP = 4
                if len(new_population) < MIN_POP:
                    logger.warning("Population collapsed, refilling with accepted children")
                    for ind in combined_pop:
                        if ind not in new_population:
                            new_population.append(ind)
                        if len(new_population) >= MIN_POP:
                            break
                
                population = new_population

                unique = {}
                for ind in population:
                    try:
                        topo = ind.graph.topological_sort()
                        struct_key = tuple(ind.graph.nodes[n].op_type for n in topo)
                        key = (struct_key, ind.f_cheap["params"], ind.f_cheap["flops"])
                        unique[key] = ind
                    except Exception:
                        unique[id(ind)] = ind

                population = list(unique.values())

            _print_generation_summary(gen, population)

        except Exception as e:
            tb = traceback.format_exc()
            error_logger.error("Unhandled error in generation %d: %s\n%s", gen, e, tb)

        finally:
            # Clean up the parent files from the disk
            for path in parent_paths_to_clean:
                try: os.remove(path)
                except: pass
                
            gen_dur = time.time() - gen_start
            logger.info("Generation %d completed in %.2fs", gen, gen_dur)

    return population






# # evolution/lemonade_full.py

# import os
# import pickle
# import time
# import traceback
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from tqdm import tqdm  # Added tqdm for progress bars

# from evolution.individual import Individual
# from evolution.pareto import pareto_front
# from evolution.sampling import KDESampler
# from evolution.operators import random_operator
# from utils.logger import get_logger

# logger = get_logger("lemonade", logfile="logs/lemonade.log")
# error_logger = get_logger("lemonade_errors", logfile="logs/lemonade_errors.log")


# def _worker_train_child(idx, pickled_payload, epochs, batch_size, num_workers_loader, requested_device):
#     """
#     Worker function executed in a separate process.
#     Accepts a 4-tuple payload to facilitate both Lamarckian Inheritance and Knowledge Distillation.
#     """
#     try:
#         import os
#         import pickle
#         import time
#         import traceback
#         import torch

#         # Reduce noisy progress bars in worker
#         os.environ["TQDM_DISABLE"] = "1"

#         # Limit BLAS / torch threads inside worker to avoid CPU oversubscription
#         try:
#             torch.set_num_threads(1)
#         except Exception:
#             pass
#         for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
#             os.environ.setdefault(v, "1")

#         start = time.time()

#         # Unpack the 4-tuple Distillation Payload
#         child_graph, child_sd, parent_graph, parent_sd = pickle.loads(pickled_payload)
        
#         # 1. Build the Student (Child) and inject Lamarckian weights
#         child = Individual(child_graph)
#         student_model = child.build_model()
#         if child_sd is not None:
#             student_model.load_state_dict(child_sd)

#         # 2. Build the Teacher (Parent) for Knowledge Distillation
#         teacher_model = None
#         if parent_graph is not None:
#             teacher = Individual(parent_graph)
#             teacher_model = teacher.build_model()
#             if parent_sd is not None:
#                 teacher_model.load_state_dict(parent_sd)
#             teacher_model.eval()  # Freeze the teacher completely

#         # Recompute cheap objectives inside worker to ensure consistency
#         try:
#             child.evaluate_cheap()
#         except Exception as e:
#             tb = traceback.format_exc()
#             return {"idx": idx, "status": "error", "error": f"cheap_eval_failed: {e}", "traceback": tb}

#         # FIX: Strict 3-way split import using the corrected filename
#         from data.cifar10 import get_cifar_loaders
#         train_loader_w, val_loader_w, _ = get_cifar_loaders(
#             batch_size=batch_size, 
#             num_workers=num_workers_loader, 
#             split_test=True
#         )

#         device_to_use = "cpu"
#         if requested_device != "cpu":
#             pass

#         # 3. Run expensive evaluation WITH Distillation
#         child.evaluate_expensive(
#             train_loader_w, 
#             val_loader_w, 
#             device=device_to_use, 
#             epochs=epochs,
#             teacher_model=teacher_model  # Inject the teacher here
#         )

#         duration = time.time() - start

#         # Return the pickled trained child
#         return {"idx": idx, "status": "ok", "pickled_child": pickle.dumps(child), "duration": duration}

#     except Exception as exc:
#         tb = traceback.format_exc()
#         return {"idx": idx, "status": "error", "error": str(exc), "traceback": tb}


# def _print_generation_summary(gen, population, max_rows=6):
#     """
#     Terminal-friendly summary of the current Pareto population.
#     """
#     rows = []
#     for ind in population:
#         params = ind.f_cheap.get("params") if ind.f_cheap else None
#         flops = ind.f_cheap.get("flops") if ind.f_cheap else None
#         val_error = getattr(ind, "f_exp", {}).get("val_error") if getattr(ind, "f_exp", None) else None
#         rows.append((params, flops, val_error, ind))

#     def sort_key(t):
#         params, flops, val_error, _ = t
#         return (flops if flops is not None else float("inf"),
#                 val_error if val_error is not None else float("inf"),
#                 params if params is not None else float("inf"))

#     rows = sorted(rows, key=sort_key)[:max_rows]

#     print("\n" + "=" * 60)
#     print(f"Generation {gen} summary (top {len(rows)} Pareto models):")
#     print(f"{'idx':>3}  {'params':>10}  {'flops':>10}  {'val_err':>8}")
#     for i, (params, flops, val_error, ind) in enumerate(rows):
#         pe = str(params) if params is not None else "-"
#         fe = str(int(flops)) if flops is not None else "-"
#         ve = f"{val_error:.4f}" if val_error is not None else "-"
#         print(f"{i:>3}  {pe:>10}  {fe:>10}  {ve:>8}")
#     print("=" * 60 + "\n")


# def run_lemonade(
#     init_graphs,
#     generations=5,
#     n_children=6,
#     n_accept=3,
#     epochs=1,
#     train_loader=None,
#     val_loader=None,
#     device="cpu",
#     num_workers_loader=0
# ):
#     logger.info("Starting LEMONADE (CPU-parallel child training). gens=%d n_children=%d n_accept=%d epochs=%d device=%s",
#                 generations, n_children, n_accept, epochs, device)

#     population = [Individual(g) for g in init_graphs]

#     # Evaluate cheap & initial expensive (serial) for initial population
#     for idx, ind in enumerate(population):
#         try:
#             ind.evaluate_cheap()
#             if train_loader is not None:
#                 logger.info("Initial expensive eval for population member %d (serial, device=%s).", idx, device)
#                 start = time.time()
#                 # Initial population uses same epochs as children to prevent unfair advantage
#                 ind.evaluate_expensive(train_loader, val_loader, device=device, epochs=1)
#                 duration = time.time() - start
#                 logger.info("Initial expensive eval done for member %d in %.2fs", idx, duration)
#         except Exception as e:
#             tb = traceback.format_exc()
#             error_logger.error("Error evaluating initial population member %d: %s\n%s", idx, e, tb)

#     population = pareto_front(population)
#     sampler = KDESampler()

#     for gen in range(generations):
#         gen_start = time.time()
#         try:
#             logger.info("===== Generation %d =====", gen)
#             sampler.fit(population)

#             children = []
#             successful_parents = []
#             parents = sampler.sample(population, n_children)

#             for p_i, p in enumerate(parents):
#                 try:
#                     new_graph, op_name, target_info = random_operator(p)
#                     if new_graph is None:
#                         continue

#                     child = Individual(new_graph)
                    
#                     # -----------------------------------------------------------------
#                     # LAMARCKIAN INHERITANCE HOOK
#                     # -----------------------------------------------------------------
#                     parent_model = p.build_model()
#                     child_model = child.build_model()
                    
#                     try:
#                         from morphisms.weights import transfer_weights
#                         transfer_weights(parent_model, child_model, op_name, target_info)
#                     except ImportError:
#                         logger.warning("morphisms.weights.transfer_weights not found! Weights randomly initialized.")
#                     except Exception as e:
#                         error_logger.error("Weight transfer failed for child %s: %s", child.id, e)

#                     child.evaluate_cheap()
#                     children.append(child)
#                     successful_parents.append(p)
#                 except Exception as e:
#                     tb = traceback.format_exc()
#                     error_logger.error("Error creating/evaluating child from parent #%d: %s\n%s", p_i, e, tb)

#             if len(children) == 0:
#                 logger.warning("No children generated this generation.")
#                 _print_generation_summary(gen, population)
#                 continue

#             # ------------------------------
#             # Parallel Distillation & Evaluation
#             # ------------------------------
#             if train_loader is not None and len(children) > 0:
#                 cpu_count = os.cpu_count() or 1
#                 max_workers = min(cpu_count, len(children))
#                 batch_size = getattr(train_loader, "batch_size", None) or 128

#                 pickled_children = []
#                 fallback_serial_children = [] 
                
#                 # Pair the successful children with their parents to create the 4-tuple Distillation payload
#                 for idx, (ch, parent) in enumerate(zip(children, successful_parents)):
#                     try:
#                         child_sd = ch.model.state_dict() if ch.model is not None else None
#                         parent_sd = parent.model.state_dict() if parent.model is not None else None
                        
#                         pc = pickle.dumps((ch.graph, child_sd, parent.graph, parent_sd))
#                         pickled_children.append((idx, pc))
#                     except Exception as e:
#                         fallback_serial_children.append((idx, ch, parent))

#                 trained_children = []
#                 training_errors = []

#                 if pickled_children:
#                     with ProcessPoolExecutor(max_workers=max_workers) as exe:
#                         futures_map = {}
#                         for idx, pc in pickled_children:
#                             fut = exe.submit(_worker_train_child, idx, pc, epochs, batch_size, num_workers_loader, device)
#                             futures_map[fut] = idx

#                         # Added tqdm progress bar for parallel child training
#                         for fut in tqdm(as_completed(futures_map), total=len(pickled_children), desc=f"Gen {gen} Parallel Training", unit="child"):
#                             origin_idx = futures_map[fut]
#                             try:
#                                 result = fut.result()
#                                 if result.get("status") == "ok":
#                                     trained_child = pickle.loads(result["pickled_child"])
#                                     trained_children.append(trained_child)
#                                 else:
#                                     training_errors.append((origin_idx, result.get("error")))
#                                     error_logger.error("Worker error for child %d: %s", origin_idx, result.get("error"))
#                             except Exception as e:
#                                 training_errors.append((origin_idx, str(e)))
#                                 error_logger.error("Future exception for child %d: %s", origin_idx, str(e))

#                 # Serial fallback training (with distillation!)
#                 if fallback_serial_children:
#                     logger.info("Running serial fallback training for %d children", len(fallback_serial_children))
#                     # Added tqdm progress bar for serial fallback
#                     for idx, ch, parent in tqdm(fallback_serial_children, desc=f"Gen {gen} Serial Fallback", unit="child"):
#                         try:
#                             teacher_model = parent.build_model()
#                             teacher_model.eval()
#                             ch.evaluate_expensive(train_loader, val_loader, device=device, epochs=epochs, teacher_model=teacher_model)
#                             trained_children.append(ch)
#                         except Exception as e:
#                             training_errors.append((idx, str(e)))

#                 if training_errors:
#                     logger.warning("There were %d training errors this generation.", len(training_errors))

#                 children = trained_children

#             # ------------------------------
#             # Selection & Pareto Update
#             # ------------------------------
#             if len(children) > 0:
#                 sampler.fit(children)
#                 accepted = sampler.sample(children, min(n_accept, len(children)))

#                 combined_pop = population + accepted
#                 new_population = pareto_front(combined_pop)
                
#                 MIN_POP = 4
#                 if len(new_population) < MIN_POP:
#                     logger.warning("Population collapsed, refilling with accepted children")
#                     for ind in combined_pop:
#                         if ind not in new_population:
#                             new_population.append(ind)
#                         if len(new_population) >= MIN_POP:
#                             break
                
#                 population = new_population

#                 # Structural deduplication
#                 unique = {}
#                 for ind in population:
#                     try:
#                         topo = ind.graph.topological_sort()
#                         struct_key = tuple(ind.graph.nodes[n].op_type for n in topo)
#                         key = (struct_key, ind.f_cheap["params"], ind.f_cheap["flops"])
#                         unique[key] = ind
#                     except Exception:
#                         unique[id(ind)] = ind

#                 population = list(unique.values())

#             _print_generation_summary(gen, population)

#         except Exception as e:
#             tb = traceback.format_exc()
#             error_logger.error("Unhandled error in generation %d: %s\n%s", gen, e, tb)

#         finally:
#             gen_dur = time.time() - gen_start
#             logger.info("Generation %d completed in %.2fs", gen, gen_dur)

#     return population












# # # evolution/lemonade_full.py

# # import os
# # import pickle
# # import time
# # import traceback
# # import tempfile
# # from concurrent.futures import ProcessPoolExecutor, as_completed
# # from tqdm import tqdm  

# # from evolution.individual import Individual
# # from evolution.pareto import pareto_front
# # from evolution.sampling import KDESampler
# # from evolution.operators import random_operator
# # from utils.logger import get_logger

# # logger = get_logger("lemonade", logfile="logs/lemonade.log")
# # error_logger = get_logger("lemonade_errors", logfile="logs/lemonade_errors.log")


# # def _worker_train_child(idx, pickled_payload, epochs, batch_size, num_workers_loader, requested_device):
# #     """
# #     Worker function executed in a separate process.
# #     Accepts graph data and file paths to load weights directly from disk, bypassing IPC bottlenecks.
# #     """
# #     try:
# #         import os
# #         import pickle
# #         import time
# #         import traceback
# #         import torch

# #         os.environ["TQDM_DISABLE"] = "1"

# #         try:
# #             torch.set_num_threads(1)
# #         except Exception:
# #             pass
# #         for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
# #             os.environ.setdefault(v, "1")

# #         start = time.time()

# #         # Unpack the lightweight payload
# #         child_graph, child_sd_path, parent_graph, parent_sd_path = pickle.loads(pickled_payload)
        
# #         # 1. Build the Student and load Lamarckian weights from disk
# #         child = Individual(child_graph)
# #         student_model = child.build_model()
# #         if child_sd_path and os.path.exists(child_sd_path):
# #             # FIX: strict=False prevents thop profile buffers from crashing the load
# #             student_model.load_state_dict(torch.load(child_sd_path, map_location="cpu", weights_only=True), strict=False)
# #             # Clean up child temp file
# #             try: os.remove(child_sd_path) 
# #             except: pass

# #         # 2. Build the Teacher for Distillation and load from disk (If parent exists)
# #         teacher_model = None
# #         if parent_graph is not None:
# #             teacher = Individual(parent_graph)
# #             teacher_model = teacher.build_model()
# #             if parent_sd_path and os.path.exists(parent_sd_path):
# #                 # FIX: strict=False prevents thop profile buffers from crashing the load
# #                 teacher_model.load_state_dict(torch.load(parent_sd_path, map_location="cpu", weights_only=True), strict=False)
# #             teacher_model.eval() 

# #         try:
# #             child.evaluate_cheap()
# #         except Exception as e:
# #             tb = traceback.format_exc()
# #             return {"idx": idx, "status": "error", "error": f"cheap_eval_failed: {e}", "traceback": tb}

# #         from data.cifar10 import get_cifar_loaders
# #         train_loader_w, val_loader_w, _ = get_cifar_loaders(
# #             batch_size=batch_size, 
# #             num_workers=num_workers_loader, 
# #             split_test=True
# #         )

# #         device_to_use = "cpu"

# #         child.evaluate_expensive(
# #             train_loader_w, 
# #             val_loader_w, 
# #             device=device_to_use, 
# #             epochs=epochs,
# #             teacher_model=teacher_model  
# #         )

# #         duration = time.time() - start

# #         return {"idx": idx, "status": "ok", "pickled_child": pickle.dumps(child), "duration": duration}

# #     except Exception as exc:
# #         tb = traceback.format_exc()
# #         return {"idx": idx, "status": "error", "error": str(exc), "traceback": tb}


# # def _print_generation_summary(gen, population, max_rows=6):
# #     rows = []
# #     for ind in population:
# #         params = ind.f_cheap.get("params") if ind.f_cheap else None
# #         flops = ind.f_cheap.get("flops") if ind.f_cheap else None
# #         val_error = getattr(ind, "f_exp", {}).get("val_error") if getattr(ind, "f_exp", None) else None
# #         rows.append((params, flops, val_error, ind))

# #     def sort_key(t):
# #         params, flops, val_error, _ = t
# #         return (flops if flops is not None else float("inf"),
# #                 val_error if val_error is not None else float("inf"),
# #                 params if params is not None else float("inf"))

# #     rows = sorted(rows, key=sort_key)[:max_rows]

# #     print("\n" + "=" * 60)
# #     print(f"Generation {gen} summary (top {len(rows)} Pareto models):")
# #     print(f"{'idx':>3}  {'params':>10}  {'flops':>10}  {'val_err':>8}")
# #     for i, (params, flops, val_error, ind) in enumerate(rows):
# #         pe = str(params) if params is not None else "-"
# #         fe = str(int(flops)) if flops is not None else "-"
# #         ve = f"{val_error:.4f}" if val_error is not None else "-"
# #         print(f"{i:>3}  {pe:>10}  {fe:>10}  {ve:>8}")
# #     print("=" * 60 + "\n")


# # def run_lemonade(
# #     init_graphs,
# #     generations=5,
# #     n_children=6,
# #     n_accept=3,
# #     epochs=1,
# #     train_loader=None,
# #     val_loader=None,
# #     device="cpu",
# #     num_workers_loader=0
# # ):
# #     logger.info("Starting LEMONADE (CPU-parallel child training). gens=%d n_children=%d n_accept=%d epochs=%d device=%s",
# #                 generations, n_children, n_accept, epochs, device)

# #     population = [Individual(g) for g in init_graphs]
# #     temp_dir = tempfile.gettempdir()

# #     # 1. Evaluate cheap objectives (serial) for initial population
# #     for idx, ind in enumerate(population):
# #         try:
# #             ind.evaluate_cheap()
# #         except Exception as e:
# #             tb = traceback.format_exc()
# #             error_logger.error("Error evaluating initial population member %d cheap: %s\n%s", idx, e, tb)

# #     # 2. Parallel expensive evaluation for initial population
# #     if train_loader is not None and len(population) > 0:
# #         logger.info("Running parallel expensive evaluation for Generation 0.")
# #         import torch
# #         cpu_count = os.cpu_count() or 1
# #         max_workers = min(cpu_count, len(population))
# #         batch_size = getattr(train_loader, "batch_size", None) or 128

# #         pickled_initials = []
# #         fallback_serial_initials = [] 
        
# #         for idx, ind in enumerate(population):
# #             try:
# #                 ind_sd_path = None
# #                 if ind.model is None:
# #                     ind.build_model()
# #                 if ind.model is not None:
# #                     ind_sd_path = os.path.join(temp_dir, f"init_{ind.id}.pt")
# #                     torch.save(ind.model.state_dict(), ind_sd_path)
                
# #                 # Payload: (child_graph, child_sd_path, parent_graph, parent_sd_path)
# #                 # Since it's gen 0, parent info is None
# #                 pc = pickle.dumps((ind.graph, ind_sd_path, None, None))
# #                 pickled_initials.append((idx, pc))
# #             except Exception as e:
# #                 fallback_serial_initials.append((idx, ind))

# #         trained_initials = []
# #         training_errors = []

# #         if pickled_initials:
# #             with ProcessPoolExecutor(max_workers=max_workers) as exe:
# #                 futures_map = {}
# #                 for idx, pc in pickled_initials:
# #                     fut = exe.submit(_worker_train_child, idx, pc, epochs, batch_size, num_workers_loader, device)
# #                     futures_map[fut] = idx

# #                 for fut in tqdm(as_completed(futures_map), total=len(pickled_initials), desc="Gen 0 Parallel Training", unit="model"):
# #                     origin_idx = futures_map[fut]
# #                     try:
# #                         result = fut.result()
# #                         if result.get("status") == "ok":
# #                             trained_ind = pickle.loads(result["pickled_child"])
# #                             trained_initials.append(trained_ind)
# #                         else:
# #                             training_errors.append((origin_idx, result.get("error")))
# #                             error_logger.error("Worker error for initial member %d: %s", origin_idx, result.get("error"))
# #                     except Exception as e:
# #                         training_errors.append((origin_idx, str(e)))
# #                         error_logger.error("Future exception for initial member %d: %s", origin_idx, str(e))

# #         # Serial fallback training
# #         if fallback_serial_initials:
# #             logger.info("Running serial fallback training for %d initial models", len(fallback_serial_initials))
# #             for idx, ind in tqdm(fallback_serial_initials, desc="Gen 0 Serial Fallback", unit="model"):
# #                 try:
# #                     ind.evaluate_expensive(train_loader, val_loader, device=device, epochs=epochs)
# #                     trained_initials.append(ind)
# #                 except Exception as e:
# #                     training_errors.append((idx, str(e)))

# #         if training_errors:
# #             logger.warning("There were %d training errors in initial population.", len(training_errors))

# #         population = trained_initials

# #     population = pareto_front(population)
# #     sampler = KDESampler()

# #     for gen in range(1, generations + 1):
# #         gen_start = time.time()
# #         parent_paths_to_clean = []
# #         try:
# #             logger.info("===== Generation %d =====", gen)
# #             sampler.fit(population)

# #             children = []
# #             successful_parents = []
# #             parents = sampler.sample(population, n_children)

# #             for p_i, p in enumerate(parents):
# #                 try:
# #                     new_graph, op_name, target_info = random_operator(p)
# #                     if new_graph is None:
# #                         continue

# #                     child = Individual(new_graph)
                    
# #                     # LAMARCKIAN INHERITANCE HOOK
# #                     parent_model = p.build_model()
# #                     child_model = child.build_model()
                    
# #                     try:
# #                         from morphisms.weights import transfer_weights
# #                         transfer_weights(parent_model, child_model, op_name, target_info)
# #                     except ImportError:
# #                         pass
# #                     except Exception as e:
# #                         error_logger.error("Weight transfer failed for child %s: %s", child.id, e)

# #                     child.evaluate_cheap()
# #                     children.append(child)
# #                     successful_parents.append(p)
# #                 except Exception as e:
# #                     tb = traceback.format_exc()
# #                     error_logger.error("Error creating/evaluating child from parent #%d: %s\n%s", p_i, e, tb)

# #             if len(children) == 0:
# #                 logger.warning("No children generated this generation.")
# #                 _print_generation_summary(gen, population)
# #                 continue

# #             # Parallel Distillation & Evaluation
# #             if train_loader is not None and len(children) > 0:
# #                 import torch 
# #                 cpu_count = os.cpu_count() or 1
# #                 max_workers = min(cpu_count, len(children))
# #                 batch_size = getattr(train_loader, "batch_size", None) or 128

# #                 pickled_children = []
# #                 fallback_serial_children = [] 
                
# #                 # Pair the successful children with their parents and save weights to disk
# #                 for idx, (ch, parent) in enumerate(zip(children, successful_parents)):
# #                     try:
# #                         child_sd_path = None
# #                         parent_sd_path = None
                        
# #                         if ch.model is not None:
# #                             child_sd_path = os.path.join(temp_dir, f"child_{ch.id}_{gen}.pt")
# #                             torch.save(ch.model.state_dict(), child_sd_path)
                            
# #                         if parent.model is not None:
# #                             # Unique name preventing workers from deleting a shared parent's weights
# #                             parent_sd_path = os.path.join(temp_dir, f"parent_{parent.id}_for_{ch.id}_{gen}.pt")
# #                             torch.save(parent.model.state_dict(), parent_sd_path)
# #                             parent_paths_to_clean.append(parent_sd_path)
                        
# #                         # Payload is now extremely lightweight strings and dictionaries
# #                         pc = pickle.dumps((ch.graph, child_sd_path, parent.graph, parent_sd_path))
# #                         pickled_children.append((idx, pc))
# #                     except Exception as e:
# #                         fallback_serial_children.append((idx, ch, parent))

# #                 trained_children = []
# #                 training_errors = []

# #                 if pickled_children:
# #                     with ProcessPoolExecutor(max_workers=max_workers) as exe:
# #                         futures_map = {}
# #                         for idx, pc in pickled_children:
# #                             fut = exe.submit(_worker_train_child, idx, pc, epochs, batch_size, num_workers_loader, device)
# #                             futures_map[fut] = idx

# #                         for fut in tqdm(as_completed(futures_map), total=len(pickled_children), desc=f"Gen {gen} Parallel Training", unit="child"):
# #                             origin_idx = futures_map[fut]
# #                             try:
# #                                 result = fut.result()
# #                                 if result.get("status") == "ok":
# #                                     trained_child = pickle.loads(result["pickled_child"])
# #                                     trained_children.append(trained_child)
# #                                 else:
# #                                     training_errors.append((origin_idx, result.get("error")))
# #                                     error_logger.error("Worker error for child %d: %s", origin_idx, result.get("error"))
# #                             except Exception as e:
# #                                 training_errors.append((origin_idx, str(e)))
# #                                 error_logger.error("Future exception for child %d: %s", origin_idx, str(e))

# #                 # Serial fallback training
# #                 if fallback_serial_children:
# #                     logger.info("Running serial fallback training for %d children", len(fallback_serial_children))
# #                     for idx, ch, parent in tqdm(fallback_serial_children, desc=f"Gen {gen} Serial Fallback", unit="child"):
# #                         try:
# #                             teacher_model = parent.build_model()
# #                             teacher_model.eval()
# #                             ch.evaluate_expensive(train_loader, val_loader, device=device, epochs=epochs, teacher_model=teacher_model)
# #                             trained_children.append(ch)
# #                         except Exception as e:
# #                             training_errors.append((idx, str(e)))

# #                 if training_errors:
# #                     logger.warning("There were %d training errors this generation.", len(training_errors))

# #                 children = trained_children

# #             # Selection & Pareto Update
# #             if len(children) > 0:
# #                 sampler.fit(children)
# #                 accepted = sampler.sample(children, min(n_accept, len(children)))

# #                 combined_pop = population + accepted
# #                 new_population = pareto_front(combined_pop)
                
# #                 MIN_POP = 4
# #                 if len(new_population) < MIN_POP:
# #                     logger.warning("Population collapsed, refilling with accepted children")
# #                     for ind in combined_pop:
# #                         if ind not in new_population:
# #                             new_population.append(ind)
# #                         if len(new_population) >= MIN_POP:
# #                             break
                
# #                 population = new_population

# #                 unique = {}
# #                 for ind in population:
# #                     try:
# #                         topo = ind.graph.topological_sort()
# #                         struct_key = tuple(ind.graph.nodes[n].op_type for n in topo)
# #                         key = (struct_key, ind.f_cheap["params"], ind.f_cheap["flops"])
# #                         unique[key] = ind
# #                     except Exception:
# #                         unique[id(ind)] = ind

# #                 population = list(unique.values())

# #             _print_generation_summary(gen, population)

# #         except Exception as e:
# #             tb = traceback.format_exc()
# #             error_logger.error("Unhandled error in generation %d: %s\n%s", gen, e, tb)

# #         finally:
# #             # Clean up the parent files from the disk
# #             for path in parent_paths_to_clean:
# #                 try: os.remove(path)
# #                 except: pass
                
# #             gen_dur = time.time() - gen_start
# #             logger.info("Generation %d completed in %.2fs", gen, gen_dur)

# #     return population