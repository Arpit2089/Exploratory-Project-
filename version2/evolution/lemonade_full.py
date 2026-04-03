################################################################################
# FOLDER: evolution
# FILE:   lemonade_full.py
# PATH:   .\evolution\lemonade_full.py
################################################################################

import os
import pickle
import time
import traceback
import tempfile
import random
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  

from evolution.individual import Individual
from evolution.pareto import pareto_front
from evolution.sampling import KDESampler
from evolution.operators import random_operator
from utils.logger import get_logger

logger = get_logger("lemonade", logfile="logs/lemonade.log")
error_logger = get_logger("lemonade_errors", logfile="logs/lemonade_errors.log")

def _worker_train_child(idx, pickled_payload, epochs, batch_size, num_workers_loader, requested_device, lr=0.01):
    try:
        import os
        import pickle
        import time
        import traceback
        import torch
        
        # =========================================================
        # FIX: Mute TQDM in workers to prevent Windows Pipe crashes
        # =========================================================
        os.environ["TQDM_DISABLE"] = "1"

        time.sleep(random.uniform(0.1, 2.0))

        # Hard-lock PyTorch threading inside the worker for optimal multi-core scaling
        torch.set_num_threads(1)

        start = time.time()
        
        child_graph, child_sd_path, parent_graph, parent_sd_path, op_name = pickle.loads(pickled_payload)
        
        def clean_sd(sd):
            cleaned = {}
            for k, v in sd.items():
                if "total_ops" in k or "total_params" in k or "profile" in k: continue
                if isinstance(v, torch.Tensor):
                    if torch.isnan(v).any(): raise ValueError(f"NaN detected in inherited tensor {k}.")
                    cleaned[k] = v.contiguous().clone()
                else: cleaned[k] = v
            return cleaned

        child = Individual(child_graph)
        student_model = child.build_model()
        if child_sd_path and os.path.exists(child_sd_path):
            sd = torch.load(child_sd_path, map_location="cpu", weights_only=True)
            student_model.load_state_dict(clean_sd(sd), strict=False)
            try: os.remove(child_sd_path) 
            except: pass

        teacher_model = None
        if parent_graph is not None:
            teacher = Individual(parent_graph)
            teacher_model = teacher.build_model()
            if parent_sd_path and os.path.exists(parent_sd_path):
                sd = torch.load(parent_sd_path, map_location="cpu", weights_only=True)
                teacher_model.load_state_dict(clean_sd(sd), strict=False)
            teacher_model.eval() 

        try: child.evaluate_cheap()
        except Exception as e: return {"idx": idx, "status": "error", "error": f"cheap_eval_failed: {e}"}

        from data.cifar10 import get_cifar_loaders
        train_loader_w, val_loader_w, _ = get_cifar_loaders(batch_size=batch_size, num_workers=0, split_test=True, fast_dev_mode=False)

        # Force CPU to prevent CUDA Multiprocessing OOM crashes
        device_to_use = "cpu"

        # FIX 1: Correctly trigger distillation based on the op_name string assigned in the main loop
        if op_name == "approx_chain" and teacher_model is not None:
            from train.distill import train_with_distillation
            # Pure alignment, no ground truth training
            train_with_distillation(
                student_model=student_model, teacher_model=teacher_model, 
                train_loader=train_loader_w, device=device_to_use, 
                epochs=1, alpha=0.0, lr=0.01    
            )

        # Post-distillation independent training
        child.evaluate_expensive(
            train_loader_w, val_loader_w, device=device_to_use, 
            epochs=epochs, lr=lr  
        )

        duration = time.time() - start
        
        trained_sd_path = os.path.join(tempfile.gettempdir(), f"trained_child_{child.id}_{time.time()}.pt")
        torch.save(student_model.state_dict(), trained_sd_path)
        
        child.model = None 
        if teacher_model is not None: del teacher_model
        del student_model
        gc.collect() 
            
        return {
            "idx": idx, 
            "status": "ok", 
            "pickled_child": pickle.dumps(child), 
            "trained_sd_path": trained_sd_path,
            "duration": duration
        }

    except Exception as exc:
        return {"idx": idx, "status": "error", "error": str(exc), "traceback": traceback.format_exc()}


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
    init_graphs, generations=5, n_children=6, n_accept=3, epochs=1,
    train_loader=None, val_loader=None, device="cpu", num_workers_loader=0
):
    logger.info("Starting LEMONADE. gens=%d n_children=%d n_accept=%d epochs=%d device=%s",
                generations, n_children, n_accept, epochs, device)

    population = [Individual(g) for g in init_graphs]
    temp_dir = tempfile.gettempdir()
    history = {}

    for idx, ind in enumerate(population):
        try: ind.evaluate_cheap()
        except Exception as e: pass

    # --- GENERATION 0 ---
    if train_loader is not None and len(population) > 0:
        logger.info("Running parallel expensive evaluation for Generation 0.")
        import torch
        
        cpu_count = max(1, (os.cpu_count() or 2) - 1)
        max_workers = min(cpu_count, len(population))
        batch_size = getattr(train_loader, "batch_size", None) or 128

        pickled_initials, fallback_serial_initials = [], []
        
        for idx, ind in enumerate(population):
            try:
                ind_sd_path = None
                if ind.model is None: ind.build_model()
                if ind.model is not None:
                    ind_sd_path = os.path.join(temp_dir, f"init_{ind.id}.pt")
                    torch.save(ind.model.state_dict(), ind_sd_path)
                    
                pc = pickle.dumps((ind.graph, ind_sd_path, None, None, "exact_chain"))
                pickled_initials.append((idx, pc))
            except Exception as e:
                fallback_serial_initials.append((idx, ind))

        trained_initials = []
        if pickled_initials:
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                futures_map = {exe.submit(_worker_train_child, idx, pc, int(epochs/2), batch_size, num_workers_loader, device, 0.01): idx for idx, pc in pickled_initials}
                for fut in tqdm(as_completed(futures_map), total=len(pickled_initials), desc="Gen 0 Parallel Training", unit="model"):
                    try:
                        result = fut.result()
                        if result.get("status") == "ok":
                            trained_ind = pickle.loads(result["pickled_child"])
                            sd_path = result.get("trained_sd_path")
                            if sd_path and os.path.exists(sd_path):
                                trained_ind.build_model().load_state_dict(torch.load(sd_path, map_location="cpu", weights_only=True))
                                try: os.remove(sd_path)
                                except: pass
                            trained_initials.append(trained_ind)
                        else:
                            error_logger.error(f"Gen 0 Worker Error: {result.get('error')}")
                    except Exception as e: 
                        error_logger.error(f"Gen 0 Future Exception: {e}")

        population = trained_initials

    population = pareto_front(population)
    history[0] = [{"params": ind.f_cheap.get("params"), "flops": ind.f_cheap.get("flops"), "val_error": ind.f_exp.get("val_error") if ind.f_exp else None} for ind in population]
    sampler = KDESampler()

    # --- GENERATION 1+ ---
    for gen in range(1, generations + 1):
        gen_start = time.time()
        current_epochs = epochs + int(gen/5)
        current_lr = 0.001 
        
        logger.info("===== Generation %d (Epochs: %d, Fine-Tune LR: %f) =====", gen, current_epochs, current_lr)
        parent_paths_to_clean = []
        try:
            sampler.fit(population)
            children, successful_parents = [], []
            parents = sampler.sample(population, n_children)

            for p_i, p in enumerate(parents):
                current_parent = p
                approx_used = False
                
                num_mutations = random.randint(2, 4)
                for m_step in range(num_mutations):
                    new_graph, op_name, target_info = random_operator(current_parent)
                    if new_graph is None: break
                    
                    if op_name in ["prune", "remove", "sepconv"]:
                        approx_used = True
                        
                    child = Individual(new_graph)
                    child.op_name = "approx_chain" if approx_used else "exact_chain"
                    
                    # FIX 2: Explicit error catching. Do NOT fail silently if weights don't transfer.
                    try:
                        from morphisms.weights import transfer_weights
                        transfer_weights(current_parent.build_model(), child.build_model(), op_name, target_info)
                    except Exception as e: 
                        error_logger.error(f"FATAL: Lamarckian transfer failed for {child.id} from {current_parent.id}: {e}")
                        # If transfer fails, break the mutation chain so we don't evaluate a broken model
                        break 
                    
                    current_parent = child
                
                if current_parent != p:
                    try:
                        cheap_obj = current_parent.evaluate_cheap()
                        if cheap_obj['params'] < 10_000_000:
                            children.append(current_parent)
                            successful_parents.append(p)
                    except Exception: pass

            if len(children) == 0:
                history[gen] = history[gen - 1]
                continue

            if train_loader is not None and len(children) > 0:
                import torch 
                cpu_count = max(1, (os.cpu_count() or 2) - 1)
                max_workers = min(cpu_count, len(children))
                batch_size = getattr(train_loader, "batch_size", None) or 128

                pickled_children = []
                for idx, (ch, parent) in enumerate(zip(children, successful_parents)):
                    try:
                        child_sd_path, parent_sd_path = None, None
                        if ch.model is not None:
                            child_sd_path = os.path.join(temp_dir, f"child_{ch.id}_{gen}.pt")
                            torch.save(ch.model.state_dict(), child_sd_path)
                            
                        if parent.model is not None:
                            parent_sd_path = os.path.join(temp_dir, f"parent_{parent.id}_for_{ch.id}_{gen}.pt")
                            torch.save(parent.model.state_dict(), parent_sd_path)
                            parent_paths_to_clean.append(parent_sd_path)
                        
                        pc = pickle.dumps((ch.graph, child_sd_path, parent.graph, parent_sd_path, ch.op_name))
                        pickled_children.append((idx, pc))
                    except Exception as e: pass

                trained_children = []
                if pickled_children:
                    with ProcessPoolExecutor(max_workers=max_workers) as exe:
                        futures_map = {exe.submit(_worker_train_child, idx, pc, current_epochs, batch_size, num_workers_loader, device, current_lr): idx for idx, pc in pickled_children}
                        for fut in tqdm(as_completed(futures_map), total=len(pickled_children), desc=f"Gen {gen} Parallel Training", unit="child"):
                            try:
                                result = fut.result()
                                if result.get("status") == "ok":
                                    trained_child = pickle.loads(result["pickled_child"])
                                    sd_path = result.get("trained_sd_path")
                                    if sd_path and os.path.exists(sd_path):
                                        trained_child.build_model().load_state_dict(torch.load(sd_path, map_location="cpu", weights_only=True))
                                        try: os.remove(sd_path)
                                        except: pass
                                    trained_children.append(trained_child)
                                else:
                                    error_logger.error(f"Gen {gen} Worker Error: {result.get('error')}")
                            except Exception as e: 
                                error_logger.error(f"Gen {gen} Future Exception: {e}")

                children = trained_children

            if len(children) > 0:
                sampler.fit(children)
                accepted = sampler.sample(children, min(n_accept, len(children)))
                combined_pop = population + accepted
                new_population = pareto_front(combined_pop)
                
                MIN_POP = 4
                if len(new_population) < MIN_POP:
                    for ind in combined_pop:
                        if ind not in new_population: new_population.append(ind)
                        if len(new_population) >= MIN_POP: break
                
                population = list(new_population)

            _print_generation_summary(gen, population)
            history[gen] = [{"params": ind.f_cheap.get("params"), "flops": ind.f_cheap.get("flops"), "val_error": ind.f_exp.get("val_error") if ind.f_exp else None} for ind in population]

        except Exception as e: pass
        finally:
            for path in parent_paths_to_clean:
                try: os.remove(path)
                except: pass
            logger.info("Generation %d completed in %.2fs", gen, time.time() - gen_start)

    return population, history

# import os
# import pickle
# import time
# import traceback
# import tempfile
# import random
# import gc
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from tqdm import tqdm  

# from evolution.individual import Individual
# from evolution.pareto import pareto_front
# from evolution.sampling import KDESampler
# from evolution.operators import random_operator
# from utils.logger import get_logger

# logger = get_logger("lemonade", logfile="logs/lemonade.log")
# error_logger = get_logger("lemonade_errors", logfile="logs/lemonade_errors.log")

# def _worker_train_child(idx, pickled_payload, epochs, batch_size, num_workers_loader, requested_device, lr=0.01):
#     try:
#         import os
#         import pickle
#         import time
#         import traceback
#         import torch
        
#         time.sleep(random.uniform(0.1, 2.0))

#         # Hard-lock PyTorch threading inside the worker
#         torch.set_num_threads(1)

#         start = time.time()
        
#         child_graph, child_sd_path, parent_graph, parent_sd_path, op_name = pickle.loads(pickled_payload)
        
#         def clean_sd(sd):
#             cleaned = {}
#             for k, v in sd.items():
#                 if "total_ops" in k or "total_params" in k or "profile" in k: continue
#                 if isinstance(v, torch.Tensor):
#                     if torch.isnan(v).any(): raise ValueError(f"NaN detected in inherited tensor {k}.")
#                     cleaned[k] = v.contiguous().clone()
#                 else: cleaned[k] = v
#             return cleaned

#         child = Individual(child_graph)
#         student_model = child.build_model()
#         if child_sd_path and os.path.exists(child_sd_path):
#             sd = torch.load(child_sd_path, map_location="cpu", weights_only=True)
#             student_model.load_state_dict(clean_sd(sd), strict=False)
#             try: os.remove(child_sd_path) 
#             except: pass

#         teacher_model = None
#         if parent_graph is not None:
#             teacher = Individual(parent_graph)
#             teacher_model = teacher.build_model()
#             if parent_sd_path and os.path.exists(parent_sd_path):
#                 sd = torch.load(parent_sd_path, map_location="cpu", weights_only=True)
#                 teacher_model.load_state_dict(clean_sd(sd), strict=False)
#             teacher_model.eval() 

#         try: child.evaluate_cheap()
#         except Exception as e: return {"idx": idx, "status": "error", "error": f"cheap_eval_failed: {e}"}

#         from data.cifar10 import get_cifar_loaders
#         train_loader_w, val_loader_w, _ = get_cifar_loaders(batch_size=batch_size, num_workers=0, split_test=True, fast_dev_mode=False)

#         device_to_use = "cpu"

#         if op_name in ["prune", "remove", "sepconv"] and teacher_model is not None:
#             from train.distill import train_with_distillation
#             train_with_distillation(
#                 student_model=student_model, teacher_model=teacher_model, 
#                 train_loader=train_loader_w, device=device_to_use, 
#                 epochs=2, lr=0.01    
#             )

#         child.evaluate_expensive(
#             train_loader_w, val_loader_w, device=device_to_use, 
#             epochs=epochs, lr=lr  
#         )

#         duration = time.time() - start
        
#         # ==============================================================================
#         # IPC FIX: Save weights to disk and sever the PyTorch graph before pickling
#         # ==============================================================================
#         trained_sd_path = os.path.join(tempfile.gettempdir(), f"trained_child_{child.id}_{time.time()}.pt")
#         torch.save(student_model.state_dict(), trained_sd_path)
        
#         child.model = None 
#         if teacher_model is not None: del teacher_model
#         del student_model
#         gc.collect() # Force memory wipe
            
#         return {
#             "idx": idx, 
#             "status": "ok", 
#             "pickled_child": pickle.dumps(child), 
#             "trained_sd_path": trained_sd_path,
#             "duration": duration
#         }

#     except Exception as exc:
#         return {"idx": idx, "status": "error", "error": str(exc), "traceback": traceback.format_exc()}


# def _print_generation_summary(gen, population, max_rows=6):
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
#     init_graphs, generations=5, n_children=6, n_accept=3, epochs=1,
#     train_loader=None, val_loader=None, device="cpu", num_workers_loader=0
# ):
#     logger.info("Starting LEMONADE. gens=%d n_children=%d n_accept=%d epochs=%d device=%s",
#                 generations, n_children, n_accept, epochs, device)

#     population = [Individual(g) for g in init_graphs]
#     temp_dir = tempfile.gettempdir()
#     history = {}

#     for idx, ind in enumerate(population):
#         try: ind.evaluate_cheap()
#         except Exception as e: pass

#     # --- GENERATION 0 (From Scratch) ---
#     if train_loader is not None and len(population) > 0:
#         logger.info("Running parallel expensive evaluation for Generation 0.")
#         import torch
        
#         # Maximize true core usage, leave 1 core for OS
#         cpu_count = max(1, (os.cpu_count() or 2) - 1)
#         max_workers = min(cpu_count, len(population))
#         batch_size = getattr(train_loader, "batch_size", None) or 128

#         pickled_initials, fallback_serial_initials = [], []
        
#         for idx, ind in enumerate(population):
#             try:
#                 ind_sd_path = None
#                 if ind.model is None: ind.build_model()
#                 if ind.model is not None:
#                     ind_sd_path = os.path.join(temp_dir, f"init_{ind.id}.pt")
#                     torch.save(ind.model.state_dict(), ind_sd_path)
                    
#                 pc = pickle.dumps((ind.graph, ind_sd_path, None, None, "exact_chain"))
#                 pickled_initials.append((idx, pc))
#             except Exception as e:
#                 fallback_serial_initials.append((idx, ind))

#         trained_initials = []
#         if pickled_initials:
#             with ProcessPoolExecutor(max_workers=max_workers) as exe:
#                 futures_map = {exe.submit(_worker_train_child, idx, pc, epochs, batch_size, num_workers_loader, device, 0.01): idx for idx, pc in pickled_initials}
#                 for fut in tqdm(as_completed(futures_map), total=len(pickled_initials), desc="Gen 0 Parallel Training", unit="model"):
#                     try:
#                         result = fut.result()
#                         if result.get("status") == "ok":
#                             trained_ind = pickle.loads(result["pickled_child"])
                            
#                             # IPC Fix: Rebuild model in main process from SSD
#                             sd_path = result.get("trained_sd_path")
#                             if sd_path and os.path.exists(sd_path):
#                                 trained_ind.build_model().load_state_dict(torch.load(sd_path, map_location="cpu", weights_only=True))
#                                 try: os.remove(sd_path)
#                                 except: pass
                                
#                             trained_initials.append(trained_ind)
#                     except Exception as e: pass

#         population = trained_initials

#     population = pareto_front(population)
#     history[0] = [{"params": ind.f_cheap.get("params"), "flops": ind.f_cheap.get("flops"), "val_error": ind.f_exp.get("val_error") if ind.f_exp else None} for ind in population]
#     sampler = KDESampler()

#     # --- GENERATION 1+ (Inherited/Fine-Tuning) ---
#     for gen in range(1, generations + 1):
#         gen_start = time.time()
#         current_epochs = epochs + int(gen/3)
#         current_lr = 0.001 
        
#         logger.info("===== Generation %d (Epochs: %d, Fine-Tune LR: %f) =====", gen, current_epochs, current_lr)
#         parent_paths_to_clean = []
#         try:
#             sampler.fit(population)
#             children, successful_parents = [], []
#             parents = sampler.sample(population, n_children)

#             for p_i, p in enumerate(parents):
#                 current_parent = p
#                 approx_used = False
                
#                 num_mutations = random.randint(2, 4)
#                 for m_step in range(num_mutations):
#                     new_graph, op_name, target_info = random_operator(current_parent)
#                     if new_graph is None: break
                    
#                     if op_name in ["prune", "remove", "sepconv"]:
#                         approx_used = True
                        
#                     child = Individual(new_graph)
#                     child.op_name = "approx_chain" if approx_used else "exact_chain"
                    
#                     try:
#                         from morphisms.weights import transfer_weights
#                         transfer_weights(current_parent.build_model(), child.build_model(), op_name, target_info)
#                     except Exception as e: pass
                    
#                     current_parent = child
                
#                 if current_parent != p:
#                     try:
#                         cheap_obj = current_parent.evaluate_cheap()
#                         if cheap_obj['params'] < 10_000_000:
#                             children.append(current_parent)
#                             successful_parents.append(p)
#                     except Exception: pass

#             if len(children) == 0:
#                 history[gen] = history[gen - 1]
#                 continue

#             if train_loader is not None and len(children) > 0:
#                 import torch 
#                 cpu_count = max(1, (os.cpu_count() or 2) - 1)
#                 max_workers = min(cpu_count, len(children))
#                 batch_size = getattr(train_loader, "batch_size", None) or 128

#                 pickled_children = []
#                 for idx, (ch, parent) in enumerate(zip(children, successful_parents)):
#                     try:
#                         child_sd_path, parent_sd_path = None, None
#                         if ch.model is not None:
#                             child_sd_path = os.path.join(temp_dir, f"child_{ch.id}_{gen}.pt")
#                             torch.save(ch.model.state_dict(), child_sd_path)
                            
#                         if parent.model is not None:
#                             parent_sd_path = os.path.join(temp_dir, f"parent_{parent.id}_for_{ch.id}_{gen}.pt")
#                             torch.save(parent.model.state_dict(), parent_sd_path)
#                             parent_paths_to_clean.append(parent_sd_path)
                        
#                         pc = pickle.dumps((ch.graph, child_sd_path, parent.graph, parent_sd_path, ch.op_name))
#                         pickled_children.append((idx, pc))
#                     except Exception as e: pass

#                 trained_children = []
#                 if pickled_children:
#                     with ProcessPoolExecutor(max_workers=max_workers) as exe:
#                         futures_map = {exe.submit(_worker_train_child, idx, pc, current_epochs, batch_size, num_workers_loader, device, current_lr): idx for idx, pc in pickled_children}
#                         for fut in tqdm(as_completed(futures_map), total=len(pickled_children), desc=f"Gen {gen} Parallel Training", unit="child"):
#                             try:
#                                 result = fut.result()
#                                 if result.get("status") == "ok":
#                                     trained_child = pickle.loads(result["pickled_child"])
                                    
#                                     # IPC Fix: Rebuild model in main process from SSD
#                                     sd_path = result.get("trained_sd_path")
#                                     if sd_path and os.path.exists(sd_path):
#                                         trained_child.build_model().load_state_dict(torch.load(sd_path, map_location="cpu", weights_only=True))
#                                         try: os.remove(sd_path)
#                                         except: pass
                                        
#                                     trained_children.append(trained_child)
#                             except Exception as e: pass

#                 children = trained_children

#             if len(children) > 0:
#                 sampler.fit(children)
#                 accepted = sampler.sample(children, min(n_accept, len(children)))
#                 combined_pop = population + accepted
#                 new_population = pareto_front(combined_pop)
                
#                 MIN_POP = 4
#                 if len(new_population) < MIN_POP:
#                     for ind in combined_pop:
#                         if ind not in new_population: new_population.append(ind)
#                         if len(new_population) >= MIN_POP: break
                
#                 population = list(new_population)

#             _print_generation_summary(gen, population)
#             history[gen] = [{"params": ind.f_cheap.get("params"), "flops": ind.f_cheap.get("flops"), "val_error": ind.f_exp.get("val_error") if ind.f_exp else None} for ind in population]

#         except Exception as e: pass
#         finally:
#             for path in parent_paths_to_clean:
#                 try: os.remove(path)
#                 except: pass
#             logger.info("Generation %d completed in %.2fs", gen, time.time() - gen_start)

#     return population, history