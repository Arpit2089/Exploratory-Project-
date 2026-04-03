################################################################################
# FOLDER: Root Directory
# FILE:   main.py
# PATH:   .\main.py
################################################################################

import os
# ==============================================================================
# CRITICAL MULTIPROCESSING FIX: Must be set BEFORE PyTorch is imported.
# Prevents workers from spawning exponential threads and crashing the PC.
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore") 

import torch
import copy
from evolution.lemonade_full import run_lemonade
from evolution.operators import random_operator
from data.cifar10 import get_cifar_loaders 
from utils.logger import get_logger
from evolution.individual import Individual
from utils.plot import plot_all_pairs

# IMPORT YOUR NEW MODULAR GRAPH BUILDER
from models.basenet import build_basenet_graph

logger = get_logger("main", logfile="logs/main.log")

def create_diverse_seed_population(num_seeds=5):
    logger.info("Generating diverse seed population of size %d", num_seeds)
    base_graph = build_basenet_graph()
    population = [base_graph] 
    
    for _ in range(num_seeds - 1):
        for attempt in range(10): 
            temp_ind = Individual(copy.deepcopy(base_graph))
            new_graph, _, _ = random_operator(temp_ind)
            if new_graph is not None:
                new_ind = Individual(new_graph)
                try:
                    cheap_obj = new_ind.evaluate_cheap()
                    if cheap_obj['params'] < 10_000_000:
                        population.append(new_graph)
                        break
                except Exception:
                    continue
        else:
            population.append(copy.deepcopy(base_graph))
            
    return population

def main():
    logger.info("Starting FULL LEMONADE experiment")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    try:
        train_loader, val_loader, test_loader = get_cifar_loaders(
            batch_size=128, 
            split_test=True, 
            fast_dev_mode=True  
        )
    except ValueError:
        train_loader, val_loader = get_cifar_loaders(batch_size=128, fast_dev_mode=True)
        test_loader = None

    init_graphs = create_diverse_seed_population(num_seeds=6)

    final_population, history = run_lemonade(
        init_graphs=init_graphs,
        generations=4,
        n_children=10,   
        n_accept=5,     
        epochs=8,       
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    logger.info("Final Pareto population size: %d", len(final_population))
    plot_all_pairs(history, save_dir="logs")
    
    print("\n" + "="*50)
    print("LEMONADE NAS COMPLETE - FINAL PARETO FRONT")
    print("="*50)

    for i, ind in enumerate(final_population):
        val_error = ind.f_exp.get("val_error") if ind.f_exp else None
        
        test_error = None
        if test_loader is not None:
            from train.evaluate import evaluate_accuracy
            logger.info("Evaluating Final Model %d on Test Set...", i)
            test_error = evaluate_accuracy(ind.build_model(), test_loader, device=device)

        logger.info(
            "Model %d : params=%d | flops=%d | val_error=%s | test_error=%s",
            i, ind.f_cheap['params'], ind.f_cheap['flops'],
            f"{val_error:.4f}" if val_error else "N/A",
            f"{test_error:.4f}" if test_error else "N/A"
        )
        
        print(f"Model {i}: Params: {ind.f_cheap['params']:,} | FLOPs: {ind.f_cheap['flops']:,} | Val Err: {val_error:.4f}")

if __name__ == "__main__":
    main()





# ################################################################################
# # FOLDER: Root Directory
# # FILE:   main.py
# # PATH:   .\main.py
# ################################################################################

# import os
# # ==============================================================================
# # CRITICAL MULTIPROCESSING FIX: Must be set BEFORE PyTorch is imported.
# # Prevents workers from spawning exponential threads and crashing the PC.
# # ==============================================================================
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

# import warnings
# warnings.filterwarnings("ignore") # Catch-all to safely silence warnings without crashing

# import torch
# import copy
# from architectures.node import Node
# from architectures.graph import ArchitectureGraph
# from evolution.lemonade_full import run_lemonade
# from evolution.operators import random_operator
# from data.cifar10 import get_cifar_loaders 
# from utils.logger import get_logger
# from evolution.individual import Individual
# from utils.plot import plot_all_pairs

# logger = get_logger("main", logfile="logs/main.log")

# def seed_graph():
#     g = ArchitectureGraph()

#     # Stem
#     g.add_node(Node(0, 'conv', {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'padding': 1}, parents=[]))
#     g.add_node(Node(1, 'bn', {'num_features': 32}, parents=[0]))
#     g.add_node(Node(2, 'relu', {}, parents=[1]))

#     # Stage 1
#     g.add_node(Node(3, 'conv', {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1}, parents=[2]))
#     g.add_node(Node(4, 'bn', {'num_features': 64}, parents=[3]))
#     g.add_node(Node(5, 'relu', {}, parents=[4]))
    
#     # Stage 2
#     g.add_node(Node(6, 'conv', {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1}, parents=[5]))
#     g.add_node(Node(7, 'bn', {'num_features': 128}, parents=[6]))
#     g.add_node(Node(8, 'relu', {}, parents=[7]))

#     # === ADDED POOLING BOTTLENECK ===
#     # Shrinks 8x8 spatial down to 4x4. Makes the Linear layer lightning fast to train.
#     g.add_node(Node(85, 'maxpool', {'kernel_size': 2, 'stride': 2}, parents=[8]))

#     # Classifier Head (Updated for 4x4 spatial size)
#     g.add_node(Node(9, 'flatten', {}, parents=[85]))
#     g.add_node(Node(10, 'linear', {'in_features': 128 * 4 * 4, 'out_features': 10}, parents=[9]))

#     g.set_output(10)
#     return g

# def create_diverse_seed_population(num_seeds=5):
#     logger.info("Generating diverse seed population of size %d", num_seeds)
#     base_graph = seed_graph()
#     population = [base_graph] 
    
#     for _ in range(num_seeds - 1):
#         for attempt in range(10): 
#             temp_ind = Individual(copy.deepcopy(base_graph))
#             new_graph, _, _ = random_operator(temp_ind)
#             if new_graph is not None:
#                 new_ind = Individual(new_graph)
#                 try:
#                     cheap_obj = new_ind.evaluate_cheap()
#                     if cheap_obj['params'] < 10_000_000:
#                         population.append(new_graph)
#                         break
#                 except Exception:
#                     continue
#         else:
#             population.append(copy.deepcopy(base_graph))
            
#     return population

# def main():
#     logger.info("Starting FULL LEMONADE experiment")

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     logger.info("Using device: %s", device)

#     try:
#         train_loader, val_loader, test_loader = get_cifar_loaders(
#             batch_size=128, 
#             split_test=True, 
#             fast_dev_mode=True  
#         )
#     except ValueError:
#         train_loader, val_loader = get_cifar_loaders(batch_size=128, fast_dev_mode=True)
#         test_loader = None

#     init_graphs = create_diverse_seed_population(num_seeds=1)

#     final_population, history = run_lemonade(
#         init_graphs=init_graphs,
#         generations=0,
#         n_children=10,   
#         n_accept=4,     
#         epochs=10,       
#         train_loader=train_loader,
#         val_loader=val_loader,
#         device=device,
#     )

#     logger.info("Final Pareto population size: %d", len(final_population))
#     plot_all_pairs(history, save_dir="logs")
    
#     print("\n" + "="*50)
#     print("LEMONADE NAS COMPLETE - FINAL PARETO FRONT")
#     print("="*50)

#     for i, ind in enumerate(final_population):
#         val_error = ind.f_exp.get("val_error") if ind.f_exp else None
        
#         test_error = None
#         if test_loader is not None:
#             from train.evaluate import evaluate_accuracy
#             logger.info("Evaluating Final Model %d on Test Set...", i)
#             test_error = evaluate_accuracy(ind.build_model(), test_loader, device=device)

#         logger.info(
#             "Model %d : params=%d | flops=%d | val_error=%s | test_error=%s",
#             i, ind.f_cheap['params'], ind.f_cheap['flops'],
#             f"{val_error:.4f}" if val_error else "N/A",
#             f"{test_error:.4f}" if test_error else "N/A"
#         )
        
#         print(f"Model {i}: Params: {ind.f_cheap['params']:,} | FLOPs: {ind.f_cheap['flops']:,} | Val Err: {val_error:.4f}")

# if __name__ == "__main__":
#     main()