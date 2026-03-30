# main.py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch
import random
import copy
from architectures.node import Node
from architectures.graph import ArchitectureGraph
from evolution.lemonade_full import run_lemonade
from evolution.operators import random_operator
from data.cifar10 import get_cifar_loaders # Make sure this supports test_loader now!
from utils.logger import get_logger
from evolution.individual import Individual

logger = get_logger("main", logfile="logs/main.log")


def seed_graph():
    """
    Stronger baseline:
    Conv32 → BN → ReLU → Pool
    Conv64 → BN → ReLU → Pool
    Flatten → Linear
    """
    g = ArchitectureGraph()

    # Block 1
    g.add_node(Node(0, 'conv', {
        'in_channels': 3,
        'out_channels': 32,
        'kernel_size': 3,
        'padding': 1
    }, parents=[]))

    g.add_node(Node(1, 'bn', {'num_features': 32}, parents=[0]))
    g.add_node(Node(2, 'relu', {}, parents=[1]))
    g.add_node(Node(3, 'maxpool', {'kernel_size': 2}, parents=[2]))  # 32→16

    # Block 2
    g.add_node(Node(4, 'conv', {
        'in_channels': 32,
        'out_channels': 64,
        'kernel_size': 3,
        'padding': 1
    }, parents=[3]))

    g.add_node(Node(5, 'bn', {'num_features': 64}, parents=[4]))
    g.add_node(Node(6, 'relu', {}, parents=[5]))
    g.add_node(Node(7, 'maxpool', {'kernel_size': 2}, parents=[6]))  # 16→8

    # Head
    g.add_node(Node(8, 'flatten', {}, parents=[7]))

    g.add_node(Node(9, 'linear', {
        'in_features': 64 * 8 * 8,
        'out_features': 10
    }, parents=[8]))

    g.set_output(9)

    return g


def create_diverse_seed_population(num_seeds=5):
    """
    Generate a diverse initial population by applying random 
    morphisms to the base seed graph before training begins.
    """
    logger.info("Generating diverse seed population of size %d", num_seeds)
    base_graph = seed_graph()
    population = [base_graph] # Keep the original baseline
    
    # Generate mutations for the rest
    for _ in range(num_seeds - 1):
        temp_ind = Individual(copy.deepcopy(base_graph))
        # Attempt to mutate the graph up to 3 times to ensure a valid morphism
        for _ in range(3):
            new_graph, _, _ = random_operator(temp_ind)
            if new_graph is not None:
                population.append(new_graph)
                break
        else:
            # Fallback to base graph if mutations fail repeatedly
            population.append(copy.deepcopy(base_graph))
            
    return population


def main():
    logger.info("Starting FULL LEMONADE experiment")

    # FIX: Re-enable GPU for realistic training times. 
    # (Workers will still run on CPU if you set ProcessPoolExecutor to CPU mode, 
    # but the main parent process evaluations will be fast).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # FIX: Request a 3-way split from your dataloader to prevent evaluation bias
    try:
        train_loader, val_loader, test_loader = get_cifar_loaders(batch_size=128, split_test=True)
    except ValueError:
        logger.warning("get_cifar_loaders does not return test_loader. Falling back to 2-way split.")
        train_loader, val_loader = get_cifar_loaders(batch_size=128)
        test_loader = None

    # FIX: Generate a diverse starting population of 6 architectures instead of 2 identical ones
    init_graphs = create_diverse_seed_population(num_seeds=3)

    # -------------------------------------------------
    # Run LEMONADE
    # -------------------------------------------------
    final_population = run_lemonade(
        init_graphs=init_graphs,
        generations=6,
        n_children=10,
        n_accept=3,      # FIX: Lowered from 6 to 3 (30% acceptance) to increase selection pressure
        epochs=3,        # FIX: epochs=3 is viable now that we have CosineAnnealingLR and Gradient Clipping!
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # -------------------------------------------------
    # Print final results
    # -------------------------------------------------
    logger.info("Final Pareto population size: %d", len(final_population))
    print("\n" + "="*50)
    print("LEMONADE NAS COMPLETE - FINAL PARETO FRONT")
    print("="*50)

    for i, ind in enumerate(final_population):
        val_error = ind.f_exp.get("val_error") if ind.f_exp else None
        
        # Optional: Evaluate the final Pareto front on the unseen Test Set
        test_error = None
        if test_loader is not None:
            from train.evaluate import evaluate_accuracy
            logger.info("Evaluating Final Model %d on Test Set...", i)
            test_error = evaluate_accuracy(ind.build_model(), test_loader, device=device)

        logger.info(
            "Model %d : params=%d | flops=%d | val_error=%s | test_error=%s",
            i,
            ind.f_cheap['params'],
            ind.f_cheap['flops'],
            f"{val_error:.4f}" if val_error else "N/A",
            f"{test_error:.4f}" if test_error else "N/A"
        )
        
        print(f"Model {i}: Params: {ind.f_cheap['params']:,} | FLOPs: {ind.f_cheap['flops']:,} | Val Err: {val_error:.4f}")

if __name__ == "__main__":
    main()