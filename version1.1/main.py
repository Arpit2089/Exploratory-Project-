# main.py

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch
from architectures.node import Node
from architectures.graph import ArchitectureGraph
from evolution.lemonade_full import run_lemonade
from data.cifar10 import get_cifar_loaders
from utils.logger import get_logger

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

def main():

    logger.info("Starting FULL LEMONADE experiment")

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    logger.info("Using device: %s", device)

    # -------------------------------------------------
    # Load CIFAR-10 (correct function name)
    # -------------------------------------------------
    train_loader, val_loader = get_cifar_loaders(batch_size=128)

    # -------------------------------------------------
    # Run LEMONADE
    # -------------------------------------------------
    final_population = run_lemonade(
        init_graphs=[seed_graph() for _ in range(2)],
        generations=6,
        n_children=10,
        n_accept=6,      # 🔥 important
        epochs=3,        # faster
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # -------------------------------------------------
    # Print final results
    # -------------------------------------------------
    logger.info("Final Pareto population size: %d", len(final_population))

    for i, ind in enumerate(final_population):

        val_error = None
        if ind.f_exp is not None:
            val_error = ind.f_exp.get("val_error")

        logger.info(
            "Model %d : params=%d | flops=%d | val_error=%s",
            i,
            ind.f_cheap['params'],
            ind.f_cheap['flops'],
            str(val_error)
        )


if __name__ == "__main__":
    main()
