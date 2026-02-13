# main.py

import torch
from architectures.node import Node
from architectures.graph import ArchitectureGraph
from evolution.lemonade_full import run_lemonade
from data.cifar10 import get_cifar_loaders
from utils.logger import get_logger

logger = get_logger("main", logfile="logs/main.log")


def seed_graph():
    """
    Minimal working architecture:
    Conv → BN → ReLU → Flatten → Linear
    """
    g = ArchitectureGraph()

    g.add_node(Node(0, 'conv', {
        'in_channels': 3,
        'out_channels': 8
    }, parents=[]))

    g.add_node(Node(1, 'bn', {
        'num_features': 8
    }, parents=[0]))

    g.add_node(Node(2, 'relu', {}, parents=[1]))

    g.add_node(Node(3, 'flatten', {}, parents=[2]))

    g.add_node(Node(4, 'linear', {
        'in_features': 8 * 32 * 32,
        'out_features': 10
    }, parents=[3]))

    g.set_output(4)

    return g


def main():

    logger.info("Starting FULL LEMONADE experiment")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # -------------------------------------------------
    # Load CIFAR-10 (correct function name)
    # -------------------------------------------------
    train_loader, val_loader = get_cifar_loaders(batch_size=128)

    # -------------------------------------------------
    # Run LEMONADE
    # -------------------------------------------------
    final_population = run_lemonade(
        init_graphs=[seed_graph()],
        generations=5,
        n_children=6,
        n_accept=3,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
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
