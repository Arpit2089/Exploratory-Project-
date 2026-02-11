# main.py
from architectures.node import Node
from architectures.graph import ArchitectureGraph
from evolution.lemonade import run_lemonade
from utils.logger import get_logger

logger = get_logger("main", logfile="logs/main.log")

def seed_graph():
    """
    Minimal seed architecture (LEMONADE starts small by design).
    """
    g = ArchitectureGraph()
    g.add_node(Node(0, 'conv', {
        'in_channels': 3,
        'out_channels': 8
    }, parents=[]))
    g.add_node(Node(1, 'bn', {'num_features': 8}, parents=[0]))
    g.add_node(Node(2, 'relu', {}, parents=[1]))
    g.set_output(2)
    return g

def main():
    logger.info("Starting LEMONADE experiment")

    final_population = run_lemonade(
        init_graphs=[seed_graph()],
        generations=10,
        n_children=6,
        n_accept=3
    )

    logger.info("Final Pareto population size: %d", len(final_population))

    for i, ind in enumerate(final_population):
        logger.info(
            "Model %d: params=%d flops=%d",
            i,
            ind.f_cheap['params'],
            ind.f_cheap['flops']
        )

if __name__ == "__main__":
    main()
