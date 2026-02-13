# tests/test_individual.py
from architectures.node import Node
from architectures.graph import ArchitectureGraph
from evolution.individual import Individual
from utils.logger import get_logger

logger = get_logger("test_individual", logfile="logs/test_individual.log")

def build_simple_graph():
    g = ArchitectureGraph()
    g.add_node(Node(0, 'conv', {'in_channels':3, 'out_channels':8}, parents=[]))
    g.add_node(Node(1, 'bn', {'num_features':8}, parents=[0]))
    g.add_node(Node(2, 'relu', {}, parents=[1]))
    g.set_output(2)
    return g

def main():
    g = build_simple_graph()
    ind = Individual(g)

    cheap = ind.evaluate_cheap()
    logger.info("Final cheap objectives: %s", cheap)

if __name__ == "__main__":
    main()
