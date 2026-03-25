from architectures.node import Node
from architectures.graph import ArchitectureGraph
from architectures.compiler import CompiledModel
import torch

def test_basic_compile():
    graph = ArchitectureGraph()

    graph.add_node(Node(0, 'conv', {
        'in_channels': 3,
        'out_channels': 16
    }, parents=[]))

    graph.add_node(Node(1, 'bn', {
        'num_features': 16
    }, parents=[0]))

    graph.add_node(Node(2, 'relu', {}, parents=[1]))

    graph.set_output(2)

    model = CompiledModel(graph)

    x = torch.randn(1, 3, 32, 32)
    y = model(x)

    print("Output shape:", y.shape)
    assert y.shape == (1, 16, 32, 32)

if __name__ == "__main__":
    test_basic_compile()
