# tests/test_compile_and_objectives.py
from architectures.node import Node
from architectures.graph import ArchitectureGraph
from architectures.compiler import CompiledModel
from objectives.cheap import count_parameters, estimate_flops
from utils.logger import get_logger

logger = get_logger("test", logfile="logs/test.log")
import torch

def build_sample_graph():
    g = ArchitectureGraph()
    # node 0 input conv
    g.add_node(Node(0, 'conv', {'in_channels':3, 'out_channels':8, 'kernel':3, 'stride':1, 'padding':1}, parents=[]))
    g.add_node(Node(1, 'bn', {'num_features':8}, parents=[0]))
    g.add_node(Node(2, 'relu', {}, parents=[1]))

    # node 3 another conv branch
    g.add_node(Node(3, 'conv', {'in_channels':3, 'out_channels':8, 'kernel':3, 'stride':1, 'padding':1}, parents=[]))
    g.add_node(Node(4, 'bn', {'num_features':8}, parents=[3]))
    g.add_node(Node(5, 'relu', {}, parents=[4]))

    # concat node: concat outputs of node 2 and 5 -> channels 16
    g.add_node(Node(6, 'concat', {}, parents=[2,5]))
    # conv after concat (in_channels must match 16)
    g.add_node(Node(7, 'conv', {'in_channels':16, 'out_channels':16, 'kernel':3, 'stride':1, 'padding':1}, parents=[6]))
    g.add_node(Node(8, 'bn', {'num_features':16}, parents=[7]))
    g.add_node(Node(9, 'relu', {}, parents=[8]))

    # add residual: add node 9 and node 2 (shapes must match)
    # to make shapes match, add a 1x1 conv to node 2 to upsample channels to 16
    g.add_node(Node(10, 'conv', {'in_channels':8, 'out_channels':16, 'kernel':1, 'stride':1, 'padding':0}, parents=[2]))
    g.add_node(Node(11, 'bn', {'num_features':16}, parents=[10]))
    g.add_node(Node(12, 'add', {}, parents=[9,11]))

    g.set_output(12)
    return g

def main():
    g = build_sample_graph()
    model = CompiledModel(g)
    logger.info("Model built successfully")
    x = torch.randn(1,3,32,32)
    y = model(x)
    logger.info("Forward output shape: %s", tuple(y.shape))

    params = count_parameters(model)
    flops = estimate_flops(model, input_size=(1,3,32,32))
    logger.info("Params=%d FLOPs=%d", params, flops)

if __name__ == "__main__":
    main()
