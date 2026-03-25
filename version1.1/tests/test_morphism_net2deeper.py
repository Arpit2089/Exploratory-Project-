# tests/test_morphism_net2deeper.py
from architectures.node import Node
from architectures.graph import ArchitectureGraph
from architectures.compiler import CompiledModel
from morphisms.exact import apply_net2deeper, initialize_conv_as_identity, initialize_bn_as_identity, inherit_weights
from utils.logger import get_logger
import torch
import torch.nn as nn

logger = get_logger("test_morphism", logfile="logs/test_morphism.log")

def build_small_graph():
    g = ArchitectureGraph()
    # conv -> bn -> relu -> conv2 -> bn2 -> relu2
    g.add_node(Node(0, 'conv', {'in_channels':3, 'out_channels':8, 'kernel':3, 'stride':1, 'padding':1}, parents=[]))
    g.add_node(Node(1, 'bn', {'num_features':8}, parents=[0]))
    g.add_node(Node(2, 'relu', {}, parents=[1]))

    g.add_node(Node(3, 'conv', {'in_channels':8, 'out_channels':8, 'kernel':3, 'stride':1, 'padding':1}, parents=[2]))
    g.add_node(Node(4, 'bn', {'num_features':8}, parents=[3]))
    g.add_node(Node(5, 'relu', {}, parents=[4]))

    g.set_output(5)
    return g

def max_abs_diff(a: torch.Tensor, b: torch.Tensor):
    return float((a - b).abs().max().item())

def main():
    g = build_small_graph()
    parent_model = CompiledModel(g)
    parent_model.eval()

    # random input
    x = torch.randn(2,3,32,32)

    with torch.no_grad():
        out_parent = parent_model(x)

    # apply net2deeper after node 2 (a relu)
    new_graph = apply_net2deeper(g, relu_node_id=2, kernel=1, stride=1, padding=0)

    child_model = CompiledModel(new_graph)
    child_model.eval()

    # initialize identity for the newly created conv and bn in child_model
    # find keys of new nodes (conv id is max-2)
    # safer: detect modules that are new by name vs parent
    parent_keys = set(parent_model.layers.keys())
    child_keys = set(child_model.layers.keys())
    new_keys = child_keys - parent_keys
    logger.info("Parent layers: %s", sorted(parent_keys))
    logger.info("Child layers: %s", sorted(child_keys))
    logger.info("New layers in child: %s", sorted(new_keys))

    # initialize any new conv/bn we find
    for k in new_keys:
        mod = child_model.layers[k]
        if isinstance(mod, nn.Conv2d):
            logger.info("Initializing new conv module %s as identity", k)
            initialize_conv_as_identity(mod)
        elif isinstance(mod, nn.Sequential):
            # separable conv may be sequential; try to initialize first conv
            # but our net2deeper creates simple Conv2d, so this is just cautious
            for sub in mod:
                if isinstance(sub, nn.Conv2d):
                    initialize_conv_as_identity(sub)
        elif isinstance(mod, nn.BatchNorm2d):
            initialize_bn_as_identity(mod)

    # inherit weights for shared modules from parent to child
    inherit_weights(parent_model, child_model)

    with torch.no_grad():
        out_child = child_model(x)

    diff = max_abs_diff(out_parent, out_child)
    logger.info("Max absolute difference between parent and child outputs: %g", diff)
    print("max_abs_diff =", diff)
    assert diff < 1e-5, f"Net2Deeper did not preserve function; diff={diff}"

if __name__ == "__main__":
    main()
