# tests/test_approx_morphisms.py
from architectures.node import Node
from architectures.graph import ArchitectureGraph
from architectures.compiler import CompiledModel
from morphisms.approximate import (
    apply_prune_filters, inherit_weights_prune,
    apply_remove_layer, inherit_weights_remove,
    apply_replace_with_sepconv, inherit_weights_sepconv
)
from morphisms.distill import train_student_with_distillation
from utils.logger import get_logger
import torch
import torch.nn as nn
import math
import numpy as np

logger = get_logger("test_approx", logfile="logs/test_approx.log")

def build_small_graph():
    g = ArchitectureGraph()
    # conv0 -> bn -> relu -> conv1 -> bn -> relu -> fc simulated by conv->pool->linear
    g.add_node(Node(0, 'conv', {'in_channels':3, 'out_channels':8, 'kernel':3, 'stride':1, 'padding':1}, parents=[]))
    g.add_node(Node(1, 'bn', {'num_features':8}, parents=[0]))
    g.add_node(Node(2, 'relu', {}, parents=[1]))
    g.add_node(Node(3, 'conv', {'in_channels':8, 'out_channels':8, 'kernel':3, 'stride':1, 'padding':1}, parents=[2]))
    g.add_node(Node(4, 'bn', {'num_features':8}, parents=[3]))
    g.add_node(Node(5, 'relu', {}, parents=[4]))
    g.set_output(5)
    return g

def dummy_dataloader(batch_size=8, batches=5):
    for _ in range(batches):
        imgs = torch.randn(batch_size,3,32,32)
        labels = torch.randint(0,10,(batch_size,))
        yield imgs, labels

def test_prune_and_distill():
    g = build_small_graph()
    parent = CompiledModel(g).eval()
    x = torch.randn(4,3,32,32)
    with torch.no_grad():
        outp = parent(x)

    # 1) prune conv node 3 to half channels
    newg = apply_prune_filters(g, conv_node_id=3, keep_ratio=0.5)
    child = CompiledModel(newg)
    # compute keep_indices from parent conv weight norms
    p_conv = parent.layers['3']
    w = p_conv.weight.detach().cpu().numpy()
    norms = np.abs(w).sum(axis=(1,2,3))
    new_out = child.layers['3'].weight.shape[0]
    keep_idx = np.argsort(-norms)[:new_out]
    keep_idx = np.sort(keep_idx)
    logger.info("Prune keep_idx: %s", keep_idx.tolist())
    inherit_weights_prune(parent, child, conv_node_id=3, keep_indices=keep_idx)
    # quick distill
    dl = dummy_dataloader(batch_size=4, batches=8)
    child = train_student_with_distillation(parent, child, dl, device='cpu', epochs=2, lr=1e-3)
    with torch.no_grad():
        outc = child(x)
    logger.info("Prune: parent out shape %s child out shape %s", outp.shape, outc.shape)

def test_remove_and_distill():
    g = build_small_graph()
    parent = CompiledModel(g).eval()
    # remove the middle conv (3)
    newg = apply_remove_layer(g, remove_node_id=3)
    child = CompiledModel(newg)
    inherit_weights_remove(parent, child, removed_node_id=3)
    dl = dummy_dataloader(batch_size=4, batches=6)
    child = train_student_with_distillation(parent, child, dl, device='cpu', epochs=2, lr=1e-3)
    x = torch.randn(4,3,32,32)
    with torch.no_grad():
        outp = parent(x)
        outc = child(x)
    logger.info("Remove: parent out shape %s child out shape %s", outp.shape, outc.shape)

def test_replace_sep_and_distill():
    g = build_small_graph()
    parent = CompiledModel(g).eval()
    newg = apply_replace_with_sepconv(g, conv_node_id=3, kernel=3, padding=1)
    child = CompiledModel(newg)
    inherit_weights_sepconv(parent, child, conv_node_id=3)
    dl = dummy_dataloader(batch_size=4, batches=6)
    child = train_student_with_distillation(parent, child, dl, device='cpu', epochs=2, lr=1e-3)
    x = torch.randn(4,3,32,32)
    with torch.no_grad():
        outp = parent(x)
        outc = child(x)
    logger.info("Sep replace: parent out %s child out %s", outp.shape, outc.shape)

def main():
    test_prune_and_distill()
    test_remove_and_distill()
    test_replace_sep_and_distill()
    logger.info("All approximate morphism tests completed")

if __name__ == "__main__":
    main()
