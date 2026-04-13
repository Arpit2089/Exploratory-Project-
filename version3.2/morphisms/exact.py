

# morphisms/exact.py
import torch
import torch.nn as nn
from utils.logger import get_logger
from architectures.node import Node
from architectures.graph import ArchitectureGraph
from collections import deque

logger = get_logger("morphisms", logfile="logs/morphisms.log")

def _next_node_id(graph: ArchitectureGraph):
    if not graph.nodes:
        return 0
    return max(graph.nodes.keys()) + 1

def _get_channel_dependent_children(graph: ArchitectureGraph, start_node_id: int):
    """
    BFS traversal to find all downstream Conv and BN nodes that depend 
    on the output channels of start_node_id.
    """
    queue = deque(graph.get_children(start_node_id))
    visited = set()
    downstream_convs = []
    downstream_bns = []
    
    while queue:
        curr = queue.popleft()
        if curr in visited: 
            continue
        visited.add(curr)
        
        node = graph.nodes[curr]
        if node.op_type in ('conv', 'sep_conv'):
            downstream_convs.append(curr)
            # Stop traversing this branch: the Conv acts as a channel bottleneck
        elif node.op_type == 'bn':
            downstream_bns.append(curr)
            queue.extend(graph.get_children(curr))
        elif node.op_type in ('relu', 'identity', 'maxpool', 'add', 'concat'):
            queue.extend(graph.get_children(curr))
            
    return downstream_convs, downstream_bns

def apply_net2deeper(graph: ArchitectureGraph, relu_node_id: int, kernel_size=1, stride=1, padding=0):
    new_graph = graph.clone()
    if relu_node_id not in new_graph.nodes:
        raise KeyError(f"relu node {relu_node_id} not in graph")

    inferred_ch = None
    old_children = new_graph.get_children(relu_node_id)
    
    for child_id in old_children:
        node = new_graph.nodes[child_id]
        if 'in_channels' in node.params:
            inferred_ch = node.params['in_channels']
            break

    if inferred_ch is None:
        parent_node = new_graph.nodes[relu_node_id]
        for p in parent_node.parents:
            pn = new_graph.nodes[p]
            if 'out_channels' in pn.params:
                inferred_ch = pn.params['out_channels']
                break

    if inferred_ch is None:
        raise ValueError("Could not infer channel size.")

    conv_id = _next_node_id(new_graph)
    bn_id = conv_id + 1
    relu_new_id = conv_id + 2

    logger.info("Applying Net2Deeper: inserting nodes conv=%d bn=%d relu=%d after relu_node=%d (channels=%d)",
                conv_id, bn_id, relu_new_id, relu_node_id, inferred_ch)

    conv_params = {
        'in_channels': inferred_ch,
        'out_channels': inferred_ch,
        'kernel_size': kernel_size, 
        'stride': stride,
        'padding': padding,
        'groups': 1
    }
    new_graph.add_node(Node(conv_id, 'conv', conv_params, parents=[relu_node_id]))

    bn_params = {'num_features': inferred_ch}
    new_graph.add_node(Node(bn_id, 'bn', bn_params, parents=[conv_id]))
    new_graph.add_node(Node(relu_new_id, 'relu', {}, parents=[bn_id]))

    for child_id in old_children:
        child = new_graph.nodes[child_id]
        child.parents = [relu_new_id if p == relu_node_id else p for p in child.parents]

    return new_graph

def initialize_conv_as_identity(conv_module: nn.Conv2d):
    with torch.no_grad():
        w = torch.zeros_like(conv_module.weight)
        out_c, in_c, kh, kw = w.shape
        if out_c != in_c:
            return
        ch_center_h, ch_center_w = kh // 2, kw // 2
        for i in range(out_c):
            w[i, i, ch_center_h, ch_center_w] = 1.0
        conv_module.weight.copy_(w)
        if conv_module.bias is not None:
            conv_module.bias.zero_()

def initialize_bn_as_identity(bn_module: nn.BatchNorm2d):
    with torch.no_grad():
        if hasattr(bn_module, 'weight') and bn_module.weight is not None:
            bn_module.weight.fill_(1.0)
        if hasattr(bn_module, 'bias') and bn_module.bias is not None:
            bn_module.bias.zero_()
        bn_module.running_mean.zero_()
        bn_module.running_var.fill_(1.0)

def inherit_weights(parent_model: nn.Module, child_model: nn.Module):
    parent_layers = getattr(parent_model, 'layers', None)
    child_layers = getattr(child_model, 'layers', None)
    if parent_layers is None or child_layers is None:
        return

    copied, skipped = 0, 0
    for key in parent_layers.keys():
        if key in child_layers:
            p_sd = parent_layers[key].state_dict()
            c_sd = child_layers[key].state_dict()
            to_load = {}
            for name, tensor in p_sd.items():
                if name in c_sd and tensor.shape == c_sd[name].shape:
                    to_load[name] = tensor.clone()
            if to_load:
                try:
                    child_layers[key].load_state_dict(to_load, strict=False)
                    copied += 1
                except Exception:
                    skipped += 1
            else:
                skipped += 1
        else:
            skipped += 1

def apply_net2wider(graph: ArchitectureGraph, conv_node_id: int, widen_by: int = 4):
    new_graph = graph.clone()
    conv_node = new_graph.nodes[conv_node_id]
    
    old_out = conv_node.params['out_channels']
    new_out = old_out + widen_by

    conv_node.params['out_channels'] = new_out

    ds_convs, ds_bns = _get_channel_dependent_children(new_graph, conv_node_id)
    
    for nid in ds_convs:
        new_graph.nodes[nid].params['in_channels'] = new_out
        # sep_conv requires groups to match in_channels for its depthwise step
        if new_graph.nodes[nid].op_type == 'sep_conv':
            new_graph.nodes[nid].params['groups'] = new_out 
            
    for nid in ds_bns:
        new_graph.nodes[nid].params['num_features'] = new_out

    return new_graph

def inherit_weights_net2wider(parent_model, child_model, conv_node_id, widen_by):
    key = str(conv_node_id)
    parent_layers, child_layers = parent_model.layers, child_model.layers

    if key not in parent_layers or key not in child_layers:
        return

    p_conv, c_conv = parent_layers[key], child_layers[key]

    with torch.no_grad():
        # --- 1. Widen Target ---
        old_w = p_conv.weight
        old_out = old_w.shape[0]
        new_out = c_conv.weight.shape[0]

        c_conv.weight[:old_out].copy_(old_w)
        for i in range(old_out, new_out):
            src = torch.randint(0, old_out, (1,)).item()
            c_conv.weight[i].copy_(old_w[src])

        if p_conv.bias is not None and c_conv.bias is not None:
            c_conv.bias[:old_out].copy_(p_conv.bias)
            fill_val = p_conv.bias[0] if p_conv.bias.numel() > 0 else 0.0
            c_conv.bias[old_out:].fill_(fill_val)

        ds_convs, ds_bns = _get_channel_dependent_children(child_model.graph, conv_node_id)

        # --- 2. Downstream BNs ---
        for bn_id in ds_bns:
            bn_key = str(bn_id)
            if bn_key in parent_layers and bn_key in child_layers:
                p_bn, c_bn = parent_layers[bn_key], child_layers[bn_key]
                if isinstance(p_bn, nn.BatchNorm2d) and isinstance(c_bn, nn.BatchNorm2d):
                    c_bn.weight[:old_out].copy_(p_bn.weight)
                    c_bn.weight[old_out:].fill_(1.0)
                    c_bn.bias[:old_out].copy_(p_bn.bias)
                    c_bn.bias[old_out:].zero_()
                    c_bn.running_mean[:old_out]

def apply_skip_connection(graph: ArchitectureGraph, from_node: int, to_node: int):
    new_graph = graph.clone()
    
    topo = new_graph.topological_sort()
    if topo.index(from_node) >= topo.index(to_node):
        raise ValueError("from_node must precede to_node topologically")

    if from_node in new_graph.nodes[to_node].parents:
        raise ValueError("Skip already exists")

    new_id = _next_node_id(new_graph)
    merge_node = Node(new_id, op_type="add", params={}, parents=[from_node, to_node])

    # Safely update the children of to_node
    children = new_graph.get_children(to_node)
    for child_id in children:
        child = new_graph.nodes[child_id]
        child.parents = [new_id if p == to_node else p for p in child.parents]

    new_graph.add_node(merge_node)

    if new_graph.output_node == to_node:
        new_graph.set_output(new_id)

    return new_graph