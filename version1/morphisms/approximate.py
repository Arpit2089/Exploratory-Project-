# morphisms/approximate.py
import torch
import torch.nn as nn
import numpy as np
from utils.logger import get_logger
from architectures.node import Node
from architectures.graph import ArchitectureGraph
import copy

logger = get_logger("morphisms_approx", logfile="logs/morphisms_approx.log")

def _next_node_id(graph: ArchitectureGraph):
    if not graph.nodes:
        return 0
    return max(graph.nodes.keys()) + 1

# ---------------------
# 1) Prune filters
# ---------------------
def apply_prune_filters(
    graph: ArchitectureGraph,
    conv_node_id: int,
    keep_ratio: float = 0.5
):
    """
    Approximate Network Morphism:
    Prune convolution filters and update downstream channel-dependent ops.
    """
    new_graph = graph.clone()

    if conv_node_id not in new_graph.nodes:
        raise KeyError("conv node not found")

    conv = new_graph.nodes[conv_node_id]
    if conv.op_type != 'conv':
        raise ValueError("apply_prune_filters target must be conv")

    old_out = conv.params['out_channels']
    new_out = max(1, int(old_out * keep_ratio))

    logger.info(
        "Pruning conv node %d: old_out=%d -> new_out=%d",
        conv_node_id, old_out, new_out
    )

    # 1️⃣ Update conv
    conv.params['out_channels'] = new_out

    # 2️⃣ Update immediate children
    for nid, node in new_graph.nodes.items():
        if conv_node_id not in node.parents:
            continue

        # Conv child → update in_channels
        if node.op_type == 'conv':
            if node.params.get('in_channels') == old_out:
                node.params['in_channels'] = new_out
                logger.debug(
                    "Updated child conv %d in_channels=%d",
                    nid, new_out
                )

        # BN child → update num_features
        elif node.op_type == 'bn':
            if node.params.get('num_features') == old_out:
                node.params['num_features'] = new_out
                logger.debug(
                    "Updated child BN %d num_features=%d",
                    nid, new_out
                )

        # ReLU / identity are channel-agnostic → do nothing
        elif node.op_type in ('relu', 'identity'):
            continue

        else:
            logger.warning(
                "Prune: unhandled child op %s at node %d",
                node.op_type, nid
            )

    return new_graph

def inherit_weights_prune(parent_model: nn.Module, child_model: nn.Module, conv_node_id: int, keep_indices=None):
    """
    Copy weights from parent conv to child conv for selected indices.
    keep_indices: list/np.array of indices to KEEP (length == child_out)
    If keep_indices is None: choose top-k by filter L1 norm.
    Also adjust subsequent convs' input channels by selecting corresponding input channels.
    """
    parent_layers = parent_model.layers
    child_layers = child_model.layers
    key = str(conv_node_id)
    if key not in parent_layers or key not in child_layers:
        logger.error("Conv node %s missing in parent/child models", key)
        return

    p_conv = parent_layers[key]
    c_conv = child_layers[key]
    with torch.no_grad():
        p_w = p_conv.weight.detach().cpu().numpy()  # (out, in, kh, kw)
        old_out = p_w.shape[0]
        new_out = c_conv.weight.shape[0]

        if keep_indices is None:
            # choose by filter L1 norm
            norms = np.abs(p_w).sum(axis=(1,2,3))
            keep_indices = np.argsort(-norms)[:new_out]
            keep_indices = np.sort(keep_indices)
            logger.info("Auto-selected keep_indices for pruning: %s", keep_indices.tolist())

        # copy selected filters (order preserved)
        for i_new, i_old in enumerate(keep_indices):
            c_conv.weight[i_new].copy_(parent_layers[key].weight[i_old])
            if p_conv.bias is not None and c_conv.bias is not None:
                c_conv.bias[i_new].copy_(p_conv.bias[i_old])

        # For downstream convs that accepted this output as input, we need to select corresponding input channels.
        # We attempt to copy only the kept input channels into the child's downstream conv modules.
        for k in parent_layers.keys():
            if k == key:
                continue
            # for children: if parent's key is a parent in the graph, there exists a conv child with in_channels == old_out
            # We can't access the graph here, so we attempt a best-effort: if child layer has in_channels==old_out, trim it
            mod_p = parent_layers[k]
            mod_c = child_layers.get(k, None)
            if isinstance(mod_p, nn.Conv2d) and mod_c is not None and isinstance(mod_c, nn.Conv2d):
                if mod_p.in_channels == old_out and mod_c.in_channels == new_out:
                    # copy weights: for each out_channel in child, copy weights of kept input channels
                    with torch.no_grad():
                        # mod_p.weight shape: (out_ch, in_ch, kh, kw)
                        # mod_c.weight shape: (out_ch_c, in_ch_c, kh, kw)
                        # We assume out_ch same or compatible
                        min_out = min(mod_p.weight.shape[0], mod_c.weight.shape[0])
                        for o in range(min_out):
                            src = mod_p.weight[o].detach().cpu().numpy()  # (in_ch,kh,kw)
                            # copy selected channels into mod_c.weight[o,:,:,:]
                            new_src = torch.tensor(src[keep_indices], dtype=mod_c.weight.dtype)
                            mod_c.weight[o, :new_src.shape[0], :, :].copy_(new_src)
                        logger.debug("Adjusted downstream conv %s input channels by keeping indices", k)
    logger.info("Prune weight inheritance done for conv %s", key)

# ---------------------
# 2) Remove a layer (node)
# ---------------------
def apply_remove_layer(graph: ArchitectureGraph, remove_node_id: int):
    """
    Remove node remove_node_id and reconnect its parents to its children.
    Conditions:
      - Node should be removable (not the input node or output node)
      - We only handle simple cases: if node has single parent and single child or simple merges.
    """
    new_graph = graph.clone()
    if remove_node_id not in new_graph.nodes:
        raise KeyError("node not found")

    if new_graph.output_node == remove_node_id:
        raise ValueError("Cannot remove output node")

    node = new_graph.nodes[remove_node_id]

    #extra line added later
    if node.op_type in ('conv', 'bn'):
        raise ValueError(
            f"Unsafe remove: cannot remove node type {node.op_type}"
        )

    parents = node.parents.copy()
    # find children
    children = [nid for nid, n in new_graph.nodes.items() if remove_node_id in n.parents]

    logger.info("Removing node %d, parents=%s children=%s", remove_node_id, parents, children)

    # Rewire: each child replaces the parent remove_node_id with all parents of removed node
    for child_id in children:
        child = new_graph.nodes[child_id]
        new_parents = []
        for p in child.parents:
            if p == remove_node_id:
                new_parents.extend(parents)
            else:
                new_parents.append(p)
        child.parents = new_parents
        logger.debug("Child %d new parents: %s", child_id, new_parents)

    # finally delete the node
    del new_graph.nodes[remove_node_id]
    # If removed node was output, must set new output (we guarded earlier)
    return new_graph

def inherit_weights_remove(parent_model: nn.Module, child_model: nn.Module, removed_node_id: int):
    """
    Best-effort: copy weights from modules with matching keys.
    For nodes removed, their parameters are gone; the rest remain identical if shapes match.
    """
    copied = 0
    for key in parent_model.layers.keys():
        if key in child_model.layers:
            p_mod = parent_model.layers[key]
            c_mod = child_model.layers[key]
            p_sd = p_mod.state_dict()
            c_sd = c_mod.state_dict()
            to_load = {k: v.clone() for k, v in p_sd.items() if k in c_sd and p_sd[k].shape == c_sd[k].shape}
            if to_load:
                c_mod.load_state_dict(to_load, strict=False)
                copied += 1
    logger.info("Removed-layer weight inheritance: copied_modules=%d", copied)

# ---------------------
# 3) Replace conv with separable conv
# ---------------------
import math

def apply_replace_with_sepconv(graph: ArchitectureGraph, conv_node_id: int, kernel=3, padding=1):
    """
    Replace a Conv2d node with a separable conv: depthwise (groups=in_ch) + pointwise 1x1
    Changes node op_type to 'sep_conv' and params accordingly.
    """
    new_graph = graph.clone()
    if conv_node_id not in new_graph.nodes:
        raise KeyError("conv node not found")
    node = new_graph.nodes[conv_node_id]
    if node.op_type != 'conv':
        raise ValueError("apply_replace_with_sepconv target must be conv")

    in_ch = node.params['in_channels']
    out_ch = node.params['out_channels']
    node.op_type = 'sep_conv'
    node.params['kernel'] = kernel
    node.params['padding'] = padding
    node.params['in_channels'] = in_ch
    node.params['out_channels'] = out_ch
    logger.info("Replaced conv %d with separable conv (in=%d out=%d kernel=%d)", conv_node_id, in_ch, out_ch, kernel)
    return new_graph

def inherit_weights_sepconv(parent_model: nn.Module, child_model: nn.Module, conv_node_id: int):
    """
    Map parent conv weights (out, in, kh, kw) -> child depthwise + pointwise:
      depthwise: (in, 1, kh, kw) initialize as channel-wise avg or delta
      pointwise: (out, in, 1, 1) approximate by averaging spatial kernel
    """
    key = str(conv_node_id)
    if key not in parent_model.layers or key not in child_model.layers:
        logger.error("Conv node %s missing in parent/child", key)
        return

    p_mod = parent_model.layers[key]
    c_mod = child_model.layers[key]
    # child c_mod may be nn.Sequential([depthwise, pointwise])
    with torch.no_grad():
        p_w = p_mod.weight.detach()  # (out, in, kh, kw)
        out, inn, kh, kw = p_w.shape
        # find depthwise and pointwise
        if isinstance(c_mod, nn.Sequential):
            depth = c_mod[0]   # depthwise conv
            point = c_mod[1]   # pointwise conv
            # depthweight shape: (in,1,kh,kw) if groups=in
            # pointweight shape: (out,in,1,1)
            # initialize depth to delta on center
            dw_w = torch.zeros_like(depth.weight)
            ch_center_h = kh // 2
            ch_center_w = kw // 2
            for i in range(inn):
                if dw_w.shape[1] == 1:
                    dw_w[i, 0, ch_center_h, ch_center_w] = 1.0
                else:
                    # if different shapes, try to set central weight
                    dw_w[i, :, ch_center_h, ch_center_w] = 1.0
            depth.weight.copy_(dw_w)
            if hasattr(depth, 'bias') and depth.bias is not None:
                depth.bias.zero_()
            # pointwise: average parent kernel spatially and copy into 1x1
            pw = p_w.mean(dim=(2,3))  # (out, in)
            # ensure shapes match
            pw = pw.view(out, inn, 1, 1)
            if point.weight.shape == pw.shape:
                point.weight.copy_(pw)
            else:
                # try broadcasting/truncating
                min_out = min(point.weight.shape[0], pw.shape[0])
                min_in = min(point.weight.shape[1], pw.shape[1])
                point.weight[:min_out, :min_in, 0, 0].copy_(pw[:min_out, :min_in, 0, 0])
            if hasattr(point, 'bias') and point.bias is not None:
                point.bias.zero_()
            logger.info("Initialized separable conv for node %s (depthwise+pointwise)", key)
        else:
            logger.warning("Child module for sepconv did not have expected Sequential layout: %s", type(c_mod))
