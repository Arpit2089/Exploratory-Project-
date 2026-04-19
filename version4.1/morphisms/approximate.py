# morphisms/approximate.py
import torch
import torch.nn as nn
import numpy as np
from utils.logger import get_logger
from architectures.graph import ArchitectureGraph
from morphisms.exact import _get_channel_dependent_children

logger = get_logger("morphisms_approx", logfile="logs/morphisms_approx.log")


# =============================================================================
# apply_* functions: operate on ArchitectureGraph, return new graph
# =============================================================================

def apply_prune_filters(graph: ArchitectureGraph, conv_node_id: int,
                        keep_ratio: float = 0.8) -> ArchitectureGraph:
    new_graph = graph.clone()
    conv      = new_graph.nodes[conv_node_id]

    old_out = conv.params["out_channels"]
    new_out = max(1, int(old_out * keep_ratio))
    conv.params["out_channels"] = new_out

    ds_convs, ds_bns = _get_channel_dependent_children(new_graph, conv_node_id)
    for nid in ds_convs:
        new_graph.nodes[nid].params["in_channels"] = new_out
        if new_graph.nodes[nid].op_type == "sep_conv":
            new_graph.nodes[nid].params["groups"] = new_out
    for nid in ds_bns:
        new_graph.nodes[nid].params["num_features"] = new_out

    return new_graph


def apply_remove_layer(graph: ArchitectureGraph,
                       remove_node_id: int) -> ArchitectureGraph:
    new_graph = graph.clone()
    node      = new_graph.nodes[remove_node_id]

    # Safety: do not attempt to remove channel-altering nodes here
    if node.op_type in ("conv", "sep_conv"):
        raise ValueError(
            f"apply_remove_layer: cannot remove channel-altering op {node.op_type}"
        )

    parents  = list(node.parents)
    children = new_graph.get_children(remove_node_id)

    for child_id in children:
        child_node = new_graph.nodes[child_id]
        new_parents = []
        for p in child_node.parents:
            if p == remove_node_id:
                for inherited in parents:
                    if inherited not in new_parents:
                        new_parents.append(inherited)
            else:
                if p not in new_parents:
                    new_parents.append(p)
        child_node.parents = new_parents

    del new_graph.nodes[remove_node_id]
    return new_graph


def apply_replace_with_sepconv(graph: ArchitectureGraph,
                               conv_node_id: int,
                               kernel_size: int = 3,
                               padding: int = 1) -> ArchitectureGraph:
    new_graph = graph.clone()
    node      = new_graph.nodes[conv_node_id]
    in_ch     = node.params["in_channels"]
    out_ch    = node.params["out_channels"]
    node.op_type = "sep_conv"
    node.params  = {
        "in_channels":  in_ch,
        "out_channels": out_ch,
        "kernel_size":  kernel_size,
        "padding":      padding,
        "groups":       in_ch,
    }
    return new_graph


# =============================================================================
# inherit_weights_* functions: operate on built nn.Module objects
# FIX: these now receive ArchitectureGraph directly, NOT CompiledModel.
# =============================================================================

def inherit_weights_prune(parent_model: nn.Module, child_model: nn.Module,
                          graph: ArchitectureGraph, conv_node_id: int,
                          keep_indices=None):
    """
    Copy the highest-magnitude filters from parent_model to child_model
    after a prune operation.

    Parameters
    ----------
    parent_model   : trained parent (nn.Module)
    child_model    : child after prune (nn.Module, smaller conv at conv_node_id)
    graph          : the CHILD's ArchitectureGraph (used for topology traversal)
    conv_node_id   : the pruned conv node id
    keep_indices   : optional pre-computed filter indices to keep
    """
    parent_layers = parent_model.layers
    child_layers  = child_model.layers
    key           = str(conv_node_id)

    if key not in parent_layers or key not in child_layers:
        return

    p_conv = parent_layers[key]
    c_conv = child_layers[key]

    with torch.no_grad():
        p_w     = p_conv.weight.detach().cpu().numpy()
        old_out = p_w.shape[0]
        new_out = c_conv.weight.shape[0]

        if keep_indices is None:
            norms        = np.abs(p_w).sum(axis=(1, 2, 3))
            safe_new_out = min(new_out, old_out)
            keep_indices = np.sort(np.argsort(-norms)[:safe_new_out])

        for i_new, i_old in enumerate(keep_indices):
            if i_new >= c_conv.weight.shape[0]:
                break
            c_conv.weight[i_new].copy_(p_conv.weight[i_old])
            if p_conv.bias is not None and c_conv.bias is not None:
                c_conv.bias[i_new].copy_(p_conv.bias[i_old])

        # FIX: use the supplied ArchitectureGraph, NOT child_model.graph
        ds_convs, ds_bns = _get_channel_dependent_children(graph, conv_node_id)

        for ds_id in ds_convs:
            ds_key = str(ds_id)
            if ds_key not in parent_layers or ds_key not in child_layers:
                continue
            p_mod = parent_layers[ds_key]
            c_mod = child_layers[ds_key]

            p_ds = p_mod[0] if isinstance(p_mod, nn.Sequential) else p_mod
            c_ds = c_mod[0] if isinstance(c_mod, nn.Sequential) else c_mod
            is_dw = (hasattr(p_ds, "groups") and
                     p_ds.groups == p_ds.in_channels)

            with torch.no_grad():
                if is_dw:
                    for i_new, i_old in enumerate(keep_indices):
                        if i_new >= c_ds.weight.shape[0]:
                            break
                        c_ds.weight[i_new].copy_(p_ds.weight[i_old])
                else:
                    src = p_ds.weight.detach().cpu().numpy()
                    valid = [idx for idx in keep_indices if idx < src.shape[1]]
                    for o in range(c_ds.weight.shape[0]):
                        new_row = torch.tensor(src[o][valid],
                                               dtype=c_ds.weight.dtype)
                        lim = min(new_row.shape[0], c_ds.weight.shape[1])
                        c_ds.weight[o, :lim, :, :].copy_(
                            new_row[:lim].unsqueeze(-1).unsqueeze(-1)
                        )
                if p_ds.bias is not None and c_ds.bias is not None:
                    c_ds.bias.copy_(p_ds.bias)

        for bn_id in ds_bns:
            bn_key = str(bn_id)
            if bn_key not in parent_layers or bn_key not in child_layers:
                continue
            p_bn = parent_layers[bn_key]
            c_bn = child_layers[bn_key]
            valid = [idx for idx in keep_indices if idx < p_bn.weight.shape[0]]
            lim   = len(valid)
            with torch.no_grad():
                c_bn.weight[:lim].copy_(p_bn.weight[valid])
                c_bn.bias[:lim].copy_(p_bn.bias[valid])
                c_bn.running_mean[:lim].copy_(p_bn.running_mean[valid])
                c_bn.running_var[:lim].copy_(p_bn.running_var[valid])


def inherit_weights_sepconv(parent_model: nn.Module, child_model: nn.Module,
                             conv_node_id: int):
    """
    Initialise a sep-conv (depthwise + pointwise) from the weights of
    the original conv it replaced, so accuracy is preserved at init.
    """
    key = str(conv_node_id)
    if key not in parent_model.layers or key not in child_model.layers:
        return

    p_mod = parent_model.layers[key]
    c_mod = child_model.layers[key]

    with torch.no_grad():
        p_w = p_mod.weight.detach()
        out_c, in_c, kh, kw = p_w.shape

        if not isinstance(c_mod, nn.Sequential) or len(c_mod) < 2:
            return

        depth, point = c_mod[0], c_mod[1]

        # Depthwise: identity-like init (centre pixel = 1.0)
        dw_w = torch.zeros_like(depth.weight)
        for i in range(in_c):
            dw_w[i, 0, kh // 2, kw // 2] = 1.0
        depth.weight.copy_(dw_w)
        if depth.bias is not None:
            depth.bias.zero_()

        # Pointwise: spatial average of original conv weights
        pw = p_w.mean(dim=(2, 3)).view(out_c, in_c, 1, 1)
        min_o = min(point.weight.shape[0], pw.shape[0])
        min_i = min(point.weight.shape[1], pw.shape[1])
        point.weight[:min_o, :min_i, 0, 0].copy_(pw[:min_o, :min_i, 0, 0])
        if point.bias is not None:
            if p_mod.bias is not None:
                point.bias[:min_o].copy_(p_mod.bias[:min_o])
            else:
                point.bias.zero_()