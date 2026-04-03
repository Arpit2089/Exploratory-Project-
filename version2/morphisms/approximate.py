################################################################################
# FOLDER: morphisms
# FILE:   approximate.py
# PATH:   .\morphisms\approximate.py
################################################################################

import torch
import torch.nn as nn
import numpy as np
from utils.logger import get_logger
from architectures.node import Node
from architectures.graph import ArchitectureGraph
from morphisms.exact import _get_channel_dependent_children

logger = get_logger("morphisms_approx", logfile="logs/morphisms_approx.log")

def apply_prune_filters(graph: ArchitectureGraph, conv_node_id: int, keep_ratio: float = 0.5):
    new_graph = graph.clone()
    conv = new_graph.nodes[conv_node_id]

    old_out = conv.params['out_channels']
    new_out = max(1, int(old_out * keep_ratio))

    conv.params['out_channels'] = new_out

    ds_convs, ds_bns = _get_channel_dependent_children(new_graph, conv_node_id)
    
    for nid in ds_convs:
        new_graph.nodes[nid].params['in_channels'] = new_out
        if new_graph.nodes[nid].op_type == 'sep_conv':
            new_graph.nodes[nid].params['groups'] = new_out 
            
    for nid in ds_bns:
        new_graph.nodes[nid].params['num_features'] = new_out

    return new_graph

def inherit_weights_prune(parent_model: nn.Module, child_model: nn.Module, conv_node_id: int, keep_indices=None):
    parent_layers = parent_model.layers
    child_layers = child_model.layers
    key = str(conv_node_id)
    
    if key not in parent_layers or key not in child_layers:
        return

    p_conv = parent_layers[key]
    c_conv = child_layers[key]
    
    with torch.no_grad():
        p_w = p_conv.weight.detach().cpu().numpy()  
        old_out = p_w.shape[0]
        new_out = c_conv.weight.shape[0]

        if keep_indices is None:
            norms = np.abs(p_w).sum(axis=(1,2,3))
            # Safely cap the new_out to the old_out limit
            safe_new_out = min(new_out, old_out)
            keep_indices = np.argsort(-norms)[:safe_new_out]
            keep_indices = np.sort(keep_indices)

        for i_new, i_old in enumerate(keep_indices):
            if i_new >= c_conv.weight.shape[0]: break
            c_conv.weight[i_new].copy_(p_conv.weight[i_old])
            if p_conv.bias is not None and c_conv.bias is not None:
                c_conv.bias[i_new].copy_(p_conv.bias[i_old])

        ds_convs, ds_bns = _get_channel_dependent_children(child_model.graph, conv_node_id)

        for ds_id in ds_convs:
            ds_key = str(ds_id)
            if ds_key in parent_layers and ds_key in child_layers:
                p_mod = parent_layers[ds_key]
                c_mod = child_layers[ds_key]
                
                p_ds_conv = p_mod[0] if isinstance(p_mod, nn.Sequential) else p_mod
                c_ds_conv = c_mod[0] if isinstance(c_mod, nn.Sequential) else c_mod
                
                is_depthwise = p_ds_conv.groups == p_ds_conv.in_channels

                if is_depthwise:
                    for i_new, i_old in enumerate(keep_indices):
                        if i_new >= c_ds_conv.weight.shape[0]: break
                        c_ds_conv.weight[i_new].copy_(p_ds_conv.weight[i_old])
                else:
                    for o in range(c_ds_conv.weight.shape[0]):
                        src = p_ds_conv.weight[o].detach().cpu().numpy()
                        # Safe array sizing
                        valid_indices = [idx for idx in keep_indices if idx < src.shape[0]]
                        new_src = torch.tensor(src[valid_indices], dtype=c_ds_conv.weight.dtype)
                        c_ds_conv.weight[o, :new_src.shape[0], :, :].copy_(new_src)
                
                if p_ds_conv.bias is not None and c_ds_conv.bias is not None:
                    c_ds_conv.bias.copy_(p_ds_conv.bias)

        for bn_id in ds_bns:
            bn_key = str(bn_id)
            if bn_key in parent_layers and bn_key in child_layers:
                p_bn = parent_layers[bn_key]
                c_bn = child_layers[bn_key]
                
                valid_indices = [idx for idx in keep_indices if idx < p_bn.weight.shape[0]]
                limit = len(valid_indices)
                
                c_bn.weight[:limit].copy_(p_bn.weight[valid_indices])
                c_bn.bias[:limit].copy_(p_bn.bias[valid_indices])
                c_bn.running_mean[:limit].copy_(p_bn.running_mean[valid_indices])
                c_bn.running_var[:limit].copy_(p_bn.running_var[valid_indices])

def apply_remove_layer(graph: ArchitectureGraph, remove_node_id: int):
    new_graph = graph.clone()
    node = new_graph.nodes[remove_node_id]

    if node.op_type in ('conv', 'bn', 'sep_conv'):
        raise ValueError(f"Unsafe remove: cannot remove channel-altering node type {node.op_type}")

    parents = node.parents.copy()
    children = new_graph.get_children(remove_node_id)

    for child_id in children:
        child = new_graph.nodes[child_id]
        new_parents = []
        for p in child.parents:
            if p == remove_node_id:
                for inherited_parent in parents:
                    if inherited_parent not in new_parents:
                        new_parents.append(inherited_parent)
            else:
                if p not in new_parents:
                    new_parents.append(p)
        child.parents = new_parents

    del new_graph.nodes[remove_node_id]
    return new_graph

def apply_replace_with_sepconv(graph: ArchitectureGraph, conv_node_id: int, kernel_size=3, padding=1):
    new_graph = graph.clone()
    node = new_graph.nodes[conv_node_id]

    in_ch = node.params['in_channels']
    out_ch = node.params['out_channels']
    
    node.op_type = 'sep_conv'
    node.params = {
        'in_channels': in_ch,
        'out_channels': out_ch,
        'kernel_size': kernel_size,
        'padding': padding,
        'groups': in_ch
    }
    
    return new_graph

def inherit_weights_sepconv(parent_model: nn.Module, child_model: nn.Module, conv_node_id: int):
    key = str(conv_node_id)
    if key not in parent_model.layers or key not in child_model.layers:
        return

    p_mod = parent_model.layers[key]
    c_mod = child_model.layers[key]
    
    with torch.no_grad():
        p_w = p_mod.weight.detach() 
        out, inn, kh, kw = p_w.shape
        
        if isinstance(c_mod, nn.Sequential):
            depth = c_mod[0]
            point = c_mod[1]
            
            dw_w = torch.zeros_like(depth.weight)
            ch_center_h, ch_center_w = kh // 2, kw // 2
            
            for i in range(inn):
                dw_w[i, 0, ch_center_h, ch_center_w] = 1.0
            depth.weight.copy_(dw_w)
            
            if hasattr(depth, 'bias') and depth.bias is not None:
                depth.bias.zero_()
                
            pw = p_w.mean(dim=(2,3)).view(out, inn, 1, 1)
            
            if point.weight.shape == pw.shape:
                point.weight.copy_(pw)
            else:
                min_out = min(point.weight.shape[0], pw.shape[0])
                min_in = min(point.weight.shape[1], pw.shape[1])
                point.weight[:min_out, :min_in, 0, 0].copy_(pw[:min_out, :min_in, 0, 0])
                
            if hasattr(point, 'bias') and point.bias is not None:
                if p_mod.bias is not None:
                    point.bias.copy_(p_mod.bias)
                else:
                    point.bias.zero_()

# # morphisms/approximate.py
# import torch
# import torch.nn as nn
# import numpy as np
# from utils.logger import get_logger
# from architectures.node import Node
# from architectures.graph import ArchitectureGraph
# from morphisms.exact import _get_channel_dependent_children

# logger = get_logger("morphisms_approx", logfile="logs/morphisms_approx.log")

# def apply_prune_filters(graph: ArchitectureGraph, conv_node_id: int, keep_ratio: float = 0.5):
#     new_graph = graph.clone()
#     conv = new_graph.nodes[conv_node_id]

#     old_out = conv.params['out_channels']
#     new_out = max(1, int(old_out * keep_ratio))

#     conv.params['out_channels'] = new_out

#     ds_convs, ds_bns = _get_channel_dependent_children(new_graph, conv_node_id)
    
#     for nid in ds_convs:
#         new_graph.nodes[nid].params['in_channels'] = new_out
#         if new_graph.nodes[nid].op_type == 'sep_conv':
#             new_graph.nodes[nid].params['groups'] = new_out 
            
#     for nid in ds_bns:
#         new_graph.nodes[nid].params['num_features'] = new_out

#     return new_graph

# def inherit_weights_prune(parent_model: nn.Module, child_model: nn.Module, conv_node_id: int, keep_indices=None):
#     parent_layers = parent_model.layers
#     child_layers = child_model.layers
#     key = str(conv_node_id)
    
#     if key not in parent_layers or key not in child_layers:
#         return

#     p_conv = parent_layers[key]
#     c_conv = child_layers[key]
    
#     with torch.no_grad():
#         p_w = p_conv.weight.detach().cpu().numpy()  
#         old_out = p_w.shape[0]
#         new_out = c_conv.weight.shape[0]

#         if keep_indices is None:
#             norms = np.abs(p_w).sum(axis=(1,2,3))
#             keep_indices = np.argsort(-norms)[:new_out]
#             keep_indices = np.sort(keep_indices)

#         for i_new, i_old in enumerate(keep_indices):
#             c_conv.weight[i_new].copy_(p_conv.weight[i_old])
#             if p_conv.bias is not None and c_conv.bias is not None:
#                 c_conv.bias[i_new].copy_(p_conv.bias[i_old])

#         ds_convs, ds_bns = _get_channel_dependent_children(child_model.graph, conv_node_id)

#         for ds_id in ds_convs:
#             ds_key = str(ds_id)
#             if ds_key in parent_layers and ds_key in child_layers:
#                 p_mod = parent_layers[ds_key]
#                 c_mod = child_layers[ds_key]
                
#                 # Unwrap Sequential for sep_conv safely
#                 p_ds_conv = p_mod[0] if isinstance(p_mod, nn.Sequential) else p_mod
#                 c_ds_conv = c_mod[0] if isinstance(c_mod, nn.Sequential) else c_mod
                
#                 is_depthwise = p_ds_conv.groups == p_ds_conv.in_channels

#                 if is_depthwise:
#                     # Slice dim 0 for depthwise
#                     for i_new, i_old in enumerate(keep_indices):
#                         c_ds_conv.weight[i_new].copy_(p_ds_conv.weight[i_old])
#                 else:
#                     # Slice dim 1 for standard conv
#                     for o in range(c_ds_conv.weight.shape[0]):
#                         src = p_ds_conv.weight[o].detach().cpu().numpy()
#                         new_src = torch.tensor(src[keep_indices], dtype=c_ds_conv.weight.dtype)
#                         c_ds_conv.weight[o, :new_src.shape[0], :, :].copy_(new_src)
                
#                 if p_ds_conv.bias is not None and c_ds_conv.bias is not None:
#                     c_ds_conv.bias.copy_(p_ds_conv.bias)

#         for bn_id in ds_bns:
#             bn_key = str(bn_id)
#             if bn_key in parent_layers and bn_key in child_layers:
#                 p_bn = parent_layers[bn_key]
#                 c_bn = child_layers[bn_key]
                
#                 c_bn.weight.copy_(p_bn.weight[keep_indices])
#                 c_bn.bias.copy_(p_bn.bias[keep_indices])
#                 c_bn.running_mean.copy_(p_bn.running_mean[keep_indices])
#                 c_bn.running_var.copy_(p_bn.running_var[keep_indices])

# def apply_remove_layer(graph: ArchitectureGraph, remove_node_id: int):
#     new_graph = graph.clone()
#     node = new_graph.nodes[remove_node_id]

#     if node.op_type in ('conv', 'bn', 'sep_conv'):
#         raise ValueError(f"Unsafe remove: cannot remove channel-altering node type {node.op_type}")

#     parents = node.parents.copy()
#     children = new_graph.get_children(remove_node_id)

#     for child_id in children:
#         child = new_graph.nodes[child_id]
#         new_parents = []
#         for p in child.parents:
#             if p == remove_node_id:
#                 for inherited_parent in parents:
#                     if inherited_parent not in new_parents:
#                         new_parents.append(inherited_parent)
#             else:
#                 if p not in new_parents:
#                     new_parents.append(p)
#         child.parents = new_parents

#     del new_graph.nodes[remove_node_id]
#     return new_graph

# def apply_replace_with_sepconv(graph: ArchitectureGraph, conv_node_id: int, kernel_size=3, padding=1):
#     new_graph = graph.clone()
#     node = new_graph.nodes[conv_node_id]

#     in_ch = node.params['in_channels']
#     out_ch = node.params['out_channels']
    
#     node.op_type = 'sep_conv'
#     node.params = {
#         'in_channels': in_ch,
#         'out_channels': out_ch,
#         'kernel_size': kernel_size,
#         'padding': padding,
#         'groups': in_ch
#     }
    
#     return new_graph

# def inherit_weights_sepconv(parent_model: nn.Module, child_model: nn.Module, conv_node_id: int):
#     key = str(conv_node_id)
#     if key not in parent_model.layers or key not in child_model.layers:
#         return

#     p_mod = parent_model.layers[key]
#     c_mod = child_model.layers[key]
    
#     with torch.no_grad():
#         p_w = p_mod.weight.detach() 
#         out, inn, kh, kw = p_w.shape
        
#         if isinstance(c_mod, nn.Sequential):
#             depth = c_mod[0]
#             point = c_mod[1]
            
#             dw_w = torch.zeros_like(depth.weight)
#             ch_center_h, ch_center_w = kh // 2, kw // 2
            
#             for i in range(inn):
#                 dw_w[i, 0, ch_center_h, ch_center_w] = 1.0
#             depth.weight.copy_(dw_w)
            
#             if hasattr(depth, 'bias') and depth.bias is not None:
#                 depth.bias.zero_()
                
#             pw = p_w.mean(dim=(2,3)).view(out, inn, 1, 1)
            
#             if point.weight.shape == pw.shape:
#                 point.weight.copy_(pw)
#             else:
#                 min_out = min(point.weight.shape[0], pw.shape[0])
#                 min_in = min(point.weight.shape[1], pw.shape[1])
#                 point.weight[:min_out, :min_in, 0, 0].copy_(pw[:min_out, :min_in, 0, 0])
                
#             if hasattr(point, 'bias') and point.bias is not None:
#                 if p_mod.bias is not None:
#                     point.bias.copy_(p_mod.bias)
#                 else:
#                     point.bias.zero_()