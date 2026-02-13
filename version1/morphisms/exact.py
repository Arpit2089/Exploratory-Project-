# morphisms/exact.py
import torch
import torch.nn as nn
from utils.logger import get_logger
from architectures.node import Node
from architectures.graph import ArchitectureGraph
import math

logger = get_logger("morphisms", logfile="logs/morphisms.log")

def _next_node_id(graph: ArchitectureGraph):
    if not graph.nodes:
        return 0
    return max(graph.nodes.keys()) + 1

def apply_net2deeper(graph: ArchitectureGraph, relu_node_id: int, kernel=1, stride=1, padding=0):
    """
    Insert Conv -> BN -> ReLU *after* the node `relu_node_id` in graph.
    Preconditions:
        - relu_node_id must exist
        - The ReLU node's output channel count must be known:
          we require the node to be followed by convs that reveal the out_channels, or
          the user should ensure the next conv expects channels M. To be safe, we infer
          channels from first child that expects input channels (conv/node with 'in_channels').
    We will:
        - clone graph
        - create conv node with in_channels = out_channels = inferred
        - create BN node with num_features = in_channels
        - create ReLU node
        - rewire children of relu_node_id to now have new_relu_node_id as parent
    Returns: new_graph (ArchitectureGraph)
    """
    new_graph = graph.clone()
    if relu_node_id not in new_graph.nodes:
        raise KeyError(f"relu node {relu_node_id} not in graph")

    # infer channels: look for a child that expects in_channels (conv or sep_conv)
    inferred_ch = None
    old_children = []
    for nid, node in new_graph.nodes.items():
        if relu_node_id in node.parents:
            old_children.append(nid)
            # search node.params for 'in_channels' if present
            if 'in_channels' in node.params and inferred_ch is None:
                inferred_ch = node.params['in_channels']

    if inferred_ch is None:
        # fallback: try to check relu node itself (if it followed a BN/conv)
        parent_node = new_graph.nodes[relu_node_id]
        # scan parents of this relu for conv with out_channels
        for p in parent_node.parents:
            pn = new_graph.nodes[p]
            if 'out_channels' in pn.params:
                inferred_ch = pn.params['out_channels']
                break

    if inferred_ch is None:
        raise ValueError("Could not infer channel size. Ensure graph nodes expose 'in_channels' or 'out_channels'.")

    # create new node ids
    conv_id = _next_node_id(new_graph)
    bn_id = conv_id + 1
    relu_new_id = conv_id + 2

    logger.info("Applying Net2Deeper: inserting nodes conv=%d bn=%d relu=%d after relu_node=%d (channels=%d)",
                conv_id, bn_id, relu_new_id, relu_node_id, inferred_ch)

    # Create conv node params: keep same in/out channels so we can initialize as identity
    conv_params = {
        'in_channels': inferred_ch,
        'out_channels': inferred_ch,
        'kernel': kernel,
        'stride': stride,
        'padding': padding,
        'groups': 1
    }
    new_graph.add_node(Node(conv_id, 'conv', conv_params, parents=[relu_node_id]))

    # BN node
    bn_params = {
        'num_features': inferred_ch
    }
    new_graph.add_node(Node(bn_id, 'bn', bn_params, parents=[conv_id]))

    # ReLU node (new)
    new_graph.add_node(Node(relu_new_id, 'relu', {}, parents=[bn_id]))

    # Rewire old children: replace relu_node_id in their parents with relu_new_id
    for child_id in old_children:
        child = new_graph.nodes[child_id]
        new_parents = []
        replaced = False
        for p in child.parents:
            if p == relu_node_id:
                new_parents.append(relu_new_id)
                replaced = True
            else:
                new_parents.append(p)
        child.parents = new_parents
        logger.debug("Rewired child %d parents: replaced=%s new_parents=%s", child_id, replaced, new_parents)

    return new_graph

def initialize_conv_as_identity(conv_module: nn.Conv2d):
    """
    Initialize 1x1 conv as identity mapping when in_channels == out_channels.
    Works for kernel=1. For larger kernels, center element is set as identity.
    """
    with torch.no_grad():
        w = torch.zeros_like(conv_module.weight)
        out_c, in_c, kh, kw = w.shape
        if out_c != in_c:
            logger.warning("Identity init requested but out_c != in_c (%d != %d). Skipping identity init.", out_c, in_c)
            return
        # center position
        ch_center_h = kh // 2
        ch_center_w = kw // 2
        for i in range(out_c):
            w[i, i, ch_center_h, ch_center_w] = 1.0
        conv_module.weight.copy_(w)
        if conv_module.bias is not None:
            conv_module.bias.zero_()
    logger.info("Initialized Conv module as identity (shape=%s)", tuple(conv_module.weight.shape))

def initialize_bn_as_identity(bn_module: nn.BatchNorm2d):
    with torch.no_grad():
        if hasattr(bn_module, 'weight'):
            bn_module.weight.fill_(1.0)
        if hasattr(bn_module, 'bias'):
            bn_module.bias.zero_()
        # running stats: set to 0 mean, 1 var to act neutrally
        bn_module.running_mean.zero_()
        bn_module.running_var.fill_(1.0)
    logger.info("Initialized BatchNorm as identity (num_features=%d)", bn_module.num_features)

def inherit_weights(parent_model: nn.Module, child_model: nn.Module):
    """
    For any module that exists in both parent_model.layers and child_model.layers under same key,
    try to copy state_dict entries where shapes match.
    """
    parent_layers = getattr(parent_model, 'layers', None)
    child_layers = getattr(child_model, 'layers', None)
    if parent_layers is None or child_layers is None:
        logger.error("Parent or child model has no 'layers' ModuleDict")
        return

    copied = 0
    skipped = 0
    for key in parent_layers.keys():
        if key in child_layers:
            p_mod = parent_layers[key]
            c_mod = child_layers[key]
            p_sd = p_mod.state_dict()
            c_sd = c_mod.state_dict()
            # prepare mapping of items with same key and shape
            to_load = {}
            for name, tensor in p_sd.items():
                if name in c_sd and tensor.shape == c_sd[name].shape:
                    to_load[name] = tensor.clone()
            if to_load:
                try:
                    c_mod.load_state_dict(to_load, strict=False)
                    copied += 1
                    logger.debug("Copied matching params for module %s", key)
                except Exception as e:
                    logger.exception("Failed to load matching params for module %s: %s", key, str(e))
                    skipped += 1
            else:
                skipped += 1
        else:
            skipped += 1
    logger.info("Weight inheritance finished: copied_modules=%d, skipped_modules=%d", copied, skipped)
def apply_net2wider(graph: ArchitectureGraph, conv_node_id: int, widen_by: int = 4):
    """
    Net2WiderNet: widen conv_node_id by `widen_by` filters and update downstream
    channel-dependent nodes (conv.in_channels and bn.num_features).
    Returns a new cloned graph.
    """
    new_graph = graph.clone()

    if conv_node_id not in new_graph.nodes:
        raise KeyError(f"Conv node {conv_node_id} not found")

    conv_node = new_graph.nodes[conv_node_id]
    if conv_node.op_type != 'conv':
        raise ValueError("Net2Wider can only be applied to conv nodes")

    old_out = conv_node.params['out_channels']
    new_out = old_out + widen_by

    logger.info(
        "Applying Net2Wider: node=%d old_out=%d new_out=%d",
        conv_node_id, old_out, new_out
    )

    # 1) Update widened conv's param
    conv_node.params['out_channels'] = new_out

    # 2) Update immediate children that consume channels:
    # - conv children: update in_channels if they expected old_out
    # - bn children: update num_features if they expected old_out
    for nid, node in new_graph.nodes.items():
        if conv_node_id in node.parents:
            if node.op_type == 'conv':
                if node.params.get('in_channels') == old_out:
                    node.params['in_channels'] = new_out
                    logger.debug("Updated downstream conv %d in_channels=%d", nid, new_out)
                else:
                    logger.warning("Downstream conv %d in_channels=%s != expected %d", nid, node.params.get('in_channels'), old_out)
            elif node.op_type == 'bn':
                if node.params.get('num_features') == old_out:
                    node.params['num_features'] = new_out
                    logger.debug("Updated downstream BN %d num_features=%d", nid, new_out)
                else:
                    logger.warning("Downstream BN %d num_features=%s != expected %d", nid, node.params.get('num_features'), old_out)
            elif node.op_type in ('relu', 'identity'):
                # channel-agnostic
                continue
            else:
                logger.warning("Net2Wider: unhandled downstream op %s at node %d", node.op_type, nid)

    return new_graph
def inherit_weights_net2wider(parent_model, child_model, conv_node_id, widen_by):
    """
    Copies weights for Net2Wider:
    - copies original conv filters to the front of the child conv
    - duplicates/randomly picks filters for the new channels
    - updates BN parameters: copy existing entries, initialize new entries
      as neutral (weight=1, bias=0, running_mean=0, running_var=1)
    """
    key = str(conv_node_id)
    parent_layers = parent_model.layers
    child_layers = child_model.layers

    if key not in parent_layers or key not in child_layers:
        logger.error("Conv node %s missing in parent/child models", key)
        return

    p_conv = parent_layers[key]
    c_conv = child_layers[key]

    with torch.no_grad():
        # --- Conv weights ---
        old_w = p_conv.weight  # (old_out, in, kh, kw)
        old_out = old_w.shape[0]
        new_out = c_conv.weight.shape[0]

        # copy old filters
        c_conv.weight[:old_out].copy_(old_w)

        # duplicate some filters for new channels (simple strategy)
        for i in range(old_out, new_out):
            src = torch.randint(0, old_out, (1,)).item()
            c_conv.weight[i].copy_(old_w[src])

        # bias if present
        if p_conv.bias is not None and c_conv.bias is not None:
            # copy existing and set new biases to first bias or 0
            c_conv.bias[:old_out].copy_(p_conv.bias)
            fill_val = p_conv.bias[0] if p_conv.bias.numel() > 0 else 0.0
            c_conv.bias[old_out:].fill_(fill_val)

        logger.info("Net2Wider conv weights copied for node %s (old_out=%d new_out=%d)", key, old_out, new_out)

        # --- BatchNorm weights for the downstream BN child (if exists) ---
        # Strategy: find BN modules that used to have num_features == old_out and now are new_out
        for k in parent_layers.keys():
            if k in child_layers:
                p_mod = parent_layers[k]
                c_mod = child_layers[k]

                # only handle BatchNorm2d modules
                import torch.nn as nn
                if isinstance(p_mod, nn.BatchNorm2d) and isinstance(c_mod, nn.BatchNorm2d):
                    p_sd = p_mod.state_dict()
                    c_sd = c_mod.state_dict()

                    # Prepare new state dict for child BN
                    new_state = {}
                    for name, tensor in p_sd.items():
                        # name e.g. 'weight', 'bias', 'running_mean', 'running_var'
                        if name in c_sd:
                            # If sizes match, copy as many as possible
                            if tensor.shape == c_sd[name].shape:
                                new_state[name] = tensor.clone()
                            else:
                                # tensor is length old_out, c_sd expects new_out
                                if tensor.ndim == 1 and tensor.shape[0] == old_out and c_sd[name].ndim == 1 and c_sd[name].shape[0] == new_out:
                                    # create new vector: copy old then init remainder
                                    out_vec = c_sd[name].clone()
                                    out_vec[:old_out].copy_(tensor)
                                    # init remainder: weights -> 1, bias -> 0, running_mean->0, running_var->1
                                    if name == 'weight':
                                        out_vec[old_out:].fill_(1.0)
                                    elif name == 'bias':
                                        out_vec[old_out:].zero_()
                                    elif name == 'running_mean':
                                        out_vec[old_out:].zero_()
                                    elif name == 'running_var':
                                        out_vec[old_out:].fill_(1.0)
                                    new_state[name] = out_vec
                                else:
                                    # shape mismatch we can't handle: skip
                                    logger.debug("Skipping BN param %s for module %s due to unexpected shapes", name, k)
                        # else name not in child -> skip

                    # load new_state into child BN (non-strict)
                    if new_state:
                        c_mod.load_state_dict(new_state, strict=False)
                        logger.info("Updated BN module %s after widening (copied old entries and initialized new ones)", k)

# def apply_skip_connection(graph: ArchitectureGraph):
#     """
#     Adds a cycle-safe skip connection.
#     Only allows src -> dst if src comes before dst in topo order.
#     """
#     new_graph = graph.clone()

#     topo = new_graph.topological_sort()
#     topo_index = {n: i for i, n in enumerate(topo)}

#     valid_pairs = []

#     for src in topo:
#         for dst in topo:
#             if topo_index[src] < topo_index[dst]:
#                 # dst must not already depend on src
#                 if src not in new_graph.nodes[dst].parents:
#                     valid_pairs.append((src, dst))

#     if not valid_pairs:
#         raise ValueError("No valid skip connections available")

#     src, dst = random.choice(valid_pairs)

#     merge_id = new_graph.next_node_id()

#     logger.info(
#         "Adding skip connection: from=%d to=%d via add_node=%d",
#         src, dst, merge_id
#     )

#     # Create merge node
#     merge_node = Node(
#         merge_id,
#         op_type="merge",
#         params={"op": "add"},
#         parents=[src, dst]
#     )

#     # Rewire dst to take output from merge node
#     new_graph.nodes[dst].parents = [merge_id]
#     new_graph.add_node(merge_node)

#     return new_graph
def apply_skip_connection(
    graph: ArchitectureGraph,
    from_node: int,
    to_node: int,
):
    """
    Adds a skip connection from `from_node` to `to_node`
    by inserting a merge (add) node.
    """

    new_graph = graph.clone()

    # sanity checks
    if from_node == to_node:
        raise ValueError("Skip connection cannot be self-loop")

    if from_node not in new_graph.nodes or to_node not in new_graph.nodes:
        raise ValueError("Invalid node ids for skip connection")

    # prevent cycles
    if from_node in new_graph.nodes[to_node].parents:
        raise ValueError("Skip already exists")

    # create merge node
    new_id = max(new_graph.nodes.keys()) + 1

    merge_node = Node(
        id=new_id,
        op="add",
        params={},
        parents=[from_node, to_node],
    )

    # redirect output of to_node
    for n in new_graph.nodes.values():
        n.parents = [
            new_id if p == to_node else p
            for p in n.parents
        ]

    new_graph.add_node(merge_node)

    if new_graph.output_node == to_node:
        new_graph.set_output(new_id)

    return new_graph
