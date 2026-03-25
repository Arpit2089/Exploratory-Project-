# architectures/compiler.py
import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger("compiler", logfile="logs/compiler.log")


class CompiledModel(nn.Module):
    """
    Compile an ArchitectureGraph into an nn.Module with:
      - ModuleDict of node modules
      - precompiled topological execution plan for fast forward()
      - runtime fixes for Linear and Conv input-shape mismatches caused by graph morphisms
    """
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.layers = nn.ModuleDict()
        logger.info("Initializing CompiledModel")
        self._execution_plan = []
        self._output_node_id = None
        self._build()

    def _build(self):
        logger.info("Building modules for graph with %d nodes", len(self.graph.nodes))

        # create modules for each node
        for node_id, node in self.graph.nodes.items():
            op = getattr(node, "op_type", None)
            if op is None:
                # fallback to older attribute name if present
                op = getattr(node, "op", "").lower()
            op = op.lower()
            key = str(node_id)

            try:
                # ---------------- Conv ----------------
                if op == 'conv':
                    self.layers[key] = nn.Conv2d(
                        node.params['in_channels'],
                        node.params['out_channels'],
                        kernel_size=node.params.get('kernel_size', 3),
                        stride=node.params.get('stride', 1),
                        padding=node.params.get('padding', 1),
                        dilation=node.params.get('dilation', 1),
                        bias=node.params.get('bias', False),
                        groups=node.params.get('groups', 1)
                    )
                    logger.debug("Created Conv2d node %s: in=%d out=%d", key, node.params['in_channels'], node.params['out_channels'])

                # ---------------- Separable Conv ----------------
                elif op in ('sep_conv', 'separableconv2d'):
                    in_c = node.params['in_channels']
                    out_c = node.params['out_channels']
                    k = node.params.get('kernel_size', 3)
                    stride = node.params.get('stride', 1)
                    pad = node.params.get('padding', 1)

                    self.layers[key] = nn.Sequential(
                        nn.Conv2d(in_c, in_c, kernel_size=k, stride=stride, padding=pad, groups=in_c, bias=False),
                        nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                    )
                    logger.debug("Created SeparableConv node %s: in=%d out=%d", key, in_c, out_c)

                # ---------------- BatchNorm ----------------
                elif op == 'bn':
                    self.layers[key] = nn.BatchNorm2d(node.params['num_features'])
                    logger.debug("Created BN node %s: features=%d", key, node.params['num_features'])

                # ---------------- ReLU ----------------
                elif op == 'relu':
                    self.layers[key] = nn.ReLU(inplace=True)
                    logger.debug("Created ReLU node %s", key)

                # ---------------- MaxPool ----------------
                elif op in ('maxpool', 'max_pool', 'max_pool2d'):
                    k = node.params.get('kernel_size', 2)
                    s = node.params.get('stride', k)
                    p = node.params.get('padding', 0)
                    self.layers[key] = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)
                    logger.debug("Created MaxPool node %s: k=%s stride=%s pad=%s", key, k, s, p)

                # ---------------- Flatten ----------------
                elif op == 'flatten':
                    self.layers[key] = nn.Flatten(start_dim=1)
                    logger.debug("Created Flatten node %s", key)

                # ---------------- Linear / FC ----------------
                elif op in ('fc', 'linear'):
                    self.layers[key] = nn.Linear(node.params['in_features'], node.params['out_features'])
                    logger.debug("Created Linear node %s: in=%d out=%d", key, node.params['in_features'], node.params['out_features'])

                # ---------------- Identity & Merge ----------------
                elif op in ('identity', 'add', 'concat'):
                    # placeholder: actual merge logic handled in forward()
                    self.layers[key] = nn.Identity()
                    logger.debug("Created %s node %s", op, key)

                # ---------------- Unknown ----------------
                else:
                    logger.warning("Unknown op_type '%s' for node %s — creating Identity placeholder", op, key)
                    self.layers[key] = nn.Identity()

            except Exception:
                logger.exception("Failed creating module for node %s op=%s params=%s", key, op, getattr(node, "params", None))
                raise

        # compile execution plan once
        self._compile_execution_plan()

    def _compile_execution_plan(self):
        order = self.graph.topological_sort()
        self._execution_plan = []
        for node_id in order:
            node = self.graph.nodes[node_id]
            op = getattr(node, "op_type", None)
            if op is None:
                op = getattr(node, "op", "").lower()
            self._execution_plan.append((node_id, str(node_id), node.parents, op.lower()))
        self._output_node_id = self.graph.output_node

    def forward(self, x):
        cache = {}

        for node_id, layer_key, parents, op in self._execution_plan:
            # Input gathering
            if not parents:
                inp = x
                logger.debug("Node %s has no parents — using graph input", node_id)
            else:
                missing = [p for p in parents if p not in cache]
                if missing:
                    logger.error("Node %s parent(s) %s not in cache. Available keys: %s", node_id, missing, list(cache.keys()))
                    raise KeyError(f"Missing parents for node {node_id}: {missing}")

                if len(parents) == 1:
                    inp = cache[parents[0]]
                else:
                    tensors = [cache[p] for p in parents]
                    if op == 'add':
                        # iterative add to reduce allocations
                        try:
                            inp = tensors[0]
                            for t in tensors[1:]:
                                inp = inp + t
                        except Exception:
                            logger.exception("Add merge failed for node %s with parent shapes: %s", node_id, [t.shape for t in tensors])
                            raise
                    elif op == 'concat':
                        try:
                            inp = torch.cat(tensors, dim=1)
                        except Exception:
                            logger.exception("Concat merge failed for node %s with parent shapes: %s", node_id, [t.shape for t in tensors])
                            raise
                    else:
                        # default to concat
                        try:
                            inp = torch.cat(tensors, dim=1)
                        except Exception:
                            logger.exception("Default concat failed for node %s with parent shapes: %s", node_id, [t.shape for t in tensors])
                            raise
                        logger.debug("Multiple parents for node %s but op=%s — defaulted to concat", node_id, op)

            # Apply layer
            if layer_key not in self.layers:
                logger.error("No layer registered for node %s. Available layers: %s", node_id, list(self.layers.keys()))
                raise KeyError(node_id)

            layer = self.layers[layer_key]

            # --- Runtime fixes for shape mismatches (Conv / Sequential (sep conv) / Linear) ---
            try:
                # Fix Linear: auto-flatten + recreate with correct in_features
                if isinstance(layer, nn.Linear):
                    if inp.dim() > 2:
                        inp = inp.view(inp.size(0), -1)
                        logger.debug("Auto-flattened input for Linear at node %s -> shape %s", node_id, tuple(inp.shape))

                    expected_in = layer.in_features
                    actual_in = inp.size(1)
                    if expected_in != actual_in:
                        logger.info("Linear shape mismatch at node %s: expected %d got %d — recreating Linear(%d -> %d)",
                                    node_id, expected_in, actual_in, actual_in, layer.out_features)
                        new_linear = nn.Linear(actual_in, layer.out_features)
                        try:
                            new_linear = new_linear.to(layer.weight.device)
                        except Exception:
                            pass
                        self.layers[layer_key] = new_linear
                        layer = new_linear

                # Fix Conv2d: if inp channels != conv.in_channels, recreate conv to accept actual channels.
                # preserve kernel/stride/pad/dilation/groups/out_features
                if isinstance(layer, nn.Conv2d):
                    expected_in = layer.in_channels
                    actual_in = inp.size(1)
                    if expected_in != actual_in:
                        logger.info("Conv mismatch at node %s: expected %d got %d — recreating Conv2d to accept %d channels",
                                    node_id, expected_in, actual_in, actual_in)
                        new_conv = nn.Conv2d(
                            actual_in,
                            layer.out_channels,
                            kernel_size=layer.kernel_size,
                            stride=layer.stride,
                            padding=layer.padding,
                            dilation=layer.dilation,
                            groups=layer.groups if hasattr(layer, 'groups') else 1,
                            bias=(layer.bias is not None)
                        )
                        try:
                            new_conv = new_conv.to(next(layer.parameters()).device)
                        except Exception:
                            pass
                        self.layers[layer_key] = new_conv
                        layer = new_conv

                # Fix Sequential sepconv (depthwise+pointwise) if first conv expects in_channels mismatch
                if isinstance(layer, nn.Sequential):
                    # try detect a Conv2d at [0]
                    if len(layer) > 0 and isinstance(layer[0], nn.Conv2d):
                        depth = layer[0]
                        expected_in = depth.in_channels
                        actual_in = inp.size(1)
                        if expected_in != actual_in:
                            logger.info("Sequential (sepconv) mismatch at node %s: expected %d got %d — rebuilding Sequential block",
                                        node_id, expected_in, actual_in)
                            # Rebuild sequential: keep same kernel/stride/padding/groups for depthwise; groups may need set to actual_in for depthwise behaviour
                            k = depth.kernel_size
                            s = depth.stride
                            p = depth.padding
                            dilation = depth.dilation
                            # choose groups = actual_in to be depthwise (best-effort)
                            new_depth = nn.Conv2d(actual_in, actual_in, kernel_size=k, stride=s, padding=p, dilation=dilation, groups=actual_in, bias=(depth.bias is not None))
                            # second (pointwise): out_channels should match existing second conv if present, otherwise attempt to keep child shape
                            point = None
                            if len(layer) > 1 and isinstance(layer[1], nn.Conv2d):
                                out_c = layer[1].out_channels
                                new_point = nn.Conv2d(actual_in, out_c, kernel_size=1, bias=(layer[1].bias is not None))
                            else:
                                # fallback: keep identity
                                new_point = nn.Conv2d(actual_in, actual_in, kernel_size=1, bias=False)
                            new_seq = nn.Sequential(new_depth, new_point)
                            try:
                                new_seq = new_seq.to(next(layer.parameters()).device)
                            except Exception:
                                pass
                            self.layers[layer_key] = new_seq
                            layer = new_seq

                # After recreations above, if layer is Linear and input still has spatial dims, flatten it
                if isinstance(layer, nn.Linear) and inp.dim() > 2:
                    inp = inp.view(inp.size(0), -1)

            except Exception:
                logger.exception("Runtime shape-fix failed for node %s op=%s inp_shape=%s", node_id, op, getattr(inp, 'shape', None))
                raise

            # Apply layer (actual forward)
            try:
                out = layer(inp)
            except Exception as exc:
                # Provide detailed log to make debugging easier
                logger.exception("Layer forward failed at node %s op=%s inp_shape=%s layer_type=%s", node_id, op, getattr(inp, 'shape', None), type(layer))
                raise

            cache[node_id] = out
            logger.debug("Node %s produced output shape %s", node_id, tuple(out.shape))

        # return output
        if self._output_node_id not in cache:
            logger.error("Output node %s not computed. Cache keys: %s", self._output_node_id, list(cache.keys()))
            raise KeyError("Output not computed")

        return cache[self._output_node_id]