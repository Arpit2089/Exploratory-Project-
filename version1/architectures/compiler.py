# architectures/compiler.py
import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger("compiler", logfile="logs/compiler.log")


class CompiledModel(nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.layers = nn.ModuleDict()
        logger.info("Initializing CompiledModel")
        self._build()

    def _build(self):
        logger.info("Building modules for graph with %d nodes", len(self.graph.nodes))

        for node_id, node in self.graph.nodes.items():
            op = node.op_type.lower()
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
                        bias=False,
                        groups=node.params.get('groups', 1)
                    )
                    logger.debug(
                        "Created Conv2d node %s: in=%d out=%d",
                        key,
                        node.params['in_channels'],
                        node.params['out_channels'],
                    )

                # ---------------- Separable Conv ----------------
                elif op in ('sep_conv', 'separableconv2d'):
                    in_c = node.params['in_channels']
                    out_c = node.params['out_channels']
                    k = node.params.get('kernel_size', 3)
                    stride = node.params.get('stride', 1)
                    pad = node.params.get('padding', 1)

                    self.layers[key] = nn.Sequential(
                        nn.Conv2d(in_c, in_c, kernel_size=k, stride=stride,
                                  padding=pad, groups=in_c, bias=False),
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

                # ---------------- Flatten ----------------
                elif op == 'flatten':
                    # flatten from channel dimension onward; keep batch dim
                    self.layers[key] = nn.Flatten(start_dim=1)
                    logger.debug("Created Flatten node %s", key)

                # ---------------- Linear / FC ----------------
                elif op in ('fc', 'linear'):
                    # create linear with the declared shape; it's possible later morphisms change upstream
                    self.layers[key] = nn.Linear(
                        node.params['in_features'],
                        node.params['out_features']
                    )
                    logger.debug(
                        "Created Linear node %s: in=%d out=%d",
                        key,
                        node.params['in_features'],
                        node.params['out_features']
                    )

                # ---------------- Identity ----------------
                elif op == 'identity':
                    self.layers[key] = nn.Identity()
                    logger.debug("Created Identity node %s", key)

                # ---------------- Merge Ops ----------------
                elif op in ('add', 'concat'):
                    # represent merge ops as identity placeholders; actual merge handled in forward
                    self.layers[key] = nn.Identity()
                    logger.debug("Created merge node %s (op=%s)", key, op)

                # ---------------- Unknown ----------------
                else:
                    logger.warning(
                        "Unknown op_type '%s' for node %s — creating Identity placeholder",
                        op, key
                    )
                    self.layers[key] = nn.Identity()

            except Exception:
                logger.exception(
                    "Failed creating module for node %s op=%s params=%s",
                    key, op, node.params
                )
                raise

    def forward(self, x):
        cache = {}
        order = self.graph.topological_sort()
        # This was very noisy at INFO level; keep as DEBUG to avoid console overflow
        # logger.debug("Forward pass order: %s", order)

        for node_id in order:
            node = self.graph.nodes[node_id]
            op = node.op_type.lower()
            parents = node.parents

            # -------- Input Gathering --------
            if not parents:
                inp = x
                logger.debug("Node %s has no parents — using graph input", node_id)
            else:
                missing = [p for p in parents if p not in cache]
                if missing:
                    logger.error(
                        "Node %s parent(s) %s not in cache. Available keys: %s",
                        node_id, missing, list(cache.keys())
                    )
                    raise KeyError(f"Missing parents for node {node_id}: {missing}")

                if len(parents) == 1:
                    inp = cache[parents[0]]
                else:
                    tensors = [cache[p] for p in parents]

                    if op == 'add':
                        try:
                            inp = torch.stack(tensors, dim=0).sum(dim=0)
                        except Exception:
                            logger.exception(
                                "Add merge failed for node %s with parent shapes: %s",
                                node_id, [t.shape for t in tensors]
                            )
                            raise

                    elif op == 'concat':
                        try:
                            inp = torch.cat(tensors, dim=1)
                        except Exception:
                            logger.exception(
                                "Concat merge failed for node %s with parent shapes: %s",
                                node_id, [t.shape for t in tensors]
                            )
                            raise
                    else:
                        # default concatenation if op not recognized
                        try:
                            inp = torch.cat(tensors, dim=1)
                        except Exception:
                            logger.exception(
                                "Default concat failed for node %s with parent shapes: %s",
                                node_id, [t.shape for t in tensors]
                            )
                            raise
                        logger.debug(
                            "Multiple parents for node %s but op=%s — defaulted to concat",
                            node_id, op
                        )

            # -------- Apply Layer --------
            layer_key = str(node_id)
            if layer_key not in self.layers:
                logger.error(
                    "No layer registered for node %s. Available layers: %s",
                    node_id, list(self.layers.keys())
                )
                raise KeyError(node_id)

            layer = self.layers[layer_key]

            # If we're about to apply a Linear layer but inp still has spatial dims, flatten it.
            # Also, if Linear's in_features doesn't match, recreate the Linear to match current shape.
            try:
                # special-case: flatten inputs for linear layers
                if isinstance(layer, nn.Linear):
                    if inp.dim() > 2:
                        # reshape to (batch, features)
                        inp = inp.view(inp.size(0), -1)
                        logger.debug("Auto-flattened input for Linear at node %s -> shape %s", node_id, tuple(inp.shape))

                    expected_in = layer.in_features
                    actual_in = inp.size(1)
                    if expected_in != actual_in:
                        # recreate Linear with correct in_features but same out_features
                        logger.info(
                            "Linear shape mismatch at node %s: expected %d got %d — recreating Linear(%d -> %d)",
                            node_id, expected_in, actual_in, actual_in, layer.out_features
                        )
                        # create a new linear layer and put it in the ModuleDict so it's registered
                        new_linear = nn.Linear(actual_in, layer.out_features)
                        # move to same device / dtype as existing layer if possible
                        try:
                            new_linear = new_linear.to(layer.weight.device)
                        except Exception:
                            # fallback: do nothing if moving fails (CPU case)
                            pass
                        self.layers[layer_key] = new_linear
                        layer = new_linear

                # if layer is a Sequential conv block and input is e.g. flattened by mistake, try to re-expand? that's unlikely;
                # we just attempt the normal call and catch exceptions below.
                out = layer(inp)

            except Exception as exc:
                logger.exception(
                    "Layer forward failed at node %s op=%s inp_shape=%s",
                    node_id, op, getattr(inp, 'shape', None)
                )
                raise

            cache[node_id] = out
            # only log debug for per-node outputs to avoid huge output at INFO level
            logger.debug("Node %s produced output shape %s", node_id, tuple(out.shape))

        if self.graph.output_node not in cache:
            logger.error(
                "Output node %s not computed. Cache keys: %s",
                self.graph.output_node, list(cache.keys())
            )
            raise KeyError("Output not computed")

        return cache[self.graph.output_node]
