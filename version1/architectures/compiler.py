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
                if op == 'conv':
                    self.layers[key] = nn.Conv2d(
                        node.params['in_channels'],
                        node.params['out_channels'],
                        kernel_size=node.params.get('kernel', 3),
                        stride=node.params.get('stride', 1),
                        padding=node.params.get('padding', 1),
                        bias=False,
                        groups=node.params.get('groups', 1)
                    )
                    logger.debug("Created Conv2d node %s: in=%d out=%d k=%s s=%s p=%s groups=%s",
                                 key,
                                 node.params['in_channels'],
                                 node.params['out_channels'],
                                 node.params.get('kernel', 3),
                                 node.params.get('stride', 1),
                                 node.params.get('padding', 1),
                                 node.params.get('groups',1))
                elif op == 'sep_conv' or op == 'separableconv2d':
                    # approximated as depthwise + pointwise
                    in_c = node.params['in_channels']
                    out_c = node.params['out_channels']
                    k = node.params.get('kernel', 3)
                    stride = node.params.get('stride', 1)
                    pad = node.params.get('padding', 1)
                    self.layers[key] = nn.Sequential(
                        nn.Conv2d(in_c, in_c, kernel_size=k, stride=stride, padding=pad, groups=in_c, bias=False),
                        nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                    )
                    logger.debug("Created SeparableConv node %s: in=%d out=%d", key, in_c, out_c)
                elif op == 'bn':
                    self.layers[key] = nn.BatchNorm2d(node.params['num_features'])
                    logger.debug("Created BN node %s: features=%d", key, node.params['num_features'])
                elif op == 'relu':
                    self.layers[key] = nn.ReLU(inplace=True)
                    logger.debug("Created ReLU node %s", key)
                elif op == 'identity':
                    self.layers[key] = nn.Identity()
                    logger.debug("Created Identity node %s", key)
                elif op == 'fc' or op == 'linear':
                    self.layers[key] = nn.Linear(node.params['in_features'], node.params['out_features'])
                    logger.debug("Created Linear node %s: in=%d out=%d", key, node.params['in_features'], node.params['out_features'])
                elif op in ('add', 'concat'):
                    # add/concat have no parameters
                    self.layers[key] = nn.Identity()
                    logger.debug("Created merge node %s (op=%s)", key, op)
                else:
                    logger.warning("Unknown op_type '%s' for node %s — creating Identity placeholder", op, key)
                    self.layers[key] = nn.Identity()
            except Exception as e:
                logger.exception("Failed creating module for node %s op=%s params=%s", key, op, node.params)
                raise

    def forward(self, x):
        cache = {}
        order = self.graph.topological_sort()
        logger.info("Forward pass order: %s", order)

        for node_id in order:
            node = self.graph.nodes[node_id]
            op = node.op_type.lower()
            parents = node.parents

            # Gather inputs
            if not parents:
                inp = x
                logger.debug("Node %s has no parents — using graph input", node_id)
            else:
                # Check parents resolved
                missing = [p for p in parents if p not in cache]
                if missing:
                    logger.error("Node %s parent(s) %s not in cache. Available keys: %s", node_id, missing, list(cache.keys()))
                    raise KeyError(f"Missing parents for node {node_id}: {missing}")

                if len(parents) == 1:
                    inp = cache[parents[0]]
                else:
                    # multiple parents -> either add or concat
                    if op == 'add':
                        # ensure same shape
                        tensors = [cache[p] for p in parents]
                        try:
                            inp = torch.stack(tensors, dim=0).sum(dim=0)
                        except Exception as e:
                            logger.exception("Add merge failed for node %s with parent shapes: %s", node_id, [t.shape for t in tensors])
                            raise
                    elif op == 'concat':
                        tensors = [cache[p] for p in parents]
                        try:
                            inp = torch.cat(tensors, dim=1)  # concat along channel dim
                        except Exception as e:
                            logger.exception("Concat merge failed for node %s with parent shapes: %s", node_id, [t.shape for t in tensors])
                            raise
                    else:
                        # default: if op not merge, but multiple parents, try concat
                        tensors = [cache[p] for p in parents]
                        inp = torch.cat(tensors, dim=1)
                        logger.debug("Multiple parents for node %s but op=%s — defaulted to concat", node_id, op)

            # apply layer
            layer_key = str(node_id)
            if layer_key not in self.layers:
                logger.error("No layer registered for node %s. Available layers: %s",
                            node_id, list(self.layers.keys()))
                raise KeyError(node_id)

            layer = self.layers[layer_key]


            try:
                out = layer(inp)
            except Exception as e:
                logger.exception("Layer forward failed at node %s op=%s inp_shape=%s", node_id, op, getattr(inp, 'shape', None))
                raise

            cache[node_id] = out
            logger.debug("Node %s produced output shape %s", node_id, tuple(out.shape))

        if self.graph.output_node not in cache:
            logger.error("Output node %s not computed. Cache keys: %s", self.graph.output_node, list(cache.keys()))
            raise KeyError("Output not computed")

        return cache[self.graph.output_node]
