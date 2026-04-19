# models/resnet_graph.py
# =============================================================================
# ResNet18 backbone as an ArchitectureGraph.
#
# When USE_PRETRAINED_BACKBONE=True, this module:
#   1. Builds a ResNet18 architecture using the SAME Node/Graph system as
#      BaseNet — so ALL morphism operators (net2deeper, net2wider, prune,
#      skip, sepconv, remove) work on it without any changes.
#   2. Loads torchvision's ImageNet-pretrained weights into the compiled model,
#      mapping torchvision layer names → CompiledModel node IDs.
#
# Dataset adaptations:
#   TINY-IMAGENET  : 3×3 stem conv, stride=1 (no maxpool) — appropriate for 64×64.
#                    Layers 2/3/4 still have stride=2, giving 64→32→16→8 before head.
#   IMAGENET       : 7×7 stem conv, stride=2 + maxpool — standard ResNet18.
#   CIFAR-10/100   : 3×3 stem conv, stride=1 — but no pretrained weights are loaded
#                    (pretrained ResNet18 was trained on 224×224 ImageNet images).
#
# Pretrained weight strategy:
#   - Layers 1-4 (all 3×3 convs): shapes match → weights copied directly.
#   - Stem conv: shapes mismatch (7×7 pretrained vs our 3×3) → random init.
#   - Final fc:  class count differs (1000 pretrained vs 200 target) → random init.
#   - Result: NAS starts with strong feature extractors from ImageNet, only the
#     stem and head need to learn from scratch. Converges much faster than
#     training from random initialization.
# =============================================================================

import torch
import torch.nn as nn
from architectures.node import Node
from architectures.graph import ArchitectureGraph
from utils.logger import get_logger

logger = get_logger("resnet_graph", logfile="logs/resnet_graph.log")


class _GraphBuilder:
    """Thin helper: adds nodes with auto-incrementing IDs."""
    def __init__(self):
        self.g          = ArchitectureGraph()
        self.node_count = 0

    def add_node(self, op_type, params, parents):
        node = Node(self.node_count, op_type, params, parents)
        self.g.add_node(node)
        nid = self.node_count
        self.node_count += 1
        return nid


def _add_basic_block(builder, node_map, C_in, C_out, stride, parent_id, layer_name):
    """
    Add one ResNet18 BasicBlock to the graph and record node IDs in node_map.

    Structure:
        conv1 → bn1 → relu → conv2 → bn2
                                        ↘
                                         add → relu  (output)
                                        ↗
                     shortcut (identity or 1×1 proj)

    Parameters
    ----------
    builder    : _GraphBuilder
    node_map   : dict — updated in-place with {torchvision_prefix: str(node_id)}
    C_in       : input channels
    C_out      : output channels
    stride     : stride for conv1 (and shortcut projection if needed)
    parent_id  : node_id of the input tensor
    layer_name : e.g. "layer1.0"

    Returns
    -------
    node_id of the output relu
    """
    # Main path
    conv1_id = builder.add_node('conv', {
        'in_channels': C_in, 'out_channels': C_out,
        'kernel_size': 3, 'stride': stride, 'padding': 1,
    }, [parent_id])
    node_map[f"{layer_name}.conv1"] = str(conv1_id)

    bn1_id = builder.add_node('bn', {'num_features': C_out}, [conv1_id])
    node_map[f"{layer_name}.bn1"] = str(bn1_id)

    relu1_id = builder.add_node('relu', {}, [bn1_id])

    conv2_id = builder.add_node('conv', {
        'in_channels': C_out, 'out_channels': C_out,
        'kernel_size': 3, 'stride': 1, 'padding': 1,
    }, [relu1_id])
    node_map[f"{layer_name}.conv2"] = str(conv2_id)

    bn2_id = builder.add_node('bn', {'num_features': C_out}, [conv2_id])
    node_map[f"{layer_name}.bn2"] = str(bn2_id)

    # Shortcut path
    if stride != 1 or C_in != C_out:
        # Projection: 1×1 conv + BN to match shape
        proj_conv_id = builder.add_node('conv', {
            'in_channels': C_in, 'out_channels': C_out,
            'kernel_size': 1, 'stride': stride, 'padding': 0,
        }, [parent_id])
        node_map[f"{layer_name}.downsample.0"] = str(proj_conv_id)

        proj_bn_id = builder.add_node('bn', {'num_features': C_out}, [proj_conv_id])
        node_map[f"{layer_name}.downsample.1"] = str(proj_bn_id)

        shortcut_id = proj_bn_id
    else:
        # Identity shortcut — direct connection from block input
        shortcut_id = parent_id

    # Merge: add main path + shortcut
    add_id  = builder.add_node('add',  {}, [bn2_id, shortcut_id])
    relu_id = builder.add_node('relu', {}, [add_id])
    return relu_id


def build_resnet18_graph(num_classes: int = 200,
                         dataset_type: str = "TINY-IMAGENET"):
    """
    Build a ResNet18 architecture as an ArchitectureGraph.

    The returned graph is fully compatible with CompiledModel and all
    LEMONADE morphism operators — it's just a larger graph than BaseNet.

    Parameters
    ----------
    num_classes  : output classes (200 for Tiny ImageNet)
    dataset_type : "TINY-IMAGENET" | "IMAGENET" | "CIFAR-10" | "CIFAR-100"

    Returns
    -------
    (graph: ArchitectureGraph, node_map: dict)
        node_map maps torchvision parameter prefixes to CompiledModel layer keys.
        Attached to graph._pretrained_node_map for use by training workers.
    """
    builder  = _GraphBuilder()
    node_map = {}   # torchvision_prefix → str(node_id)

    # ------------------------------------------------------------------
    # 1. Stem — resolution-specific
    # ------------------------------------------------------------------
    if dataset_type == "IMAGENET":
        # Standard ResNet18: 7×7 conv stride-2 → maxpool → 56×56
        stem_conv = builder.add_node('conv', {
            'in_channels': 3, 'out_channels': 64,
            'kernel_size': 7, 'stride': 2, 'padding': 3,
        }, [])
        node_map["conv1"] = str(stem_conv)   # matches pretrained 7×7 shape

        stem_bn = builder.add_node('bn', {'num_features': 64}, [stem_conv])
        node_map["bn1"] = str(stem_bn)

        stem_relu = builder.add_node('relu', {}, [stem_bn])
        curr = builder.add_node('maxpool', {'kernel_size': 3, 'stride': 2},
                                [stem_relu])
        # After stem: 56×56
    else:
        # Tiny ImageNet (64×64) and CIFAR: 3×3 conv, no stride, no maxpool
        # Pretrained 7×7 weights don't apply → NOT added to node_map
        stem_conv = builder.add_node('conv', {
            'in_channels': 3, 'out_channels': 64,
            'kernel_size': 3, 'stride': 1, 'padding': 1,
        }, [])
        stem_bn   = builder.add_node('bn',   {'num_features': 64}, [stem_conv])
        curr      = builder.add_node('relu', {}, [stem_bn])
        # After stem: 64×64 (Tiny) or 32×32 (CIFAR)

    # ------------------------------------------------------------------
    # 2. ResNet18 Layers (8 BasicBlocks total)
    # ------------------------------------------------------------------
    # Layer 1: 64→64, stride=1, no downsampling (identity shortcuts)
    curr = _add_basic_block(builder, node_map, 64,  64,  1, curr, "layer1.0")
    curr = _add_basic_block(builder, node_map, 64,  64,  1, curr, "layer1.1")

    # Layer 2: 64→128, first block stride=2 (projection shortcut)
    curr = _add_basic_block(builder, node_map, 64,  128, 2, curr, "layer2.0")
    curr = _add_basic_block(builder, node_map, 128, 128, 1, curr, "layer2.1")

    # Layer 3: 128→256, first block stride=2 (projection shortcut)
    curr = _add_basic_block(builder, node_map, 128, 256, 2, curr, "layer3.0")
    curr = _add_basic_block(builder, node_map, 256, 256, 1, curr, "layer3.1")

    # Layer 4: 256→512, first block stride=2 (projection shortcut)
    curr = _add_basic_block(builder, node_map, 256, 512, 2, curr, "layer4.0")
    curr = _add_basic_block(builder, node_map, 512, 512, 1, curr, "layer4.1")

    # ------------------------------------------------------------------
    # 3. Head — global pool → flatten → classify
    # Spatial size reaching the head:
    #   TINY-IMAGENET (64×64): layer2 halves to 32, layer3→16, layer4→8   → pool_size=8
    #   IMAGENET (56×56 after stem): layer2→28, layer3→14, layer4→7       → pool_size=7
    #   CIFAR-10/100 (32×32):  layer2→16, layer3→8, layer4→4              → pool_size=4
    # ------------------------------------------------------------------
    pool_size = {
        "IMAGENET":       7,
        "TINY-IMAGENET":  8,
        "CIFAR-10":       4,
        "CIFAR-100":      4,
    }.get(dataset_type, 8)

    pool_id  = builder.add_node('maxpool',  {'kernel_size': pool_size,
                                             'stride':      pool_size}, [curr])
    flat_id  = builder.add_node('flatten',  {}, [pool_id])
    out_id   = builder.add_node('linear',   {'in_features':  512,
                                             'out_features': num_classes}, [flat_id])
    # fc NOT added to node_map — class count differs from pretrained 1000

    builder.g.set_output(out_id)
    graph = builder.g

    # Attach node_map to graph so training workers can access it after pickling
    graph._pretrained_node_map = node_map

    logger.info(
        "Built ResNet18 graph: %d nodes, dataset=%s, classes=%d",
        len(graph.nodes), dataset_type, num_classes,
    )
    return graph, node_map


# =============================================================================
# Pretrained weight loader
# =============================================================================

def load_pretrained_resnet18_weights(compiled_model: nn.Module,
                                     node_map: dict,
                                     num_classes: int = 200) -> nn.Module:
    """
    Load torchvision's ImageNet-pretrained ResNet18 weights into a CompiledModel.

    Only layers whose shapes match (layers 1-4) are loaded.
    The stem conv (7×7→3×3 mismatch) and final fc (1000→N classes) are
    left with their random initialisation.

    Parameters
    ----------
    compiled_model : CompiledModel — freshly built from build_resnet18_graph()
    node_map       : dict — returned by build_resnet18_graph()
    num_classes    : target class count (used only for logging)

    Returns
    -------
    compiled_model with pretrained weights loaded in-place
    """
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        logger.info("Downloading/loading pretrained ResNet18 (ImageNet1K)…")
        pretrained    = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        pretrained_sd = pretrained.state_dict()
    except Exception as e:
        logger.warning(
            "Could not load pretrained ResNet18 weights: %s\n"
            "→ Starting from random initialisation.", e
        )
        print(f"  [WARN] Pretrained ResNet18 unavailable ({e}). Using random init.")
        return compiled_model

    our_sd  = compiled_model.state_dict()
    new_sd  = {k: v.clone() for k, v in our_sd.items()}   # copy our weights

    loaded = skipped_shape = skipped_missing = 0
    param_suffixes = [
        '.weight', '.bias',
        '.running_mean', '.running_var', '.num_batches_tracked',
    ]

    for tv_prefix, node_id in node_map.items():
        for suffix in param_suffixes:
            tv_key  = tv_prefix + suffix
            our_key = f"layers.{node_id}{suffix}"

            if tv_key not in pretrained_sd:
                skipped_missing += 1
                continue
            if our_key not in our_sd:
                skipped_missing += 1
                continue

            tv_val  = pretrained_sd[tv_key]
            our_val = our_sd[our_key]

            if tv_val.shape == our_val.shape:
                new_sd[our_key] = tv_val.clone().contiguous()
                loaded += 1
            else:
                # Shape mismatch — skip (should only happen for stem on IMAGENET)
                skipped_shape += 1
                logger.debug(
                    "Shape mismatch: %s %s vs %s %s — skipped",
                    tv_key, tv_val.shape, our_key, our_val.shape,
                )

    compiled_model.load_state_dict(new_sd, strict=False)

    del pretrained, pretrained_sd
    import gc; gc.collect()

    total = loaded + skipped_shape
    print(
        f"  [Pretrained] ResNet18 (ImageNet1K) → {num_classes} classes: "
        f"loaded {loaded}/{total} weight tensors  "
        f"(stem + fc initialised randomly)"
    )
    logger.info("Pretrained weights loaded: %d/%d tensors", loaded, total)
    return compiled_model