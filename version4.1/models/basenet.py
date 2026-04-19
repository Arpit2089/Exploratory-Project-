# models/basenet.py
# =============================================================================
# BaseNet graph constructor.
# Produces a different stem and head depending on the input resolution:
#
#   CIFAR-10 / CIFAR-100  : 32×32  → 3×3 conv, no stride → 2 reductions → 8×8
#   TINY-IMAGENET         : 64×64  → 3×3 conv, no stride → 2 reductions → 16×16
#   IMAGENET              : 224×224→ 7×7 conv stride-2 + maxpool → 2 red → 14×14
#
# The head always uses a single MaxPool that collapses the remaining spatial
# dimension to 1×1, followed by Flatten → Linear(num_classes).
# This means the NAS operators (deeper, wider, skip, prune …) can safely
# change intermediate channel counts without touching the head.
# =============================================================================

from architectures.node import Node
from architectures.graph import ArchitectureGraph
from models.cells import add_cell, add_reduction_cell


class GraphBuilder:
    """
    Tiny helper: adds nodes to an ArchitectureGraph and auto-increments IDs.
    """
    def __init__(self):
        self.g          = ArchitectureGraph()
        self.node_count = 0

    def add_node(self, op_type, params, parents):
        node = Node(self.node_count, op_type, params, parents)
        self.g.add_node(node)
        self.node_count += 1
        return node.id


def build_basenet_graph(num_classes: int = 10,
                        dataset_type: str = "CIFAR-10") -> ArchitectureGraph:
    """
    Construct the BaseNet ArchitectureGraph for the requested dataset.

    Parameters
    ----------
    num_classes  : number of output classes
    dataset_type : one of "CIFAR-10", "CIFAR-100", "TINY-IMAGENET", "IMAGENET"

    Returns
    -------
    ArchitectureGraph ready for CompiledModel compilation.
    """
    builder       = GraphBuilder()
    init_channels = 16

    # ------------------------------------------------------------------
    # 1. Stem  (resolution-specific downsampling)
    # ------------------------------------------------------------------
    if dataset_type == "IMAGENET":
        # 224×224 → 7×7 conv stride-2 → 112 → maxpool stride-2 → 56×56
        curr = builder.add_node('conv', {
            'in_channels': 3, 'out_channels': init_channels,
            'kernel_size': 7, 'stride': 2, 'padding': 3,
        }, [])
        curr = builder.add_node('bn',      {'num_features': init_channels}, [curr])
        curr = builder.add_node('relu',    {}, [curr])
        curr = builder.add_node('maxpool', {'kernel_size': 3, 'stride': 2}, [curr])
        # After stem: 56×56

    elif dataset_type == "TINY-IMAGENET":
        # 64×64 → 3×3 conv (no stride) → 64×64
        # Two reduction cells will bring this down to 16×16.
        curr = builder.add_node('conv', {
            'in_channels': 3, 'out_channels': init_channels,
            'kernel_size': 3, 'stride': 1, 'padding': 1,
        }, [])
        curr = builder.add_node('bn',   {'num_features': init_channels}, [curr])
        curr = builder.add_node('relu', {}, [curr])
        # After stem: 64×64

    else:
        # CIFAR-10 / CIFAR-100: 32×32 → 3×3 conv (no stride) → 32×32
        curr = builder.add_node('conv', {
            'in_channels': 3, 'out_channels': init_channels,
            'kernel_size': 3, 'stride': 1, 'padding': 1,
        }, [])
        curr = builder.add_node('bn',   {'num_features': init_channels}, [curr])
        curr = builder.add_node('relu', {}, [curr])
        # After stem: 32×32

    C_curr = init_channels

    # ------------------------------------------------------------------
    # 2. Body: 3 stages, each with a normal cell; stages 0 and 1 end
    #          with a reduction cell that halves spatial dims.
    #
    #   CIFAR / CIFAR-100:  32→16→8   (2 reductions)
    #   TINY-IMAGENET:      64→32→16  (2 reductions)
    #   IMAGENET:           56→28→14  (2 reductions)
    # ------------------------------------------------------------------
    num_cells_per_stage = 1

    for stage in range(3):
        # Normal cell (no spatial change)
        for _ in range(num_cells_per_stage):
            curr = add_cell(builder, C_curr, C_curr, stride=1, parent_id=curr)

        # Reduction cell (halve spatial, double channels)
        if stage < 2:
            C_next = C_curr * 2
            curr   = add_reduction_cell(builder, C_curr, C_next, parent_id=curr)
            C_curr = C_next

    # ------------------------------------------------------------------
    # 3. Head: collapse remaining spatial dims → flatten → classify
    #
    #   Spatial size after body:
    #     CIFAR-10 / CIFAR-100 : 8×8
    #     TINY-IMAGENET        : 16×16
    #     IMAGENET             : 14×14
    # ------------------------------------------------------------------
    if dataset_type == "IMAGENET":
        final_spatial = 14
    elif dataset_type == "TINY-IMAGENET":
        final_spatial = 16
    else:
        # CIFAR-10 and CIFAR-100
        final_spatial = 8

    # Single maxpool collapses exactly to 1×1
    curr = builder.add_node('maxpool', {
        'kernel_size': final_spatial, 'stride': final_spatial,
    }, [curr])
    curr    = builder.add_node('flatten', {}, [curr])
    out_node = builder.add_node('linear', {
        'in_features': C_curr, 'out_features': num_classes,
    }, [curr])

    builder.g.set_output(out_node)
    return builder.g


# ################################################################################
# # FOLDER: models
# # FILE:   basenet.py
# # PATH:   .\models\basenet.py
# ################################################################################

# from architectures.node import Node
# from architectures.graph import ArchitectureGraph
# from models.cells import add_cell, add_reduction_cell

# class GraphBuilder:
#     def __init__(self):
#         self.g = ArchitectureGraph()
#         self.node_count = 0
        
#     def add_node(self, op_type, params, parents):
#         node = Node(self.node_count, op_type, params, parents)
#         self.g.add_node(node)
#         self.node_count += 1
#         return node.id

# def build_basenet_graph(num_classes=10, dataset_type="CIFAR-10"):
#     """
#     Constructs the exact BaseNet architecture dynamically based on the dataset geometry!
#     """
#     builder = GraphBuilder()
#     init_channels = 16
    
#     # ------------------------------------------------------------------
#     # 1. The Stem (Image Size Handling)
#     # ------------------------------------------------------------------
#     if dataset_type == "IMAGENET":
#         # Aggressive downsampling for 224x224 images (Shrinks to 56x56 immediately)
#         curr = builder.add_node('conv', {'in_channels': 3, 'out_channels': init_channels, 'kernel_size': 7, 'stride': 2, 'padding': 3}, [])
#         curr = builder.add_node('bn', {'num_features': init_channels}, [curr])
#         curr = builder.add_node('relu', {}, [curr])
#         curr = builder.add_node('maxpool', {'kernel_size': 3, 'stride': 2}, [curr])
#     else:
#         # Standard CIFAR stem for 32x32 images
#         curr = builder.add_node('conv', {'in_channels': 3, 'out_channels': init_channels, 'kernel_size': 3, 'stride': 1, 'padding': 1}, [])
#         curr = builder.add_node('bn', {'num_features': init_channels}, [curr])
#         curr = builder.add_node('relu', {}, [curr])
        
#     C_curr = init_channels
#     num_cells_per_stage = 1
    
#     # ------------------------------------------------------------------
#     # 2. The Body (Stacked Cells)
#     # ------------------------------------------------------------------
#     for stage in range(3):
#         for _ in range(num_cells_per_stage):
#             curr = add_cell(builder, C_curr, C_curr, stride=1, parent_id=curr)
            
#         if stage < 2:
#             C_next = C_curr * 2
#             curr = add_reduction_cell(builder, C_curr, C_next, parent_id=curr)
#             C_curr = C_next
            
#     # ------------------------------------------------------------------
#     # 3. The Head (Classification)
#     # ------------------------------------------------------------------
#     # Calculate the remaining spatial dimension mathematically
#     final_spatial_size = 14 if dataset_type == "IMAGENET" else 8
    
#     # Reduces the final spatial dimensions down to 1x1 perfectly.
#     curr = builder.add_node('maxpool', {'kernel_size': final_spatial_size, 'stride': final_spatial_size}, [curr])
#     curr = builder.add_node('flatten', {}, [curr])
#     out_node = builder.add_node('linear', {'in_features': C_curr, 'out_features': num_classes}, [curr])
    
#     builder.g.set_output(out_node)
#     return builder.g

# # ################################################################################
# # # FOLDER: models
# # # FILE:   basenet.py
# # # PATH:   .\models\basenet.py
# # ################################################################################

# # from architectures.node import Node
# # from architectures.graph import ArchitectureGraph
# # from models.cells import add_cell, add_reduction_cell

# # class GraphBuilder:
# #     """
# #     Helper utility to sequentially construct ArchitectureGraph nodes 
# #     without needing to manually track IDs.
# #     """
# #     def __init__(self):
# #         self.g = ArchitectureGraph()
# #         self.node_count = 0
        
# #     def add_node(self, op_type, params, parents):
# #         node = Node(self.node_count, op_type, params, parents)
# #         self.g.add_node(node)
# #         self.node_count += 1
# #         return node.id

# # def build_basenet_graph():
# #     """
# #     Constructs the exact BaseNet architecture (Stem -> Cells -> ReductionCells)
# #     as a dynamic, mutable ArchitectureGraph!
# #     """
# #     builder = GraphBuilder()
    
# #     # ------------------------------------------------------------------
# #     # 1. The Stem (Initial convolutions)
# #     # ------------------------------------------------------------------
# #     init_channels = 16
# #     curr = builder.add_node('conv', {'in_channels': 3, 'out_channels': init_channels, 'kernel_size': 3, 'stride': 1, 'padding': 1}, [])
# #     curr = builder.add_node('bn', {'num_features': init_channels}, [curr])
# #     curr = builder.add_node('relu', {}, [curr])
    
# #     C_curr = init_channels
# #     num_cells_per_stage = 1
    
# #     # ------------------------------------------------------------------
# #     # 2. The Body (Stacked Cells)
# #     # ------------------------------------------------------------------
# #     for stage in range(3):
# #         # Apply Normal Cells for feature extraction
# #         for _ in range(num_cells_per_stage):
# #             curr = add_cell(builder, C_curr, C_curr, stride=1, parent_id=curr)
            
# #         # Apply Reduction Cells to downsample (except on the last stage)
# #         if stage < 2:
# #             C_next = C_curr * 2
# #             curr = add_reduction_cell(builder, C_curr, C_next, parent_id=curr)
# #             C_curr = C_next
            
# #     # ------------------------------------------------------------------
# #     # 3. The Head (Classification)
# #     # ------------------------------------------------------------------
# #     # Reduces the final 8x8 spatial dimensions down to 1x1 perfectly.
# #     curr = builder.add_node('maxpool', {'kernel_size': 8, 'stride': 8}, [curr])
# #     curr = builder.add_node('flatten', {}, [curr])
# #     out_node = builder.add_node('linear', {'in_features': C_curr, 'out_features': 10}, [curr])
    
# #     builder.g.set_output(out_node)
# #     return builder.g