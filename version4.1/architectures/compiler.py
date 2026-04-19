# architectures/compiler.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import get_logger

logger = get_logger("compiler", logfile="logs/compiler.log")

class CompiledModel(nn.Module):
    def __init__(self, graph, input_shape=(1, 3, 32, 32)):
        super().__init__()
        self.graph = graph
        self.input_shape = input_shape
        self.layers = nn.ModuleDict()
        self.projections = nn.ModuleDict() # Stores 1x1 convs or linear layers to align mismatched nodes
        
        logger.info("Initializing CompiledModel")
        self._execution_plan = []
        self._output_node_id = None
        
        self._compile_execution_plan()
        self._build_with_shape_inference()

    def _compile_execution_plan(self):
        order = self.graph.topological_sort()
        self._execution_plan = []
        for node_id in order:
            node = self.graph.nodes[node_id]
            self._execution_plan.append((node_id, str(node_id), list(node.parents), node.op_type))
        self._output_node_id = self.graph.output_node

    def _align_tensors(self, tensors, op, layer_key, is_build_phase):
        """Helper to safely align spatial and channel dimensions for multi-parent nodes."""
        base_tensor = tensors[0]
        aligned_tensors = [base_tensor]
        
        for i, t in enumerate(tensors[1:]):
            # 1. Dimensionality Mismatch (e.g., 4D Conv connected to 2D Linear)
            if base_tensor.dim() != t.dim():
                if base_tensor.dim() == 4:
                    base_tensor = base_tensor.view(base_tensor.size(0), -1)
                    aligned_tensors[0] = base_tensor
                if t.dim() == 4:
                    t = t.view(t.size(0), -1)
            
            # 2. Spatial Mismatch (Only applies to 4D spatial tensors)
            if base_tensor.dim() == 4 and base_tensor.shape[2:] != t.shape[2:]:
                t = F.interpolate(t, size=base_tensor.shape[2:], mode='bilinear', align_corners=False)
                
            # 3. Channel Mismatch (Only strictly required for 'add', concat natively handles different channels)
            if op == 'add' and base_tensor.size(1) != t.size(1):
                proj_key = f"{layer_key}_proj_{i}"
                if is_build_phase:
                    if base_tensor.dim() == 4:
                        self.projections[proj_key] = nn.Conv2d(t.size(1), base_tensor.size(1), kernel_size=1, bias=False)
                    else:
                        self.projections[proj_key] = nn.Linear(t.size(1), base_tensor.size(1), bias=False)
                t = self.projections[proj_key](t)
                
            aligned_tensors.append(t)
            
        return aligned_tensors

    def _build_with_shape_inference(self):
        device = torch.device('cpu') 
        cache = {'input': torch.zeros(self.input_shape, device=device)}
        
        for node_id, layer_key, parents, op in self._execution_plan:
            node = self.graph.nodes[node_id]
            
            if not parents:
                inp = cache['input']
            elif len(parents) == 1:
                inp = cache[parents[0]]
            else:
                tensors = [cache[p] for p in parents]
                aligned_tensors = self._align_tensors(tensors, op, layer_key, is_build_phase=True)
                
                if op == 'add':
                    inp = aligned_tensors[0]
                    for t in aligned_tensors[1:]:
                        inp = inp + t
                else: # concat or identity defaults to cat
                    inp = torch.cat(aligned_tensors, dim=1)
            
            out = inp
            
            if op == 'conv':
                actual_in = inp.size(1)
                self.layers[layer_key] = nn.Conv2d(
                    actual_in,
                    node.params['out_channels'],
                    kernel_size=node.params.get('kernel_size', 3),
                    stride=node.params.get('stride', 1),
                    padding=node.params.get('padding', 1),
                    bias=node.params.get('bias', False)
                )
                out = self.layers[layer_key](inp)
                
            elif op in ('sep_conv', 'separableconv2d'):
                actual_in = inp.size(1)
                out_c = node.params['out_channels']
                k = node.params.get('kernel_size', 3)
                self.layers[layer_key] = nn.Sequential(
                    nn.Conv2d(actual_in, actual_in, kernel_size=k, padding=k//2, groups=actual_in, bias=False),
                    nn.Conv2d(actual_in, out_c, kernel_size=1, bias=False)
                )
                out = self.layers[layer_key](inp)

            elif op == 'bn':
                features = inp.size(1)
                if inp.dim() == 2:
                    self.layers[layer_key] = nn.BatchNorm1d(features)
                else:
                    self.layers[layer_key] = nn.BatchNorm2d(features)
                out = self.layers[layer_key](inp)

            elif op == 'relu':
                self.layers[layer_key] = nn.ReLU(inplace=False) 
                out = self.layers[layer_key](inp)

            elif op in ('maxpool', 'max_pool'):
                k = node.params.get('kernel_size', 2)
                s = node.params.get('stride', k)
                self.layers[layer_key] = nn.MaxPool2d(kernel_size=k, stride=s)
                out = self.layers[layer_key](inp)

            elif op == 'flatten':
                self.layers[layer_key] = nn.Flatten(start_dim=1)
                out = self.layers[layer_key](inp)

            elif op in ('fc', 'linear'):
                if inp.dim() > 2:
                    inp = inp.view(inp.size(0), -1)
                actual_in = inp.size(1)
                self.layers[layer_key] = nn.Linear(actual_in, node.params['out_features'])
                out = self.layers[layer_key](inp)

            elif op in ('identity', 'add', 'concat'):
                self.layers[layer_key] = nn.Identity()
                out = self.layers[layer_key](inp)

            else:
                self.layers[layer_key] = nn.Identity()
                out = self.layers[layer_key](inp)

            cache[node_id] = out

    def forward(self, x):
        cache = {'input': x}

        for node_id, layer_key, parents, op in self._execution_plan:
            if not parents:
                inp = cache['input']
            elif len(parents) == 1:
                inp = cache[parents[0]]
            else:
                tensors = [cache[p] for p in parents]
                aligned_tensors = self._align_tensors(tensors, op, layer_key, is_build_phase=False)
                
                if op == 'add':
                    inp = aligned_tensors[0]
                    for t in aligned_tensors[1:]:
                        inp = inp + t
                else:
                    inp = torch.cat(aligned_tensors, dim=1)

            layer = self.layers[layer_key]
            
            if isinstance(layer, nn.Linear) and inp.dim() > 2:
                inp = inp.view(inp.size(0), -1)

            cache[node_id] = layer(inp)

        return cache[self._output_node_id]

# # architectures/compiler.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from utils.logger import get_logger

# logger = get_logger("compiler", logfile="logs/compiler.log")

# class CompiledModel(nn.Module):
#     def __init__(self, graph, input_shape=(1, 3, 32, 32)):
#         super().__init__()
#         self.graph = graph
#         self.input_shape = input_shape
#         self.layers = nn.ModuleDict()
#         self.projections = nn.ModuleDict() # Stores 1x1 convs to align mismatched 'add' nodes
        
#         logger.info("Initializing CompiledModel")
#         self._execution_plan = []
#         self._output_node_id = None
        
#         # Compile order first, then build layers via shape inference
#         self._compile_execution_plan()
#         self._build_with_shape_inference()

#     def _compile_execution_plan(self):
#         order = self.graph.topological_sort()
#         self._execution_plan = []
#         for node_id in order:
#             node = self.graph.nodes[node_id]
#             # Copy parents to ensure plan doesn't break if graph is mutated later
#             self._execution_plan.append((node_id, str(node_id), list(node.parents), node.op_type))
#         self._output_node_id = self.graph.output_node

#     def _build_with_shape_inference(self):
#         """
#         Passes a dummy tensor through the graph sequentially. 
#         Creates PyTorch modules based on EXACT tensor shapes, preventing 
#         runtime shape mismatches and ensuring the Optimizer tracks everything.
#         """
#         device = torch.device('cpu') # Build phase is safely done on CPU
#         cache = {'input': torch.zeros(self.input_shape, device=device)}
        
#         for node_id, layer_key, parents, op in self._execution_plan:
#             node = self.graph.nodes[node_id]
            
#             # 1. Gather Inputs
#             if not parents:
#                 inp = cache['input']
#             elif len(parents) == 1:
#                 inp = cache[parents[0]]
#             else:
#                 # Handle Multi-parent routing
#                 tensors = [cache[p] for p in parents]
#                 if op == 'add':
#                     # Fix: Add nodes MUST have matching shapes. Generate projections if they don't.
#                     base_tensor = tensors[0]
#                     aligned_tensors = [base_tensor]
                    
#                     for i, t in enumerate(tensors[1:]):
#                         if base_tensor.shape != t.shape:
#                             # Channel mismatch requires a 1x1 conv projection
#                             if base_tensor.size(1) != t.size(1):
#                                 proj_key = f"{layer_key}_proj_{i}"
#                                 self.projections[proj_key] = nn.Conv2d(t.size(1), base_tensor.size(1), kernel_size=1, bias=False)
#                                 t = self.projections[proj_key](t)
                            
#                             # Spatial mismatch requires interpolation
#                             if base_tensor.shape[2:] != t.shape[2:]:
#                                 t = F.interpolate(t, size=base_tensor.shape[2:], mode='bilinear', align_corners=False)
#                         aligned_tensors.append(t)
                        
#                     inp = aligned_tensors[0]
#                     for t in aligned_tensors[1:]:
#                         inp = inp + t
#                 elif op == 'concat' or op == 'identity': # Multi-parent identity defaults to concat
#                     inp = torch.cat(tensors, dim=1)
#                 else:
#                     inp = torch.cat(tensors, dim=1)
            
#             # 2. Instantiate correct layer based on EXACT `inp` shape
#             out = inp # Default passthrough
            
#             if op == 'conv':
#                 actual_in = inp.size(1)
#                 self.layers[layer_key] = nn.Conv2d(
#                     actual_in,
#                     node.params['out_channels'],
#                     kernel_size=node.params.get('kernel_size', 3),
#                     stride=node.params.get('stride', 1),
#                     padding=node.params.get('padding', 1),
#                     bias=node.params.get('bias', False)
#                 )
#                 out = self.layers[layer_key](inp)
                
#             elif op in ('sep_conv', 'separableconv2d'):
#                 actual_in = inp.size(1)
#                 out_c = node.params['out_channels']
#                 k = node.params.get('kernel_size', 3)
#                 self.layers[layer_key] = nn.Sequential(
#                     nn.Conv2d(actual_in, actual_in, kernel_size=k, padding=k//2, groups=actual_in, bias=False),
#                     nn.Conv2d(actual_in, out_c, kernel_size=1, bias=False)
#                 )
#                 out = self.layers[layer_key](inp)

#             elif op == 'bn':
#                 # Fix: Guard against 1D inputs crashing BatchNorm2d
#                 features = inp.size(1)
#                 if inp.dim() == 2:
#                     self.layers[layer_key] = nn.BatchNorm1d(features)
#                 else:
#                     self.layers[layer_key] = nn.BatchNorm2d(features)
#                 out = self.layers[layer_key](inp)

#             elif op == 'relu':
#                 # Fix: In-place modifications break gradient calculation in skip connections
#                 self.layers[layer_key] = nn.ReLU(inplace=False) 
#                 out = self.layers[layer_key](inp)

#             elif op in ('maxpool', 'max_pool'):
#                 k = node.params.get('kernel_size', 2)
#                 s = node.params.get('stride', k)
#                 self.layers[layer_key] = nn.MaxPool2d(kernel_size=k, stride=s)
#                 out = self.layers[layer_key](inp)

#             elif op == 'flatten':
#                 self.layers[layer_key] = nn.Flatten(start_dim=1)
#                 out = self.layers[layer_key](inp)

#             elif op in ('fc', 'linear'):
#                 if inp.dim() > 2:
#                     inp = inp.view(inp.size(0), -1)
#                 actual_in = inp.size(1)
#                 self.layers[layer_key] = nn.Linear(actual_in, node.params['out_features'])
#                 out = self.layers[layer_key](inp)

#             elif op in ('identity', 'add', 'concat'):
#                 self.layers[layer_key] = nn.Identity()
#                 out = self.layers[layer_key](inp)

#             else:
#                 self.layers[layer_key] = nn.Identity()
#                 out = self.layers[layer_key](inp)

#             # 3. Cache exact output tensor for next iteration
#             cache[node_id] = out

#     def forward(self, x):
#         # Forward pass is now STRICTLY execution. No mutations.
#         cache = {'input': x}

#         for node_id, layer_key, parents, op in self._execution_plan:
            
#             if not parents:
#                 inp = cache['input']
#             elif len(parents) == 1:
#                 inp = cache[parents[0]]
#             else:
#                 tensors = [cache[p] for p in parents]
#                 if op == 'add':
#                     base_tensor = tensors[0]
#                     aligned_tensors = [base_tensor]
                    
#                     for i, t in enumerate(tensors[1:]):
#                         # Apply runtime projections mapped during _build
#                         if base_tensor.shape != t.shape:
#                             if base_tensor.size(1) != t.size(1):
#                                 proj_key = f"{layer_key}_proj_{i}"
#                                 t = self.projections[proj_key](t)
#                             if base_tensor.shape[2:] != t.shape[2:]:
#                                 t = F.interpolate(t, size=base_tensor.shape[2:], mode='bilinear', align_corners=False)
#                         aligned_tensors.append(t)
                        
#                     inp = aligned_tensors[0]
#                     for t in aligned_tensors[1:]:
#                         inp = inp + t
#                 elif op == 'concat' or op == 'identity':
#                     inp = torch.cat(tensors, dim=1)
#                 else:
#                     inp = torch.cat(tensors, dim=1)

#             layer = self.layers[layer_key]
            
#             # Runtime flatten prep for Linear
#             if isinstance(layer, nn.Linear) and inp.dim() > 2:
#                 inp = inp.view(inp.size(0), -1)

#             cache[node_id] = layer(inp)

#         return cache[self._output_node_id]