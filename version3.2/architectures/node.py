# architectures/node.py
import copy

class Node:
    def __init__(self, node_id, op_type, params, parents):
        """
        node_id : int
        op_type : str  ('conv', 'bn', 'relu', 'add', 'concat', 'identity', etc.)
        params  : dict (channels, kernel_size, stride, etc.)
        parents : list[int]
        """
        self.id = node_id
        # Standardize strictly on op_type to prevent node.op vs node.op_type bugs
        self.op_type = op_type.lower() if op_type else "identity"
        
        # Deepcopy params and parents to prevent reference leakage across morphisms
        self.params = copy.deepcopy(params) if params else {}
        self.parents = list(parents) if parents else []
        
    def __repr__(self):
        return f"Node(id={self.id}, op_type={self.op_type}, parents={self.parents})"