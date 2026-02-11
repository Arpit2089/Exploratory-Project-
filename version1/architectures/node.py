# architectures/node.py

class Node:
    def __init__(self, node_id, op_type, params, parents):
        """
        node_id : int
        op_type : str  ('conv', 'bn', 'relu', 'add', 'concat', 'identity')
        params  : dict (channels, kernel_size, stride, etc.)
        parents : list[int]
        """
        self.id = node_id
        self.op_type = op_type
        self.params = params
        self.parents = parents
