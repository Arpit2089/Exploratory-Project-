# architectures/graph.py
import copy
from collections import defaultdict, deque

class ArchitectureGraph:
    def __init__(self):
        self.nodes = {}          # node_id -> Node
        self.output_node = None

    def add_node(self, node):
        assert node.id not in self.nodes, f"Duplicate node id {node.id}"
        self.nodes[node.id] = node

    def set_output(self, node_id):  
        assert node_id in self.nodes, "Output node must exist"
        self.output_node = node_id

    def get_parents(self, node_id):
        return self.nodes[node_id].parents

    def get_children(self, node_id):
        """
        Crucial for net2wider and transitive channel updates.
        Returns a list of node_ids that have the given node_id as a parent.
        """
        children = []
        for nid, node in self.nodes.items():
            if node_id in node.parents:
                children.append(nid)
        return children

    def topological_sort(self):
        indegree = {nid: 0 for nid in self.nodes}
        children = defaultdict(list)

        # Build graph explicitly ensuring all nodes are tracked
        for node_id, node in self.nodes.items():
            for p in node.parents:
                if p in self.nodes:  # Guard against implicit input nodes not in dict
                    children[p].append(node_id)
                    indegree[node_id] += 1

        queue = deque([nid for nid in self.nodes if indegree[nid] == 0])
        order = []
        
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in children[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    queue.append(v)

        # If order length != nodes length, a cycle exists (e.g., bad skip connection)
        if len(order) != len(self.nodes):
            raise AssertionError("Graph has a cycle! Check morphism operations.")
        return order

    def assert_acyclic(self):
        try:
            self.topological_sort()
        except AssertionError as e:
            raise RuntimeError(f"Invalid architecture: {str(e)}")

    def clone(self):
        """
        Deepcopy is expensive but necessary for branching evolutionary paths.
        Ensure node params aren't holding heavy tensors when cloned.
        """
        return copy.deepcopy(self)

    def __repr__(self):
        lines = ["ArchitectureGraph:"]
        for nid in self.topological_sort():
            n = self.nodes[nid]
            lines.append(f"  Node {nid}: op={n.op_type}, parents={n.parents}")
        lines.append(f"  Output node: {self.output_node}")
        return "\n".join(lines)