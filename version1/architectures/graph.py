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

    def topological_sort(self):
        """
        Kahn's algorithm for topological sorting.
        Raises AssertionError if a cycle exists.
        """
        indegree = defaultdict(int)
        children = defaultdict(list)

        # Build graph
        for node_id, node in self.nodes.items():
            for p in node.parents:
                children[p].append(node_id)
                indegree[node_id] += 1

        # Nodes with no incoming edges
        queue = deque([nid for nid in self.nodes if indegree[nid] == 0])

        order = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in children[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    queue.append(v)

        assert len(order) == len(self.nodes), "Graph has a cycle!"
        return order

    def assert_acyclic(self):
        """
        Explicit validation hook for morphisms / compiler.
        """
        try:
            self.topological_sort()
        except AssertionError:
            raise RuntimeError("Invalid architecture: cycle detected")

    def clone(self):
        """
        Return a deep copy of the ArchitectureGraph.
        """
        return copy.deepcopy(self)

    def __repr__(self):
        lines = ["ArchitectureGraph:"]
        for nid in sorted(self.nodes):
            n = self.nodes[nid]
            lines.append(
                f"  Node {nid}: op={n.op_type}, parents={n.parents}"
            )
        lines.append(f"  Output node: {self.output_node}")
        return "\n".join(lines)
