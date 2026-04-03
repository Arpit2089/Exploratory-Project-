import random
from utils.logger import get_logger
from morphisms.exact import apply_net2deeper, apply_net2wider, apply_skip_connection
from morphisms.approximate import (
    apply_prune_filters,
    apply_remove_layer,
    apply_replace_with_sepconv,
)

logger = get_logger("operators", logfile="logs/operators.log")

# More conservative than before:
# exact ops are favored, but approximate ops still remain in the search.
OP_WEIGHTS = {
    "net2deeper": 25,
    "net2wider": 25,
    "skip": 20,
    "prune": 12,
    "remove": 10,
    "sepconv": 8,
}


def _log_new_topology(ind_id, op, new_graph):
    """Helper to safely log the topology of the mutated graph."""
    if new_graph is not None:
        try:
            topo_after = new_graph.topological_sort()
            logger.debug("Individual %s Topology AFTER %s: %s", ind_id, op, topo_after)
        except Exception as e:
            logger.error("Graph cycle or error detected AFTER %s on Individual %s: %s", op, ind_id, e)


def random_operator(individual):
    """
    Returns:
        new_graph, op_name, target_info
    """
    graph = individual.graph.clone()
    nodes = list(graph.nodes.keys())

    ops = list(OP_WEIGHTS.keys())
    weights = list(OP_WEIGHTS.values())
    op = random.choices(ops, weights=weights, k=1)[0]

    logger.info("Attempting operator: %s on Individual %s", op, individual.id)

    try:
        topo_before = graph.topological_sort()
        logger.debug("Individual %s Topology BEFORE %s: %s", individual.id, op, topo_before)
    except Exception as e:
        logger.warning("Invalid Graph BEFORE operator on Individual %s: %s", individual.id, e)

    try:
        convs = [n for n in nodes if graph.nodes[n].op_type == "conv"]

        if op == "net2deeper":
            safe_relus = []
            for n in nodes:
                if graph.nodes[n].op_type == "relu":
                    children_ops = [graph.nodes[c].op_type for c in graph.get_children(n)]
                    if not any(c_op in ("flatten", "linear", "fc") for c_op in children_ops):
                        safe_relus.append(n)

            if not safe_relus:
                raise ValueError("No safe ReLU nodes available")

            target = random.choice(safe_relus)
            new_conv_id = max(graph.nodes.keys()) + 1
            new_bn_id = new_conv_id + 1

            new_graph = apply_net2deeper(graph, target)
            _log_new_topology(individual.id, op, new_graph)

            return new_graph, op, {
                "target_node": target,
                "new_conv_id": new_conv_id,
                "new_bn_id": new_bn_id,
            }

        if op == "net2wider":
            if not convs:
                raise ValueError("No Conv nodes available")

            target = random.choice(convs)
            widen_by = 4  # conservative and stable
            new_graph = apply_net2wider(graph, target, widen_by=widen_by)
            _log_new_topology(individual.id, op, new_graph)

            return new_graph, op, {
                "target_node": target,
                "widen_by": widen_by,
            }

        if op == "skip":
            topo = graph.topological_sort()
            if len(topo) < 2:
                raise ValueError("Graph too small for skip connection")

            a_idx = random.randint(0, len(topo) - 2)
            b_idx = random.randint(a_idx + 1, len(topo) - 1)

            from_node = topo[a_idx]
            to_node = topo[b_idx]
            new_graph = apply_skip_connection(graph, from_node, to_node)
            _log_new_topology(individual.id, op, new_graph)

            return new_graph, op, {
                "from_node": from_node,
                "to_node": to_node,
            }

        if op == "prune":
            if not convs:
                raise ValueError("No Conv nodes available for pruning")

            target = random.choice(convs)
            keep_ratio = 0.80  # much safer than 0.25
            new_graph = apply_prune_filters(graph, target, keep_ratio=keep_ratio)
            _log_new_topology(individual.id, op, new_graph)

            return new_graph, op, {
                "target_node": target,
                "keep_ratio": keep_ratio,
            }

        if op == "remove":
            removable = [
                n for n in nodes
                if graph.nodes[n].op_type in ("relu", "identity")
                and n != graph.output_node
            ]
            if not removable:
                raise ValueError("No safe nodes available for removal")

            target = random.choice(removable)
            new_graph = apply_remove_layer(graph, target)
            _log_new_topology(individual.id, op, new_graph)

            return new_graph, op, {
                "target_node": target,
            }

        if op == "sepconv":
            if not convs:
                raise ValueError("No Conv nodes available for sepconv")

            target = random.choice(convs)
            new_graph = apply_replace_with_sepconv(graph, target)
            _log_new_topology(individual.id, op, new_graph)

            return new_graph, op, {
                "target_node": target,
            }

    except Exception as e:
        logger.warning("Operator %s failed: %s", op, str(e))
        return None, None, None

    return None, None, None
