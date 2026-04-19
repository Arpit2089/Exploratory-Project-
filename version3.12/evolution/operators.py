# evolution/operators.py
import random
from utils.logger import get_logger
from morphisms.exact import apply_net2deeper, apply_net2wider, apply_skip_connection
from morphisms.approximate import (
    apply_prune_filters,
    apply_remove_layer,
    apply_replace_with_sepconv,
)

logger = get_logger("operators", logfile="logs/operators.log")

# Balanced grow / shrink weights.
# Exact ops (NM)  → do not change accuracy after inheritance.
# Approx ops (ANM)→ may slightly degrade accuracy; triggers distillation.
OP_WEIGHTS = {
    "net2deeper": 20,   # NM  — grow depth
    "net2wider":  20,   # NM  — grow width
    "skip":       15,   # NM  — add skip connection
    "prune":      20,   # ANM — shrink channels (resource reduction)
    "sepconv":    20,   # ANM — replace conv with depthwise-separable
    "remove":      5,   # ANM — remove a non-essential node (light simplification)
}

ANM_OPS = {"prune", "sepconv", "remove"}


def _log_topology(ind_id, op, graph):
    if graph is None:
        return
    try:
        topo = graph.topological_sort()
        logger.debug("Individual %s after %-10s topology: %s", ind_id, op, topo)
    except Exception as e:
        logger.error("Cycle detected after %s on Individual %s: %s", op, ind_id, e)


def random_operator(individual):
    """
    Apply ONE randomly chosen operator to *individual*.

    Returns
    -------
    (new_graph, op_name, target_info)
    All three are None on failure.
    """
    graph = individual.graph.clone()
    nodes = list(graph.nodes.keys())

    op = random.choices(list(OP_WEIGHTS.keys()),
                        weights=list(OP_WEIGHTS.values()), k=1)[0]

    logger.info("Attempting op '%s' on Individual %s", op, individual.id)

    try:
        convs = [n for n in nodes if graph.nodes[n].op_type in ("conv",)]

        # ---- NM: net2deeper ----
        if op == "net2deeper":
            safe_relus = [
                n for n in nodes
                if graph.nodes[n].op_type == "relu"
                and not any(
                    graph.nodes[c].op_type in ("flatten", "linear", "fc")
                    for c in graph.get_children(n)
                )
            ]
            if not safe_relus:
                raise ValueError("No safe ReLU for net2deeper")
            target   = random.choice(safe_relus)
            new_conv_id = max(graph.nodes.keys()) + 1
            new_bn_id   = new_conv_id + 1
            new_graph = apply_net2deeper(graph, target)
            _log_topology(individual.id, op, new_graph)
            return new_graph, op, {
                "target_node": target,
                "new_conv_id": new_conv_id,
                "new_bn_id":   new_bn_id,
            }

        # ---- NM: net2wider ----
        if op == "net2wider":
            if not convs:
                raise ValueError("No Conv nodes for net2wider")
            target    = random.choice(convs)
            widen_by  = 4
            new_graph = apply_net2wider(graph, target, widen_by=widen_by)
            _log_topology(individual.id, op, new_graph)
            return new_graph, op, {
                "target_node": target,
                "widen_by":    widen_by,
            }

        # ---- NM: skip connection ----
        if op == "skip":
            topo = graph.topological_sort()
            if len(topo) < 3:
                raise ValueError("Graph too small for skip")
            a_idx = random.randint(0, len(topo) - 3)
            b_idx = random.randint(a_idx + 2, len(topo) - 1)
            new_graph = apply_skip_connection(graph, topo[a_idx], topo[b_idx])
            _log_topology(individual.id, op, new_graph)
            return new_graph, op, {
                "from_node": topo[a_idx],
                "to_node":   topo[b_idx],
            }

        # ---- ANM: prune filters ----
        if op == "prune":
            if not convs:
                raise ValueError("No Conv nodes for prune")
            target     = random.choice(convs)
            # 80 % keep ratio is safe; avoids catastrophic accuracy drop
            keep_ratio = 0.80
            new_graph  = apply_prune_filters(graph, target, keep_ratio=keep_ratio)
            _log_topology(individual.id, op, new_graph)
            return new_graph, op, {
                "target_node": target,
                "keep_ratio":  keep_ratio,
            }

        # ---- ANM: replace conv with sep-conv ----
        if op == "sepconv":
            if not convs:
                raise ValueError("No Conv nodes for sepconv")
            target    = random.choice(convs)
            new_graph = apply_replace_with_sepconv(graph, target)
            _log_topology(individual.id, op, new_graph)
            return new_graph, op, {"target_node": target}

        # ---- ANM: remove a safe node ----
        if op == "remove":
            # Target relu/bn nodes that are safe to bypass.
            # BN removal is valid: upstream tensor passes through unchanged.
            # relu removal is valid: idempotent in residual paths.
            removable = [
                n for n in nodes
                if graph.nodes[n].op_type in ("relu", "bn")
                and n != graph.output_node
                and len(graph.get_children(n)) > 0  # must not be a dead-end
            ]
            if not removable:
                raise ValueError("No safe nodes for remove")
            target    = random.choice(removable)
            new_graph = apply_remove_layer(graph, target)
            _log_topology(individual.id, op, new_graph)
            return new_graph, op, {"target_node": target}

    except Exception as e:
        logger.warning("Operator '%s' failed on %s: %s", op, individual.id, e)
        return None, None, None

    return None, None, None


def is_approx_op(op_name: str) -> bool:
    """True for approximate network morphisms (ANM) operators."""
    return op_name in ANM_OPS