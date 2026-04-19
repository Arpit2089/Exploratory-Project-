# evolution/operators.py
"""
Morphism operators for LEMONADE + ENAS evolution.

Changes vs original:
  1. Each operator is factored into a private _op_* function.
     This removes code duplication between random_operator and
     the new apply_specific_operator.

  2. apply_specific_operator(individual, op_name) — called by
     ENASController to execute a controller-selected operator.

  3. random_operator behaviour is UNCHANGED — same weighted sampling
     from OP_WEIGHTS, same operator logic.
"""

import random
from utils.logger import get_logger
from morphisms.exact import apply_net2deeper, apply_net2wider, apply_skip_connection
from morphisms.approximate import (
    apply_prune_filters,
    apply_remove_layer,
    apply_replace_with_sepconv,
)

logger = get_logger("operators", logfile="logs/operators.log")

OP_WEIGHTS = {
    "net2deeper": 20,
    "net2wider":  20,
    "skip":       15,
    "prune":      20,
    "sepconv":    20,
    "remove":      5,
}

ANM_OPS = {"prune", "sepconv", "remove"}


# =============================================================================
# Private per-operator implementations
# =============================================================================

def _op_net2deeper(graph, nodes, ind_id):
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
    target      = random.choice(safe_relus)
    new_conv_id = max(graph.nodes.keys()) + 1
    new_bn_id   = new_conv_id + 1
    new_graph   = apply_net2deeper(graph, target)
    return new_graph, "net2deeper", {
        "target_node": target,
        "new_conv_id": new_conv_id,
        "new_bn_id":   new_bn_id,
    }


def _op_net2wider(graph, nodes, ind_id):
    convs = [n for n in nodes if graph.nodes[n].op_type == "conv"]
    if not convs:
        raise ValueError("No Conv nodes for net2wider")
    target    = random.choice(convs)
    widen_by  = 4
    new_graph = apply_net2wider(graph, target, widen_by=widen_by)
    return new_graph, "net2wider", {
        "target_node": target,
        "widen_by":    widen_by,
    }


def _op_skip(graph, nodes, ind_id):
    topo = graph.topological_sort()
    if len(topo) < 3:
        raise ValueError("Graph too small for skip")
    a_idx     = random.randint(0, len(topo) - 3)
    b_idx     = random.randint(a_idx + 2, len(topo) - 1)
    new_graph = apply_skip_connection(graph, topo[a_idx], topo[b_idx])
    return new_graph, "skip", {
        "from_node": topo[a_idx],
        "to_node":   topo[b_idx],
    }


def _op_prune(graph, nodes, ind_id):
    convs = [n for n in nodes if graph.nodes[n].op_type == "conv"]
    if not convs:
        raise ValueError("No Conv nodes for prune")
    target     = random.choice(convs)
    keep_ratio = 0.80
    new_graph  = apply_prune_filters(graph, target, keep_ratio=keep_ratio)
    return new_graph, "prune", {
        "target_node": target,
        "keep_ratio":  keep_ratio,
    }


def _op_sepconv(graph, nodes, ind_id):
    convs = [n for n in nodes if graph.nodes[n].op_type == "conv"]
    if not convs:
        raise ValueError("No Conv nodes for sepconv")
    target    = random.choice(convs)
    new_graph = apply_replace_with_sepconv(graph, target)
    return new_graph, "sepconv", {"target_node": target}


def _op_remove(graph, nodes, ind_id):
    removable = [
        n for n in nodes
        if graph.nodes[n].op_type in ("relu", "bn")
        and n != graph.output_node
        and len(graph.get_children(n)) > 0
    ]
    if not removable:
        raise ValueError("No safe nodes for remove")
    target    = random.choice(removable)
    new_graph = apply_remove_layer(graph, target)
    return new_graph, "remove", {"target_node": target}


# Registry: op_name → implementation function
_OP_REGISTRY = {
    "net2deeper": _op_net2deeper,
    "net2wider":  _op_net2wider,
    "skip":       _op_skip,
    "prune":      _op_prune,
    "sepconv":    _op_sepconv,
    "remove":     _op_remove,
}


# =============================================================================
# Internal helper
# =============================================================================

def _execute_operator(op_name, graph, nodes, ind_id):
    """Execute named operator on already-cloned graph. Shared by both public functions."""
    fn = _OP_REGISTRY.get(op_name)
    if fn is None:
        return None, None, None
    try:
        new_graph, returned_op, target_info = fn(graph, nodes, ind_id)
        try:
            topo = new_graph.topological_sort()
            logger.debug("Ind %s after %-10s topology: %s", ind_id, returned_op, topo)
        except Exception as e:
            logger.error("Cycle after %s on %s: %s", returned_op, ind_id, e)
        return new_graph, returned_op, target_info
    except Exception as e:
        logger.warning("Operator '%s' failed on %s: %s", op_name, ind_id, e)
        return None, None, None


# =============================================================================
# Public API
# =============================================================================

def random_operator(individual):
    """
    Apply ONE randomly sampled operator to *individual*.
    Operator is sampled from OP_WEIGHTS.

    Returns (new_graph, op_name, target_info) or (None, None, None).
    """
    graph  = individual.graph.clone()
    nodes  = list(graph.nodes.keys())
    op     = random.choices(list(OP_WEIGHTS.keys()),
                            weights=list(OP_WEIGHTS.values()), k=1)[0]
    logger.info("Attempting random op '%s' on Individual %s", op, individual.id)
    return _execute_operator(op, graph, nodes, individual.id)


def apply_specific_operator(individual, op_name: str):
    """
    Apply a SPECIFIC named operator chosen by the ENASController.

    Parameters
    ----------
    individual : Individual
    op_name    : one of "net2deeper","net2wider","skip","prune","sepconv","remove"

    Returns (new_graph, op_name, target_info) or (None, None, None).
    Falls back to random_operator if op_name is unknown.
    """
    if op_name not in _OP_REGISTRY:
        logger.warning("Unknown op '%s' — falling back to random", op_name)
        return random_operator(individual)

    graph = individual.graph.clone()
    nodes = list(graph.nodes.keys())
    logger.info("Controller-selected op '%s' on Individual %s", op_name, individual.id)
    return _execute_operator(op_name, graph, nodes, individual.id)


def is_approx_op(op_name: str) -> bool:
    """True for approximate network morphisms (ANM) operators."""
    return op_name in ANM_OPS