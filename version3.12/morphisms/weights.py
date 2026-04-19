# morphisms/weights.py
"""
Master router for Lamarckian weight inheritance.

Call transfer_weights(parent_model, child_model, child_graph, op_name, target_info)
immediately after applying an operator and building both models.
"""
from morphisms.exact import (
    inherit_weights,
    inherit_weights_net2wider,
    initialize_conv_as_identity,
    initialize_bn_as_identity,
)
from morphisms.approximate import inherit_weights_prune, inherit_weights_sepconv
from utils.logger import get_logger

logger = get_logger("weights", logfile="logs/morphisms.log")


def transfer_weights(parent_model, child_model, child_graph,
                     op_name: str, target_info: dict):
    """
    Copy weights from parent_model → child_model after a morphism.

    Parameters
    ----------
    parent_model : CompiledModel — the trained parent
    child_model  : CompiledModel — the freshly built child
    child_graph  : ArchitectureGraph — the child's graph (used for topology)
    op_name      : operator name string (e.g. "net2deeper", "prune", ...)
    target_info  : dict returned by random_operator (contains target_node etc.)
    """
    # Step 1: Base inheritance — copy all matching layers by key
    try:
        inherit_weights(parent_model, child_model)
    except Exception as e:
        logger.warning("Base inherit_weights failed: %s", e)

    if not target_info:
        return

    target_node = target_info.get("target_node")

    try:
        if op_name == "net2deeper":
            conv_key = str(target_info.get("new_conv_id", ""))
            bn_key   = str(target_info.get("new_bn_id", ""))
            if conv_key in child_model.layers:
                initialize_conv_as_identity(child_model.layers[conv_key])
            if bn_key in child_model.layers:
                initialize_bn_as_identity(child_model.layers[bn_key])

        elif op_name == "net2wider" and target_node is not None:
            inherit_weights_net2wider(parent_model, child_model,
                                      target_node, widen_by=4)

        elif op_name == "prune" and target_node is not None:
            # FIX: pass child_graph (ArchitectureGraph), NOT child_model
            inherit_weights_prune(parent_model, child_model,
                                  child_graph, target_node)

        elif op_name == "sepconv" and target_node is not None:
            inherit_weights_sepconv(parent_model, child_model, target_node)

        # "skip" and "remove" — base inheritance already handled them

    except Exception as e:
        logger.warning("Op-specific weight transfer failed for %s: %s", op_name, e)