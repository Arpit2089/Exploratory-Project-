# evolution/operators.py
import random
from utils.logger import get_logger
from morphisms.exact import apply_net2deeper, apply_net2wider, apply_skip_connection
from morphisms.approximate import (
    apply_prune_filters,
    apply_remove_layer,
    apply_replace_with_sepconv
)

logger = get_logger("operators", logfile="logs/operators.log")


def random_operator(individual):
    graph = individual.graph
    nodes = list(graph.nodes.keys())

    ops = ["net2deeper", "net2wider", "skip", "prune", "remove", "sepconv"]
    op = random.choice(ops)

    logger.info("Applying operator: %s", op)

    try:
        convs = [n for n in nodes if graph.nodes[n].op_type == 'conv']

        if op == "net2deeper":
            relus = [n for n in nodes if graph.nodes[n].op_type == 'relu']
            if not relus:
                raise ValueError("No ReLU nodes available")
            return apply_net2deeper(graph, random.choice(relus))

        if op == "net2wider":
            if not convs:
                raise ValueError("No Conv nodes available")
            return apply_net2wider(graph, random.choice(convs), widen_by=4)

        if op == "skip":
            topo = graph.topological_sort()
            a_idx = random.randint(0, len(topo) - 2)
            b_idx = random.randint(a_idx + 1, len(topo) - 1)
            return apply_skip_connection(graph, topo[a_idx], topo[b_idx])

        if op == "prune":
            if not convs:
                raise ValueError("No Conv nodes available for pruning")
            return apply_prune_filters(graph, random.choice(convs), keep_ratio=0.5)

        if op == "remove":
            removable = [
                n for n in nodes
                if graph.nodes[n].op_type in ('relu', 'identity', 'bn')
                and n != graph.output_node
            ]
            if not removable:
                raise ValueError("No safe nodes available")
            return apply_remove_layer(graph, random.choice(removable))

        if op == "sepconv":
            if not convs:
                raise ValueError("No Conv nodes available for sepconv")
            return apply_replace_with_sepconv(graph, random.choice(convs))

    except Exception as e:
        logger.warning("Operator %s failed: %s", op, str(e))
        return None


# def random_operator(individual):
#     """
#     Randomly choose an operator and apply to individual's graph.
#     Returns new graph.
#     """
#     graph = individual.graph
#     nodes = list(graph.nodes.keys())

#     op = random.choice([
#         "net2deeper",
#         "net2wider",
#         "skip",
#         "prune",
#         "remove",
#         "sepconv"
#     ])

#     logger.info("Applying operator: %s", op)

#     try:
#         if op == "net2wider":
#             convs = [n for n in nodes if graph.nodes[n].op_type == 'conv']
#             if not convs:
#                 raise ValueError("No Conv nodes available")
#             # choose conv that actually has a downstream BN or conv expecting its channels
#             # (optional) just pick random as before
#             return apply_net2wider(graph, random.choice(convs), widen_by=4)

#         # if op == "net2wider":
#         #     convs = [n for n in nodes if graph.nodes[n].op_type == 'conv']
#         #     if not convs:
#         #         raise ValueError("No Conv nodes available")
#         #     return apply_net2wider(graph, random.choice(convs), widen_by=4)

#         if op == "skip":
#             topo = graph.topological_sort()
#             a_idx = random.randint(0, len(topo) - 2)
#             b_idx = random.randint(a_idx + 1, len(topo) - 1)

#             a = topo[a_idx]
#             b = topo[b_idx]
#             # a, b = random.sample(nodes, 2)
#             return apply_skip_connection(graph, from_node=a, to_node=b)

#         if op == "prune":
#             convs = [n for n in nodes if graph.nodes[n].op_type == 'conv']
#             if not convs:
#                 raise ValueError("No Conv nodes available for pruning")
#             return apply_prune_filters(graph, random.choice(convs), keep_ratio=0.5)

#         # if op == "remove":
#         # mids = [n for n in nodes if n != graph.output_node]
#         # return apply_remove_layer(graph, random.choice(mids))

#         if op == "remove":
#             # Only remove channel-agnostic nodes
#             removable = [
#                 n for n in nodes
#                 if graph.nodes[n].op_type in ('relu', 'identity')
#                 and n != graph.output_node
#             ]

#             if not removable:
#                 raise ValueError("No safe nodes available for removal")

#             return apply_remove_layer(graph, random.choice(removable))


#         if op == "sepconv":
#             convs = [n for n in nodes if graph.nodes[n].op_type == 'conv']
#             if not convs:
#                 raise ValueError("No Conv nodes available for sepconv")
#             return apply_replace_with_sepconv(graph, random.choice(convs))

    

#     except Exception as e:
#         logger.warning("Operator %s failed: %s", op, str(e))
#         return None
