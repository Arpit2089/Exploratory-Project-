
# evolution/pareto.py
from utils.logger import get_logger

logger = get_logger("pareto", logfile="logs/pareto.log")

def _get_all_objectives(ind):
    """
    Helper to safely merge cheap and expensive objectives.
    """
    objs = {}
    if ind.f_cheap is not None:
        objs.update(ind.f_cheap)
    if getattr(ind, "f_exp", None) is not None:
        objs.update(ind.f_exp)
    return objs

def dominates(a, b):
    """
    True if a Pareto-dominates b.
    a, b: dicts of objectives, smaller is better.
    Safely handles missing keys by treating missing evaluations as infinitely bad.
    """
    better_or_equal = True
    strictly_better = False

    # Collect all unique objectives evaluated across both individuals
    all_keys = set(a.keys()).union(set(b.keys()))

    for k in all_keys:
        # If an individual is missing an objective (e.g., failed to train),
        # treat it as infinitely bad (float('inf')) so it gets dominated.
        val_a = a.get(k, float('inf'))
        val_b = b.get(k, float('inf'))

        if val_a > val_b:
            better_or_equal = False
            break
        if val_a < val_b:
            strictly_better = True

    return better_or_equal and strictly_better


def pareto_front(individuals):
    """
    Returns list of non-dominated individuals based on ALL available objectives.
    """
    if not individuals:
        return []

    front = []
    for i, ind_i in enumerate(individuals):
        dominated = False
        objs_i = _get_all_objectives(ind_i)
        
        for j, ind_j in enumerate(individuals):
            if i == j:
                continue
                
            objs_j = _get_all_objectives(ind_j)
            
            if dominates(objs_j, objs_i):
                dominated = True
                break
                
        if not dominated:
            front.append(ind_i)

    logger.info("Pareto front size: %d / %d", len(front), len(individuals))
    return front