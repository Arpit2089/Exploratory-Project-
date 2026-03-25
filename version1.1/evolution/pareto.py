# evolution/pareto.py
from utils.logger import get_logger

logger = get_logger("pareto", logfile="logs/pareto.log")

def dominates(a, b):
    """
    True if a Pareto-dominates b.
    a, b: dicts with same keys (objectives), smaller is better.
    """
    better_or_equal = True
    strictly_better = False

    for k in a.keys():
        if a[k] > b[k]:
            better_or_equal = False
            break
        if a[k] < b[k]:
            strictly_better = True

    return better_or_equal and strictly_better


def pareto_front(individuals):
    """
    Returns list of non-dominated individuals.
    """
    front = []
    for i, ind_i in enumerate(individuals):
        dominated = False
        for j, ind_j in enumerate(individuals):
            if i != j and dominates(ind_j.f_cheap, ind_i.f_cheap):
                dominated = True
                break
        if not dominated:
            front.append(ind_i)

    logger.info("Pareto front size: %d / %d", len(front), len(individuals))
    return front
