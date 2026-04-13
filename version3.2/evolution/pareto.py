# evolution/pareto.py
import math
from utils.logger import get_logger

logger = get_logger("pareto", logfile="logs/pareto.log")


# =============================================================================
# Helpers
# =============================================================================

def _get_all_objectives(ind) -> dict:
    objs = {}
    if ind.f_cheap:
        objs.update(ind.f_cheap)
    if getattr(ind, "f_exp", None):
        objs.update(ind.f_exp)
    return objs


def _obj_keys(individuals) -> list:
    keys = set()
    for ind in individuals:
        keys.update(_get_all_objectives(ind).keys())
    return sorted(keys)


# =============================================================================
# Dominance
# =============================================================================

def dominates(objs_a: dict, objs_b: dict) -> bool:
    """
    True iff a Pareto-dominates b:
      - a is no worse than b on ALL objectives
      - a is strictly better on AT LEAST ONE objective
    Missing values treated as +∞ (worst possible).
    """
    all_keys = set(objs_a) | set(objs_b)
    better_or_equal = True
    strictly_better = False

    for k in all_keys:
        va = objs_a.get(k, math.inf)
        vb = objs_b.get(k, math.inf)
        if va > vb:
            better_or_equal = False
            break
        if va < vb:
            strictly_better = True

    return better_or_equal and strictly_better


# =============================================================================
# Pareto front
# =============================================================================

def pareto_front(individuals: list) -> list:
    """Return the subset of individuals not dominated by any other."""
    if not individuals:
        return []

    front = []
    obj_cache = [_get_all_objectives(ind) for ind in individuals]

    for i, ind_i in enumerate(individuals):
        dominated = False
        for j, ind_j in enumerate(individuals):
            if i == j:
                continue
            if dominates(obj_cache[j], obj_cache[i]):
                dominated = True
                break
        if not dominated:
            front.append(ind_i)

    logger.info("Pareto front: %d / %d non-dominated", len(front), len(individuals))
    return front


# =============================================================================
# Diversity fill using crowding distance
# =============================================================================

def fill_with_diversity(front: list, pool: list, min_pop: int) -> list:
    """
    If len(front) < min_pop, add dominated individuals from *pool* chosen
    to maximise coverage of the objective space (crowding-distance style).

    Parameters
    ----------
    front   : current Pareto-optimal individuals
    pool    : ALL individuals (front + dominated)
    min_pop : target minimum population size
    """
    if len(front) >= min_pop:
        return list(front)

    front_ids = {id(ind) for ind in front}
    dominated = [ind for ind in pool if id(ind) not in front_ids
                 and _get_all_objectives(ind)]  # must have objectives

    if not dominated:
        return list(front)

    keys = _obj_keys(front + dominated)

    def min_dist_to_front(ind):
        """Euclidean distance in normalised objective space to nearest front member."""
        ind_objs = _get_all_objectives(ind)
        min_d = math.inf
        for f_ind in front:
            f_objs = _get_all_objectives(f_ind)
            d = sum(
                (ind_objs.get(k, math.inf) - f_objs.get(k, math.inf)) ** 2
                for k in keys
                if math.isfinite(ind_objs.get(k, math.inf))
                and math.isfinite(f_objs.get(k, math.inf))
            )
            min_d = min(min_d, d)
        return min_d if math.isfinite(min_d) else 0.0

    dominated_ranked = sorted(dominated, key=min_dist_to_front, reverse=True)

    result = list(front)
    for ind in dominated_ranked:
        if len(result) >= min_pop:
            break
        result.append(ind)

    logger.info("After diversity fill: population size=%d (target min=%d)",
                len(result), min_pop)
    return result