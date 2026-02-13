# evolution/lemonade.py
from evolution.individual import Individual
from evolution.pareto import pareto_front
from evolution.sampling import KDESampler
from evolution.operators import random_operator
from utils.logger import get_logger

logger = get_logger("lemonade", logfile="logs/lemonade.log")

def run_lemonade(
    init_graphs,
    generations=10,
    n_children=8,
    n_accept=4
):
    """
    Main LEMONADE loop (cheap-objective version).
    """
    # Initialize population
    population = [Individual(g) for g in init_graphs]
    for ind in population:
        ind.evaluate_cheap()

    population = pareto_front(population)

    sampler = KDESampler()

    for gen in range(generations):
        logger.info("===== Generation %d =====", gen)

        # Fit KDE on current Pareto front
        sampler.fit(population)

        # Generate children
        children = []
        parents = sampler.sample(population, n_children)
        for p in parents:
            new_graph = random_operator(p)
            if new_graph is None:
                continue
            child = Individual(new_graph)
            child.evaluate_cheap()
            children.append(child)

        # Accept children via KDE
        sampler.fit(children)
        accepted = sampler.sample(children, min(n_accept, len(children)))

        # Merge and keep Pareto front
        population = pareto_front(population + accepted)

    logger.info("LEMONADE finished. Final population size=%d", len(population))
    return population
