# evolution/lemonade.py

from evolution.individual import Individual
from evolution.pareto import pareto_front
from evolution.sampling import KDESampler
from evolution.operators import random_operator
from utils.logger import get_logger

logger = get_logger("lemonade", logfile="logs/lemonade.log")


def run_lemonade(
    init_graphs,
    generations=5,
    n_children=6,
    n_accept=3,
    train_loader=None,
    val_loader=None,
    device="cpu"
):
    """
    FULL LEMONADE LOOP
    - Cheap objectives: params + flops
    - Expensive objective: validation error
    """

    # ------------------------------
    # Initialize population
    # ------------------------------
    population = [Individual(g) for g in init_graphs]

    for ind in population:
        ind.evaluate_cheap()

        if train_loader is not None:
            ind.evaluate_expensive(
                train_loader,
                val_loader,
                device=device,
                epochs=1
            )

    population = pareto_front(population)

    sampler = KDESampler()

    # ------------------------------
    # Evolution Loop
    # ------------------------------
    for gen in range(generations):

        logger.info("===== Generation %d =====", gen)

        sampler.fit(population)

        children = []
        parents = sampler.sample(population, n_children)

        for p in parents:

            new_graph = random_operator(p)

            if new_graph is None:
                continue

            child = Individual(new_graph)

            # Cheap evaluation
            child.evaluate_cheap()

            # Expensive evaluation (if loaders provided)
            if train_loader is not None:
                child.evaluate_expensive(
                    train_loader,
                    val_loader,
                    device=device,
                    epochs=1
                )

            children.append(child)

        if len(children) == 0:
            logger.warning("No children generated this generation.")
            continue

        # Accept best children via KDE
        sampler.fit(children)
        accepted = sampler.sample(children, min(n_accept, len(children)))

        # Merge + keep Pareto front
        population = pareto_front(population + accepted)

        logger.info("Population size after selection: %d", len(population))

    logger.info("LEMONADE finished. Final population size=%d", len(population))

    return population
