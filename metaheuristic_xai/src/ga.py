"""
Genetic algorithm (DEAP) for binary feature selection with elitism.
"""

from __future__ import annotations

import copy
import logging
import random
import time
from typing import List, Union

import numpy as np
from deap import algorithms, base, creator, tools
from numpy.typing import NDArray

from src.fitness import evaluate_mask

logger = logging.getLogger(__name__)


def _ensure_deap_primitives() -> None:
    """Create DEAP classes once; safe across repeated invocations in one process."""
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)


def run_genetic_algorithm(
    X_train: NDArray,
    X_test: NDArray,
    y_train: NDArray,
    y_test: NDArray,
    population_size: int = 50,
    generations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    mutation_indpb: float = 0.05,
    tournament_size: int = 3,
    base_seed: int = 0,
    acc_weight: float | None = None,
) -> dict[str, Union[NDArray[np.signedinteger], List[float], float, int, None]]:
    """
    Minimize the project fitness on binary masks with:

    * Binary chromosome
    * Tournament selection
    * Two-point crossover, bit-flip mutation
    * **Elitism:** the best individual of the current generation is carried into
      the next via ``selBest(offspring + [elite], pop_size)`` so it cannot be lost.

    Parameters
    ----------
    X_train, X_test, y_train, y_test
        Pre-scaled data for ``evaluate_mask`` (indirectly, via a closure).
    population_size, generations
        As specified (default 50 and 100).
    crossover_rate, mutation_rate, mutation_indpb
        Passed to :func:`deap.algorithms.varAnd` and ``mutFlipBit`` bit-flip rate.
    tournament_size
        Tournament size for :func:`deap.tools.selTournament` on the *offspring* pool
        in the elitist replacement step, after ``varAnd`` (elitist merge uses
        ``selBest``).
    base_seed
        Combined with internal RNG: ``python.random`` and :func:`np.random` are
        seeded to ``42 + base_seed`` for an independent run.

    Returns
    -------
    dict
        ``"best_mask"``, ``"best_fitness"`` (min found), ``"convergence"`` (best
        per-generation), ``"runtime"`` (seconds, wall clock).
    """
    _ensure_deap_primitives()
    n_features = int(X_train.shape[1])
    s = 42 + int(base_seed)
    random.seed(s)
    np.random.seed(s)

    def eval_fitness(
        ind: "creator.Individual",
    ) -> tuple[float, float, int]:
        """Return tuple for DEAP: only fitness scalar in ind.fitness.values."""
        f, _acc, _n, _ = evaluate_mask(
            ind, X_train, X_test, y_train, y_test, acc_weight=acc_weight
        )
        return (f,)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n=n_features,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=mutation_indpb)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    t0 = time.perf_counter()
    pop: List = toolbox.population(n=population_size)

    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    conv: list[float] = []

    try:
        for _ in range(generations):
            elite = copy.deepcopy(tools.selBest(pop, 1)[0])
            parents = toolbox.select(pop, len(pop))
            offspring = algorithms.varAnd(
                parents, toolbox, cxpb=crossover_rate, mutpb=mutation_rate
            )
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)
            combined: List = offspring + [elite]
            pop = tools.selBest(combined, len(pop))
            best_now = float(tools.selBest(pop, 1)[0].fitness.values[0])
            conv.append(best_now)
    except (ValueError, TypeError) as e:
        logger.error("DEAP GA run failed: %s", e)
        raise

    t1 = time.perf_counter()
    best = tools.selBest(pop, 1)[0]
    mask = np.array(best, dtype=np.int8)

    return {
        "best_mask": mask,
        "best_fitness": float(best.fitness.values[0]),
        "convergence": conv,
        "runtime": float(t1 - t0),
    }
