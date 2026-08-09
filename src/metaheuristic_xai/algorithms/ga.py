"""
Genetic algorithm (DEAP) for binary feature selection with elitism.
"""

from __future__ import annotations

import copy
import logging
import random
import time
from typing import List, Sequence, Union

import numpy as np
from deap import algorithms, base, creator, tools
from numpy.typing import ArrayLike, NDArray

from metaheuristic_xai.oracle import EvaluationOracle

logger = logging.getLogger(__name__)


def _repair_to_k(bits: list[int], k: int) -> bool:
    """Force ``bits`` to have exactly ``k`` ones (drop/add uniformly at random).
    Returns True if anything changed."""
    on = [i for i, b in enumerate(bits) if b]
    if len(on) == k:
        return False
    if len(on) > k:
        for i in random.sample(on, len(on) - k):
            bits[i] = 0
    else:
        off = [i for i, b in enumerate(bits) if not b]
        for i in random.sample(off, min(k - len(on), len(off))):
            bits[i] = 1
    return True


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
    init_density: float | None = None,
    crossover: str = "two_point",
    n_elites: int = 1,
    immigrants: int = 0,
    target_k: int | None = None,
    seed_masks: Sequence[ArrayLike] | None = None,
    *,
    oracle: EvaluationOracle,
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
    init_density
        If given, each gene of the initial population is set with probability
        ``init_density`` (Bernoulli) instead of 0.5, and every individual is
        repaired to have at least one feature. Low values (0.05-0.15) seed the
        search in the sparse region so it grows subsets rather than shrinks them.
    crossover
        ``"two_point"`` (default, contiguous segment swap) or ``"uniform"``
        (per-gene swap, ``indpb=0.5``) -- uniform makes no locality assumption
        about feature order.
    n_elites
        Number of best individuals copied unchanged into the next generation.
    immigrants
        Number of worst individuals replaced each generation by fresh random
        individuals (drawn at ``init_density`` if set) -- continuous diversity
        injection against premature convergence.
    target_k
        If given, every individual is repaired to hold exactly ``target_k``
        features (fixed-size subset GA), so the result is directly comparable to a
        top-k ranking method at the same cardinality.
    seed_masks
        Optional binary masks used to warm-start the initial population (e.g. the
        SHAP top-k subset). Each replaces one random individual and is repaired to
        ``target_k`` if set. Elitism guarantees the search never returns something
        worse than the best seed on the validation objective.

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

    n_evaluations = 0

    def eval_fitness(
        ind: "creator.Individual",
    ) -> tuple[float, float, int]:
        """Return tuple for DEAP: only fitness scalar in ind.fitness.values."""
        nonlocal n_evaluations
        res = oracle.evaluate(ind)
        n_evaluations = oracle.total_evaluations
        return (res.fitness,)

    def new_individual() -> "creator.Individual":
        if target_k is not None:
            bits = [0] * n_features
            for i in random.sample(range(n_features), min(target_k, n_features)):
                bits[i] = 1
        elif init_density is None:
            bits = [random.randint(0, 1) for _ in range(n_features)]
        else:
            bits = [1 if random.random() < init_density else 0 for _ in range(n_features)]
        if not any(bits):
            bits[random.randrange(n_features)] = 1
        return creator.Individual(bits)

    toolbox = base.Toolbox()
    toolbox.register("individual", new_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_fitness)
    if crossover == "uniform":
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
    else:
        toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=mutation_indpb)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    n_elites = max(1, int(n_elites))
    immigrants = max(0, int(immigrants))

    t0 = time.perf_counter()
    pop: List = toolbox.population(n=population_size)

    if seed_masks:
        for j, sm in enumerate(seed_masks):
            if j >= population_size:
                break
            bits = [int(x) for x in np.ravel(np.asarray(sm, dtype=np.int8))]
            if target_k is not None:
                _repair_to_k(bits, target_k)
            elif not any(bits):
                bits[random.randrange(n_features)] = 1
            pop[j] = creator.Individual(bits)

    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    conv: list[float] = []

    try:
        for _ in range(generations):
            # Defaults (n_elites=1, immigrants=0, two_point, dense init) reproduce
            # the original elitist replacement exactly.
            elites = [copy.deepcopy(e) for e in tools.selBest(pop, n_elites)]
            parents = toolbox.select(pop, population_size)
            offspring = algorithms.varAnd(
                parents, toolbox, cxpb=crossover_rate, mutpb=mutation_rate
            )
            for ind in offspring:
                changed = _repair_to_k(ind, target_k) if target_k is not None else False
                if changed or not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)

            combined: List = offspring + elites
            if immigrants:
                fresh = [new_individual() for _ in range(immigrants)]
                for ind in fresh:
                    ind.fitness.values = toolbox.evaluate(ind)
                combined = combined + fresh
            pop = tools.selBest(combined, population_size)

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
        "n_evaluations": n_evaluations,
    }
