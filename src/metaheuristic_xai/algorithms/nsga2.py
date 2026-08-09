"""
True Multi-Objective NSGA-II implementation for feature selection.

Optimizes TWO INDEPENDENT OBJECTIVES:
1. Error = 1 - accuracy (minimize)
2. Feature Ratio = |S| / d (minimize)

Includes fast non-dominated sorting, crowding distance assignment, binary tournament selection,
crossover, bit-flip mutation, and (P + Q) elitist environmental selection.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from metaheuristic_xai.oracle import EvaluationOracle, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class ParetoSolution:
    """A solution in the objective space for NSGA-II."""

    mask: NDArray[np.signedinteger]
    accuracy: float
    error: float  # Obj 1: 1 - accuracy (minimized)
    n_features: int
    feature_ratio: float  # Obj 2: n_features / total_d (minimized)
    fitness: float = float("inf")
    rank: int = 0
    crowding_distance: float = 0.0
    evaluation_result: Optional[EvaluationResult] = None


def dominates(sol1: ParetoSolution, sol2: ParetoSolution) -> bool:
    """
    Return True if sol1 Pareto-dominates sol2 (minimizing both error and feature_ratio).
    """
    not_worse = (sol1.error <= sol2.error) and (sol1.feature_ratio <= sol2.feature_ratio)
    strictly_better = (sol1.error < sol2.error) or (sol1.feature_ratio < sol2.feature_ratio)
    return bool(not_worse and strictly_better)


def fast_non_dominated_sort(population: list[ParetoSolution]) -> list[list[ParetoSolution]]:
    """
    Fast non-dominated sorting returning list of fronts using integer indices.
    """
    n = len(population)
    S = [[] for _ in range(n)]
    n_dominated = [0] * n
    front_indices: list[list[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(population[i], population[j]):
                S[i].append(j)
            elif dominates(population[j], population[i]):
                n_dominated[i] += 1

        if n_dominated[i] == 0:
            population[i].rank = 1
            front_indices[0].append(i)

    i = 0
    while len(front_indices[i]) > 0:
        next_front_idx: list[int] = []
        for sol_idx in front_indices[i]:
            for j in S[sol_idx]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    population[j].rank = i + 2
                    next_front_idx.append(j)
        i += 1
        front_indices.append(next_front_idx)

    if len(front_indices[-1]) == 0:
        front_indices.pop()

    return [[population[idx] for idx in f] for f in front_indices]


def calculate_crowding_distance(front: list[ParetoSolution]) -> None:
    """Calculate crowding distance for all solutions in a single front."""
    l = len(front)
    if l == 0:
        return
    for sol in front:
        sol.crowding_distance = 0.0

    if l <= 2:
        for sol in front:
            sol.crowding_distance = float("inf")
        return

    # Obj 1: Error
    front.sort(key=lambda x: x.error)
    front[0].crowding_distance = float("inf")
    front[-1].crowding_distance = float("inf")
    err_range = front[-1].error - front[0].error
    if err_range > 1e-8:
        for i in range(1, l - 1):
            front[i].crowding_distance += (front[i + 1].error - front[i - 1].error) / err_range

    # Obj 2: Feature Ratio
    front.sort(key=lambda x: x.feature_ratio)
    front[0].crowding_distance = float("inf")
    front[-1].crowding_distance = float("inf")
    ratio_range = front[-1].feature_ratio - front[0].feature_ratio
    if ratio_range > 1e-8:
        for i in range(1, l - 1):
            front[i].crowding_distance += (
                front[i + 1].feature_ratio - front[i - 1].feature_ratio
            ) / ratio_range


def binary_tournament_selection(
    population: list[ParetoSolution], rng: np.random.Generator
) -> ParetoSolution:
    """Tournament selection based on rank (lower is better) and crowding distance (higher is better)."""
    i1, i2 = rng.choice(len(population), size=2, replace=False)
    p1, p2 = population[i1], population[i2]

    if p1.rank < p2.rank:
        return p1
    elif p2.rank < p1.rank:
        return p2
    else:
        return p1 if p1.crowding_distance >= p2.crowding_distance else p2


def crossover_and_mutate(
    parent1: ParetoSolution,
    parent2: ParetoSolution,
    crossover_rate: float,
    mutation_rate: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.signedinteger], NDArray[np.signedinteger]]:
    """Two-point crossover and bit-flip mutation."""
    d = len(parent1.mask)
    c1 = parent1.mask.copy()
    c2 = parent2.mask.copy()

    if rng.random() < crossover_rate and d > 2:
        pt1, pt2 = sorted(rng.choice(d, size=2, replace=False))
        c1[pt1:pt2], c2[pt1:pt2] = parent2.mask[pt1:pt2].copy(), parent1.mask[pt1:pt2].copy()

    # Mutation
    for i in range(d):
        if rng.random() < mutation_rate:
            c1[i] = 1 - c1[i]
        if rng.random() < mutation_rate:
            c2[i] = 1 - c2[i]

    return c1, c2


def _hypervolume_2d(
    front: list[ParetoSolution], ref: tuple[float, float] = (1.0, 1.0)
) -> float:
    """2D hypervolume dominated by ``front`` w.r.t. reference point ``ref``
    (both objectives -- error, feature_ratio -- minimised)."""
    pts = sorted({(s.error, s.feature_ratio) for s in front})
    ref_e, ref_r = ref
    hv = 0.0
    prev_r = ref_r
    for e, r in pts:
        if e < ref_e and r < prev_r:
            hv += (ref_e - e) * (prev_r - r)
            prev_r = r
    return float(max(0.0, hv))


def _knee_solution(front: list[ParetoSolution]) -> ParetoSolution:
    """Knee = point closest to the ideal corner in **min-max normalised** objective
    space (raw ``error`` and ``feature_ratio`` live on very different scales, so an
    un-normalised distance just returns the sparsest point)."""
    errs = np.array([s.error for s in front], dtype=float)
    rats = np.array([s.feature_ratio for s in front], dtype=float)

    def _norm(a: np.ndarray) -> np.ndarray:
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

    dist = _norm(errs) ** 2 + _norm(rats) ** 2
    return front[int(np.argmin(dist))]


def run_nsga2(
    oracle: EvaluationOracle,
    population_size: int = 50,
    generations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.05,
    base_seed: int = 0,
) -> dict[str, Any]:
    """
    Run true multi-objective NSGA-II.

    ``best_mask`` is the point on the final Pareto front that minimises the same
    scalar fitness the single-objective methods use (``alpha * error +
    (1 - alpha) * feature_ratio``), so it is directly comparable to GA/BPSO/SA/GWO
    at the same ``alpha``. The unweighted normalised knee and the full front /
    hypervolume are also returned for front-quality analysis.
    """
    d = oracle.total_d
    rng = np.random.default_rng(42 + int(base_seed))

    t0 = time.perf_counter()

    # Initialize population
    pop_masks = rng.integers(0, 2, size=(population_size, d), dtype=np.int8)
    population: list[ParetoSolution] = []

    for m in pop_masks:
        res = oracle.evaluate(m)
        sol = ParetoSolution(
            mask=res.mask,
            accuracy=res.accuracy,
            error=res.error,
            n_features=res.n_features,
            feature_ratio=res.feature_ratio,
            fitness=float(res.fitness),
            evaluation_result=res,
        )
        population.append(sol)

    # Initial sorting
    fronts = fast_non_dominated_sort(population)
    for f in fronts:
        calculate_crowding_distance(f)

    # Snapshot the front at ~5 evenly spaced generations plus the last one.
    snap_gens = sorted(
        {int(round(x)) for x in np.linspace(1, generations, min(5, generations))} | {generations}
    )
    pareto_history: list[list[ParetoSolution]] = []
    pareto_history.append([sol for sol in fronts[0]])
    history_gens: list[int] = [0]

    for gen in range(1, generations + 1):
        offspring: list[ParetoSolution] = []

        while len(offspring) < population_size:
            p1 = binary_tournament_selection(population, rng)
            p2 = binary_tournament_selection(population, rng)
            m1, m2 = crossover_and_mutate(p1, p2, crossover_rate, mutation_rate, rng)

            for m in (m1, m2):
                if len(offspring) < population_size:
                    res = oracle.evaluate(m)
                    sol = ParetoSolution(
                        mask=res.mask,
                        accuracy=res.accuracy,
                        error=res.error,
                        n_features=res.n_features,
                        feature_ratio=res.feature_ratio,
                        fitness=float(res.fitness),
                        evaluation_result=res,
                    )
                    offspring.append(sol)

        # Environmental selection (P + Q)
        combined = population + offspring
        all_fronts = fast_non_dominated_sort(combined)

        new_pop: list[ParetoSolution] = []
        for f in all_fronts:
            calculate_crowding_distance(f)
            if len(new_pop) + len(f) <= population_size:
                new_pop.extend(f)
            else:
                f.sort(key=lambda x: x.crowding_distance, reverse=True)
                needed = population_size - len(new_pop)
                new_pop.extend(f[:needed])
                break

        population = new_pop
        current_fronts = fast_non_dominated_sort(population)
        for f in current_fronts:
            calculate_crowding_distance(f)

        if gen in snap_gens:
            pareto_history.append([sol for sol in current_fronts[0]])
            history_gens.append(gen)

    t1 = time.perf_counter()

    final_pareto_front = current_fronts[0]
    alpha = oracle.alpha
    best_scalar = min(
        final_pareto_front,
        key=lambda s: alpha * s.error + (1.0 - alpha) * s.feature_ratio,
    )
    best_knee = _knee_solution(final_pareto_front)

    return {
        "pareto_front": final_pareto_front,
        "best_scalar_solution": best_scalar,
        "best_knee_solution": best_knee,
        "best_mask": best_scalar.mask,
        "best_fitness": alpha * best_scalar.error + (1.0 - alpha) * best_scalar.feature_ratio,
        "hypervolume": _hypervolume_2d(final_pareto_front),
        "pareto_history": pareto_history,
        "pareto_history_gens": history_gens,
        "runtime": float(t1 - t0),
        "n_evaluations": oracle.total_evaluations,
    }
