"""
Unit tests for NSGA-II Pareto analysis.
"""

from __future__ import annotations

import numpy as np

from metaheuristic_xai.algorithms.nsga2 import (
    ParetoSolution,
    _hypervolume_2d,
    _knee_solution,
    dominates,
    fast_non_dominated_sort,
    run_nsga2,
)
from metaheuristic_xai.classifiers import RandomForestAdapter
from metaheuristic_xai.datasets import load_dataset
from metaheuristic_xai.oracle import build_oracles
from metaheuristic_xai.pareto import compute_pareto_metrics


def test_pareto_dominance() -> None:
    sol1 = ParetoSolution(mask=np.array([1, 0]), accuracy=0.9, error=0.1, n_features=2, feature_ratio=0.2)
    sol2 = ParetoSolution(mask=np.array([1, 1]), accuracy=0.8, error=0.2, n_features=4, feature_ratio=0.4)

    assert dominates(sol1, sol2)
    assert not dominates(sol2, sol1)


def test_fast_non_dominated_sort() -> None:
    sol1 = ParetoSolution(mask=np.array([1, 0]), accuracy=0.9, error=0.1, n_features=2, feature_ratio=0.2)
    sol2 = ParetoSolution(mask=np.array([1, 1]), accuracy=0.8, error=0.2, n_features=4, feature_ratio=0.4)
    sol3 = ParetoSolution(mask=np.array([0, 1]), accuracy=0.95, error=0.05, n_features=6, feature_ratio=0.6)

    population = [sol1, sol2, sol3]
    fronts = fast_non_dominated_sort(population)

    assert len(fronts) == 2
    # Front 0 contains sol1 and sol3 (non-dominated wrt each other)
    assert sol1 in fronts[0]
    assert sol3 in fronts[0]
    # Front 1 contains sol2 (dominated by sol1)
    assert sol2 in fronts[1]


def test_hypervolume_calculation() -> None:
    sol1 = ParetoSolution(mask=np.array([1]), accuracy=0.9, error=0.1, n_features=2, feature_ratio=0.2)
    sol2 = ParetoSolution(mask=np.array([1]), accuracy=0.8, error=0.2, n_features=1, feature_ratio=0.1)

    pareto_front = [sol1, sol2]
    metrics = compute_pareto_metrics(pareto_front)

    assert metrics["n_pareto_solutions"] == 2.0
    assert metrics["hypervolume"] > 0.0


def test_knee_is_scale_normalised() -> None:
    """error (~0-0.1) and feature_ratio (~0-1) live on different scales; the knee
    must be the balanced point, not simply the one with the fewest features."""
    sparse_bad = ParetoSolution(mask=np.array([1]), accuracy=0.70, error=0.30, n_features=1, feature_ratio=0.02)
    balanced = ParetoSolution(mask=np.array([1]), accuracy=0.92, error=0.08, n_features=10, feature_ratio=0.20)
    dense_good = ParetoSolution(mask=np.array([1]), accuracy=0.95, error=0.05, n_features=45, feature_ratio=0.90)

    knee = _knee_solution([sparse_bad, balanced, dense_good])
    assert knee is balanced                       # un-normalised distance would pick sparse_bad


def test_run_nsga2_returns_alpha_optimal_point() -> None:
    bundle = load_dataset("wdbc", random_state=42)
    search_oracle, _ = build_oracles(
        classifier=RandomForestAdapter(n_estimators=10, random_state=42),
        X_train=bundle.X_train, y_train=bundle.y_train,
        X_test=bundle.X_test, y_test=bundle.y_test,
        alpha=0.8, split_id=42,
    )
    res = run_nsga2(search_oracle, population_size=8, generations=4, base_seed=42)

    front = res["pareto_front"]
    a = search_oracle.alpha
    picked = min(front, key=lambda s: a * s.error + (1 - a) * s.feature_ratio)
    assert np.array_equal(res["best_mask"], picked.mask)      # best_mask == alpha-optimal front point
    assert res["hypervolume"] >= 0.0
    assert len(res["pareto_history"]) == len(res["pareto_history_gens"])
