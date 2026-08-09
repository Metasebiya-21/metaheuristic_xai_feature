"""
Unit tests for all 10 feature selectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from metaheuristic_xai.classifiers import RandomForestAdapter
from metaheuristic_xai.datasets import load_dataset
from metaheuristic_xai.oracle import EvaluationOracle
from metaheuristic_xai.selectors import SELECTOR_REGISTRY, get_feature_selector


@pytest.mark.parametrize("s_name", list(SELECTOR_REGISTRY.keys()))
def test_feature_selectors(s_name: str) -> None:
    bundle = load_dataset("wdbc", random_state=42)
    clf = RandomForestAdapter(n_estimators=10, random_state=42)
    oracle = EvaluationOracle(
        classifier=clf,
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_test=bundle.X_test,
        y_test=bundle.y_test,
        enable_cache=True,
    )

    kwargs = {"oracle": oracle, "random_state": 42}
    if s_name in ("ga", "bpso", "gwo", "nsga2"):
        kwargs["population_size"] = 5
        kwargs["generations"] = 2
        kwargs["iterations"] = 2
        kwargs["iters"] = 2
    elif s_name == "sa":
        kwargs["max_steps"] = 10

    selector = get_feature_selector(s_name, **kwargs)
    selector.fit(bundle.X_train, bundle.y_train)

    support = selector.get_support()
    assert support.ndim == 1
    assert support.size == bundle.X_train.shape[1]
    assert np.sum(support) >= 1  # At least 1 feature selected


def test_metaheuristic_oracle_budget_counts() -> None:
    bundle = load_dataset("wdbc", random_state=42)
    clf = RandomForestAdapter(n_estimators=10, random_state=42)
    oracle = EvaluationOracle(
        classifier=clf,
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_test=bundle.X_test,
        y_test=bundle.y_test,
        enable_cache=True,
    )

    configs = {
        "ga": {"population_size": 20, "generations": 10},
        "bpso": {"swarm_size": 20, "iters": 10},
        "sa": {"initial_temp": 50.0, "cooling": 0.95, "max_steps": 100},
        "gwo": {"population_size": 20, "iterations": 10},
        "nsga2": {"population_size": 20, "generations": 10},
    }

    nominal_budget = {
        "ga": lambda cfg: cfg["population_size"] * cfg["generations"],
        "bpso": lambda cfg: cfg["swarm_size"] * cfg["iters"],
        "sa": lambda cfg: cfg["max_steps"],
        "gwo": lambda cfg: cfg["population_size"] * cfg["iterations"],
        "nsga2": lambda cfg: cfg["population_size"] * cfg["generations"],
    }

    for method_name, method_kwargs in configs.items():
        oracle = EvaluationOracle(
            classifier=clf,
            X_train=bundle.X_train,
            y_train=bundle.y_train,
            X_test=bundle.X_test,
            y_test=bundle.y_test,
            enable_cache=True,
        )
        selector = get_feature_selector(method_name, oracle=oracle, random_state=42, **method_kwargs)
        selector.fit(bundle.X_train, bundle.y_train)

        assert selector.n_evaluations_ == oracle.total_evaluations
        nominal = int(nominal_budget[method_name](method_kwargs))
        assert oracle.total_evaluations >= int(0.7 * nominal)


def test_sa_honours_step_budget() -> None:
    """SA must spend its full ``max_steps`` budget, not terminate early on the
    temperature schedule (regression test for the old ``T <= min_temp`` stop)."""
    bundle = load_dataset("wdbc", random_state=42)
    clf = RandomForestAdapter(n_estimators=10, random_state=42)

    for max_steps in (60, 300):
        oracle = EvaluationOracle(
            classifier=clf, X_train=bundle.X_train, y_train=bundle.y_train,
            X_test=bundle.X_test, y_test=bundle.y_test, enable_cache=True,
        )
        sel = get_feature_selector(
            "sa", oracle=oracle, random_state=42,
            initial_temp=100.0, cooling=0.9, max_steps=max_steps,
        )
        sel.fit(bundle.X_train, bundle.y_train)
        # 1 initial + max_steps candidates + 1 final re-eval
        assert oracle.total_evaluations == max_steps + 2


def test_ga_operator_knobs() -> None:
    """The new GA operators run and change behaviour; defaults stay deterministic."""
    from metaheuristic_xai.algorithms.ga import run_genetic_algorithm

    bundle = load_dataset("wdbc", random_state=42)
    d = bundle.X_train.shape[1]

    def run(**kw):
        oracle = EvaluationOracle(
            classifier=RandomForestAdapter(n_estimators=10, random_state=42),
            X_train=bundle.X_train, y_train=bundle.y_train,
            X_test=bundle.X_test, y_test=bundle.y_test, enable_cache=True,
        )
        g = run_genetic_algorithm(
            oracle.X_train, oracle.X_test, oracle.y_train, oracle.y_test,
            population_size=12, generations=6, base_seed=1, oracle=oracle, **kw,
        )
        return np.asarray(g["best_mask"], dtype=np.int8)

    base_a, base_b = run(), run()
    assert np.array_equal(base_a, base_b)                      # default path deterministic
    assert base_a.size == d and base_a.sum() >= 1

    sparse = run(init_density=0.1)
    assert sparse.sum() >= 1
    # sparse init should not blow up and typically yields a smaller subset
    assert sparse.sum() <= base_a.sum() + d // 4

    uni = run(crossover="uniform")
    assert uni.size == d and uni.sum() >= 1

    imm = run(immigrants=4, n_elites=2)
    assert imm.size == d and imm.sum() >= 1
