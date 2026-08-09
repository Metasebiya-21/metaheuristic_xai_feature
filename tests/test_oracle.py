"""
Unit tests for EvaluationOracle and caching.
"""

from __future__ import annotations

import numpy as np

from metaheuristic_xai.classifiers import RandomForestAdapter
from metaheuristic_xai.datasets import load_dataset
from metaheuristic_xai.oracle import CrossValOracle, EvaluationOracle, build_oracles


def test_oracle_caching_and_edge_cases() -> None:
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

    d = bundle.X_train.shape[1]

    # Test empty mask
    empty_mask = np.zeros(d, dtype=np.int8)
    res_empty = oracle.evaluate(empty_mask)
    assert res_empty.n_features == 0
    assert res_empty.accuracy == 0.0
    assert res_empty.fitness == 1.0

    # Test full mask
    full_mask = np.ones(d, dtype=np.int8)
    res_full = oracle.evaluate(full_mask)
    assert res_full.n_features == d
    assert res_full.accuracy > 0.5
    assert not res_full.cached

    # Test cache hit
    res_full_cached = oracle.evaluate(full_mask)
    assert res_full_cached.cached
    assert oracle.cache_hits == 1
    assert oracle.total_evaluations == 3


def test_crossval_oracle_interface_and_cache() -> None:
    bundle = load_dataset("wdbc", random_state=42)
    o = CrossValOracle(
        RandomForestAdapter(n_estimators=10, random_state=42),
        bundle.X_train, bundle.y_train, n_splits=4, alpha=0.8, split_id=42,
    )
    d = bundle.X_train.shape[1]
    assert o.total_d == d and o.X_train.shape[1] == d

    empty = o.evaluate(np.zeros(d, dtype=np.int8))
    assert empty.n_features == 0 and empty.fitness == 1.0

    mask = np.zeros(d, dtype=np.int8)
    mask[[0, 5, 9, 12]] = 1
    r1 = o.evaluate(mask)
    assert r1.n_features == 4 and 0.0 < r1.accuracy <= 1.0 and not r1.cached
    r2 = o.evaluate(mask)                       # cached, same value
    assert r2.cached and r2.accuracy == r1.accuracy
    assert o.cache_hits == 1 and o.total_evaluations == 3


def test_build_oracles_cv_returns_crossval_search() -> None:
    bundle = load_dataset("wdbc", random_state=42)
    search, test = build_oracles(
        RandomForestAdapter(n_estimators=10, random_state=42),
        bundle.X_train, bundle.y_train, bundle.X_test, bundle.y_test,
        alpha=0.8, split_id=42, cv=5,
    )
    assert isinstance(search, CrossValOracle)
    assert isinstance(test, EvaluationOracle)
    # the CV search oracle only ever sees the training partition
    assert search.X.shape[0] == bundle.X_train.shape[0]
