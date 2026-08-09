"""
Unit tests for data-leakage prevention.

Covers both paths:
- classical selectors fit on X_train only;
- the metaheuristic search optimises on an inner validation split, never the
  held-out test set (``build_oracles`` + ``run_feature_selection``).
"""

from __future__ import annotations

import numpy as np

from metaheuristic_xai.classifiers import RandomForestAdapter
from metaheuristic_xai.datasets import load_dataset
from metaheuristic_xai.oracle import build_oracles
from metaheuristic_xai.runner import run_feature_selection
from metaheuristic_xai.selectors.classical import SHAPSelector


def _row_set(arr: np.ndarray) -> set[bytes]:
    a = np.ascontiguousarray(arr)
    return {a[i].tobytes() for i in range(a.shape[0])}


def test_feature_selector_fit_leakage() -> None:
    bundle = load_dataset("wdbc", random_state=42)
    selector = SHAPSelector(top_k=5, random_state=42)
    selector.fit(bundle.X_train, bundle.y_train)

    support = selector.get_support()
    assert support.ndim == 1
    assert support.size == bundle.X_train.shape[1]
    assert np.sum(support) == 5

    X_test_sel = selector.transform(bundle.X_test)
    assert X_test_sel.shape == (bundle.X_test.shape[0], 5)


def test_build_oracles_search_never_sees_test() -> None:
    bundle = load_dataset("wdbc", random_state=42)
    search_oracle, test_oracle = build_oracles(
        classifier=RandomForestAdapter(n_estimators=10, random_state=42),
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_test=bundle.X_test,
        y_test=bundle.y_test,
        dataset_name="wdbc",
        split_id=42,
        val_size=0.25,
    )

    n_train = bundle.X_train.shape[0]
    # inner split partitions the training set, nothing else
    assert search_oracle.X_train.shape[0] + search_oracle.X_test.shape[0] == n_train
    assert 0 < search_oracle.X_test.shape[0] < n_train

    train_rows = _row_set(bundle.X_train)
    test_rows = _row_set(bundle.X_test)
    val_rows = _row_set(search_oracle.X_test)          # the search "scoring" set
    fit_rows = _row_set(search_oracle.X_train)

    assert val_rows <= train_rows
    assert fit_rows <= train_rows
    assert val_rows.isdisjoint(test_rows)              # the key guarantee
    assert fit_rows.isdisjoint(test_rows)
    assert val_rows.isdisjoint(fit_rows)

    # the test oracle is the one that holds the real held-out set
    assert _row_set(test_oracle.X_test) == test_rows


def test_run_feature_selection_reports_test_metrics() -> None:
    bundle = load_dataset("wdbc", random_state=42)
    record = run_feature_selection(
        bundle,
        classifier_name="random_forest",
        method_name="ga",
        seed=42,
        alpha=0.8,
        method_kwargs={"population_size": 6, "generations": 3},
        defaults={},
    )

    # search budget is spent on the inner validation oracle
    assert record["actual_oracle_evaluations"] >= 6
    # both validation- and test-space metrics are recorded and are real numbers
    assert 0.0 <= record["accuracy"] <= 1.0
    assert 0.0 <= record["val_accuracy"] <= 1.0
    assert record["generalization_gap"] == record["val_accuracy"] - record["accuracy"]
    # selected_indices is consistent with the reported count
    assert len(record["selected_indices"]) == record["selected_features"]
    assert record["selected_features"] >= 1


def test_run_leakage_verification_shows_inflation() -> None:
    from metaheuristic_xai.runner import run_leakage_verification

    df = run_leakage_verification(
        datasets=["wdbc"], seeds=[42, 43, 44], alpha=0.9,
        ga_kwargs={"population_size": 16, "generations": 12},
        n_estimators=40, results_path=None,
    )
    assert set(df["condition"]) == {
        "original (search on test)", "leakage-free", "(no oracle)"
    }
    orig = df[(df.method == "GA") & (df.condition == "original (search on test)")]["test_accuracy"].mean()
    free = df[(df.method == "GA") & (df.condition == "leakage-free")]["test_accuracy"].mean()
    shap = df[df.method == "Tree-SHAP"]["test_accuracy"].mean()
    # searching on the test set inflates the reported accuracy
    assert orig >= free
    # and the leakage-free GA is in SHAP's neighbourhood, not far above it
    assert abs(free - shap) < 0.10
