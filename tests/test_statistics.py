"""
Unit tests for non-parametric statistical tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metaheuristic_xai.evaluation import (
    feature_stability,
    friedman_nemenyi,
    friedman_test,
    holm_wilcoxon_vs,
    method_summary,
    wilcoxon_paired_test,
)


def _toy_runs() -> pd.DataFrame:
    """3 methods x 2 datasets x 2 classifiers x 3 seeds; 'good' clearly best."""
    rng = np.random.default_rng(0)
    rows = []
    base = {"good": 0.92, "mid": 0.85, "bad": 0.78}
    for ds in ("a", "b"):
        for clf in ("rf", "xgb"):
            for seed in (42, 43, 44):
                for m, mu in base.items():
                    acc = float(np.clip(mu + rng.normal(0, 0.01), 0, 1))
                    rows.append({
                        "dataset": ds, "classifier": clf, "method": m, "seed": seed,
                        "method_family": "x", "scale_tier": "low",
                        "accuracy": acc, "val_accuracy": acc + 0.02,
                        "generalization_gap": 0.02,
                        "selected_features": {"good": 5, "mid": 10, "bad": 20}[m],
                        "feature_ratio": 0.2, "search_features": 25,
                        "fitness": 1 - acc, "runtime_seconds": 1.0,
                        "actual_oracle_evaluations": 100,
                        "selected_indices": list(range({"good": 5, "mid": 10, "bad": 20}[m])),
                    })
    return pd.DataFrame(rows)


def test_wilcoxon_paired_test() -> None:
    acc1 = np.array([0.90, 0.92, 0.91, 0.89, 0.93])
    acc2 = np.array([0.85, 0.86, 0.87, 0.84, 0.88])

    res = wilcoxon_paired_test(acc1, acc2, name_method="GA", name_baseline="SHAP")
    assert res is not None
    assert res.pvalue < 0.1
    assert res.effect_size > 0.0


def test_friedman_test() -> None:
    # 5 paired subjects, 3 methods
    mat = np.array(
        [
            [0.90, 0.85, 0.80],
            [0.92, 0.86, 0.81],
            [0.91, 0.87, 0.82],
            [0.89, 0.84, 0.79],
            [0.93, 0.88, 0.83],
        ]
    )
    methods = ["GA", "SHAP", "SA"]

    f_res, posthoc_df = friedman_test(mat, methods)
    assert f_res is not None
    assert f_res.pvalue < 0.05
    assert posthoc_df is not None
    assert len(posthoc_df) == 3


def test_friedman_nemenyi() -> None:
    df = _toy_runs()
    r = friedman_nemenyi(df, "accuracy")
    assert r.n_methods == 3 and r.n_subjects == 12
    assert r.pvalue < 0.01
    assert r.mean_ranks.index[0] == "good"          # best test accuracy -> lowest rank
    assert r.cd > 0
    assert len(r.pairs) == 3
    assert bool(r.pairs.set_index(["method_a", "method_b"]).loc[("good", "bad"), "significant"])


def test_holm_wilcoxon_vs() -> None:
    df = _toy_runs()
    out = holm_wilcoxon_vs(df, baseline="mid", metric="accuracy")
    assert set(out["method"]) == {"good", "bad"}
    assert (out["p_holm"] >= out["p_raw"]).all()
    good = out.set_index("method").loc["good"]
    assert good["mean_accuracy_diff"] > 0 and good["rank_biserial"] > 0


def test_method_summary_and_stability() -> None:
    df = _toy_runs()
    ms = method_summary(df)
    assert {"test_acc", "gen_gap", "n_features"}.issubset(ms.columns)
    assert list(ms.index) == ["good", "mid", "bad"] or set(ms.index) == {"good", "mid", "bad"}

    st = feature_stability(df)
    # identical index sets across seeds -> perfectly stable
    assert np.isclose(st["nogueira_stability"].dropna(), 1.0).all()
