"""Tests for notebook benchmark utility helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from metaheuristic_xai.evaluation import build_descriptive_summary
from metaheuristic_xai.plots import plot_metric_bars


def test_build_descriptive_summary(tmp_path: Path) -> None:
    """Descriptive summary should aggregate the requested metrics."""
    df = pd.DataFrame(
        {
            "dataset": ["wdbc", "wdbc"],
            "classifier": ["random_forest", "random_forest"],
            "method": ["ga", "shap"],
            "accuracy": [0.9, 0.8],
            "feature_ratio": [0.3, 0.6],
            "runtime": [1.0, 2.0],
            "n_evaluations": [100, 200],
        }
    )

    result = build_descriptive_summary(df, ["accuracy", "feature_ratio", "runtime", "n_evaluations"])

    assert list(result.columns[:3]) == ["dataset", "classifier", "method"]
    assert "accuracy_mean" in result.columns
    assert "runtime_std" in result.columns


def test_plot_metric_bars_creates_png(tmp_path: Path) -> None:
    """Bar plotting should save the output figure to disk."""
    df = pd.DataFrame(
        {
            "method": ["ga", "shap"],
            "accuracy": [0.9, 0.8],
        }
    )
    out_path = tmp_path / "accuracy_plot.png"

    result = plot_metric_bars(
        df,
        metric="accuracy",
        output_path=out_path,
        dataset="wdbc",
        classifier="random_forest",
        ylabel="Accuracy",
        title="Accuracy vs method",
    )

    assert result == out_path
    assert out_path.exists()
