"""
Result aggregation, CSV export, and non-parametric statistical tests (vs SHAP).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

METHODS = ("SHAP", "GA", "PSO", "SA")


@dataclass
class RunRecord:
    """One benchmark run of a method on a fixed train/test split."""

    run_id: int
    method: str
    accuracy: float
    n_features: int
    fitness: float
    runtime: float
    shap_time: float = 0.0
    best_mask_hash: str = ""


@dataclass
class WilcoxonResult:
    """Result of a two-sided Wilcoxon rank-sum (Mann-Whitney U) style comparison."""

    comparison: str
    statistic: float
    pvalue: float
    n_a: int
    n_b: int


def records_to_dataframe(
    records: list[RunRecord] | list[dict[str, Any]]
) -> pd.DataFrame:
    """
    Coerce a list of :class:`RunRecord` or row dicts to a :class:`pandas.DataFrame`.
    """
    if not records:
        return pd.DataFrame(
            columns=["run_id", "method", "accuracy", "n_features", "fitness", "runtime"]
        )
    if isinstance(records[0], RunRecord):
        data = [asdict(r) for r in records]
    else:
        data = list(records)
    return pd.DataFrame(data)


def build_summary(
    records: list[RunRecord] | list[dict[str, Any]], methods: list[str] | None = None
) -> pd.DataFrame:
    """
    Compute mean, standard deviation, and per-method counts from run records.
    """
    if methods is None:
        methods = list(METHODS)
    frame = records_to_dataframe(records)
    if frame.empty:
        return frame
    rows: list[dict[str, Any]] = []
    for m in methods:
        sub = frame[frame["method"] == m]
        if sub.empty:
            continue
        rows.append(
            {
                "method": m,
                "n_runs": int(len(sub)),
                "accuracy_mean": float(sub["accuracy"].mean()),
                "accuracy_std": float(sub["accuracy"].std(ddof=1) if len(sub) > 1 else 0.0),
                "n_features_mean": float(sub["n_features"].mean()),
                "n_features_std": float(sub["n_features"].std(ddof=1) if len(sub) > 1 else 0.0),
                "fitness_mean": float(sub["fitness"].mean()),
                "fitness_std": float(sub["fitness"].std(ddof=1) if len(sub) > 1 else 0.0),
                "runtime_mean": float(sub["runtime"].mean()),
                "runtime_std": float(sub["runtime"].std(ddof=1) if len(sub) > 1 else 0.0),
            }
        )
    return pd.DataFrame(rows)


def wilcoxon_against_baseline(
    acc_method: NDArray[np.floating],
    acc_baseline: NDArray[np.floating],
    name_method: str,
    name_baseline: str = "SHAP",
) -> Optional[WilcoxonResult]:
    """
    **Wilcoxon rank-sum test** (independent two-sample): compare a vector of
    *accuracies* (one per independent data split) from a metaheuristic to the
    SHAP baseline. Uses :func:`scipy.stats.ranksums` (two-sided).

    Returns
    -------
    result or None
        ``None`` if either sample is empty or the test cannot be run.
    """
    a = np.asarray(acc_method, dtype=np.float64).ravel()
    b = np.asarray(acc_baseline, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        logger.info(
            "Wilcoxon skipped: need n>=2 per group; got n=%d vs n=%d.",
            a.size,
            b.size,
        )
        return None
    try:
        try:
            res = stats.ranksums(a, b, alternative="two-sided")
        except TypeError:
            res = stats.ranksums(a, b)
    except (ValueError, TypeError) as e:
        logger.error("ranksums failed: %s", e)
        return None
    label = f"{name_method} vs {name_baseline} (accuracy)"
    pval = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])  # type: ignore[union-attr]
    return WilcoxonResult(
        comparison=label,
        statistic=float(res.statistic) if hasattr(res, "statistic") else float(res[0]),  # type: ignore[union-attr]
        pvalue=pval,
        n_a=a.size,
        n_b=b.size,
    )


def all_wilcoxon_vs_shap(
    frame: pd.DataFrame, baseline_name: str = "SHAP"
) -> list[WilcoxonResult]:
    """Run SHAP-baseline tests for **GA, PSO, and SA** when columns match."""
    out: list[WilcoxonResult] = []
    base = frame[frame["method"] == baseline_name]
    if base.empty:
        logger.warning("No SHAP rows; Wilcoxon not computed.")
        return out
    acc_s = base["accuracy"].to_numpy()
    for other in ("GA", "PSO", "SA"):
        sub = frame[frame["method"] == other]
        if sub.empty:
            continue
        r = wilcoxon_against_shap(
            acc_method=sub["accuracy"].to_numpy(), acc_baseline=acc_s, name_method=other
        )
        if r is not None:
            out.append(r)
    return out


def wilcoxon_against_shap(
    acc_method: NDArray,
    acc_baseline: NDArray,
    name_method: str,
) -> Optional[WilcoxonResult]:
    """Convenience wrapper using ``SHAP`` as the default baseline name."""
    return wilcoxon_against_baseline(
        acc_method, acc_baseline, name_method, "SHAP"
    )


def save_results_csv(
    records: list[RunRecord] | list[dict[str, Any]], path: Path
) -> None:
    """Write raw ``records`` to ``<path>`` (CSV)."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Cannot create results directory: %s", e)
        raise
    df = records_to_dataframe(records)
    try:
        df.to_csv(path, index=False)
    except OSError as e:
        logger.error("Failed to write %s: %s", path, e)
        raise
    logger.info("Saved run-level CSV: %s (%d rows).", path, len(df))


def save_summary_table(df: pd.DataFrame, path: Path) -> None:
    """Persist a summary (aggregated) table to CSV."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise
    df.to_csv(path, index=False)
    logger.info("Saved summary CSV: %s", path)
