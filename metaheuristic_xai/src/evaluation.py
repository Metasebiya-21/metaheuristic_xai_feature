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
    """Result of a two-sided Wilcoxon signed-rank (paired) comparison."""

    comparison: str
    statistic: float
    pvalue: float
    n_pairs: int


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
    **Wilcoxon signed-rank test** (paired): compare accuracies on matched
    train/test splits (same ``run_id``). Uses :func:`scipy.stats.wilcoxon`
    (two-sided) on the per-split differences.

    ``acc_method`` and ``acc_baseline`` must be aligned (same length, same order).

    Returns
    -------
    result or None
        ``None`` if pairing is invalid or the test cannot be run.
    """
    a = np.asarray(acc_method, dtype=np.float64).ravel()
    b = np.asarray(acc_baseline, dtype=np.float64).ravel()
    if a.size != b.size:
        logger.error(
            "Paired Wilcoxon requires equal-length vectors; got n=%d vs n=%d.",
            a.size,
            b.size,
        )
        return None
    if a.size < 2:
        logger.info(
            "Wilcoxon skipped: need n>=2 paired splits; got n=%d.",
            a.size,
        )
        return None
    diffs = a - b
    n_nonzero = int(np.count_nonzero(diffs))
    if n_nonzero < 1:
        logger.warning(
            "Wilcoxon skipped for %s vs %s: all paired differences are zero.",
            name_method,
            name_baseline,
        )
        return None
    try:
        try:
            res = stats.wilcoxon(
                a, b, alternative="two-sided", zero_method="wilcox"
            )
        except TypeError:
            res = stats.wilcoxon(a, b, zero_method="wilcox")
    except (ValueError, TypeError) as e:
        logger.error("wilcoxon failed: %s", e)
        return None
    label = f"{name_method} vs {name_baseline} (accuracy)"
    pval = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])  # type: ignore[union-attr]
    return WilcoxonResult(
        comparison=label,
        statistic=float(res.statistic) if hasattr(res, "statistic") else float(res[0]),  # type: ignore[union-attr]
        pvalue=pval,
        n_pairs=a.size,
    )


def all_wilcoxon_vs_shap(
    frame: pd.DataFrame, baseline_name: str = "SHAP"
) -> list[WilcoxonResult]:
    """
    Run paired SHAP-baseline Wilcoxon signed-rank tests for GA, PSO, and SA.

    Accuracies are aligned on ``run_id`` so each split contributes one matched pair.
    """
    out: list[WilcoxonResult] = []
    if "run_id" not in frame.columns:
        logger.warning("No run_id column; cannot pair methods for Wilcoxon.")
        return out
    base = frame[frame["method"] == baseline_name]
    if base.empty:
        logger.warning("No SHAP rows; Wilcoxon not computed.")
        return out
    base_acc = base.set_index("run_id")["accuracy"]
    for other in ("GA", "PSO", "SA"):
        sub = frame[frame["method"] == other]
        if sub.empty:
            continue
        other_acc = sub.set_index("run_id")["accuracy"]
        paired = pd.concat(
            [other_acc.rename("method"), base_acc.rename("baseline")],
            axis=1,
            join="inner",
        ).dropna()
        if paired.empty:
            logger.warning("No overlapping run_id for %s vs %s.", other, baseline_name)
            continue
        r = wilcoxon_against_shap(
            acc_method=paired["method"].to_numpy(),
            acc_baseline=paired["baseline"].to_numpy(),
            name_method=other,
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
