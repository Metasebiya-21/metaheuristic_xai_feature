"""
Statistical hypothesis tests (Wilcoxon, Friedman, Nemenyi), result CSV export,
and post-hoc summary tables for the analysis notebooks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Result of a statistical test between methods or across multiple methods."""

    test_type: str
    comparison: str
    statistic: float
    pvalue: float
    effect_size: float
    ci_lower: float
    ci_upper: float
    n_samples: int


def calculate_effect_size_cohen_d(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """Calculate Cohen's d effect size between two paired vectors."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def compute_confidence_interval(data: NDArray[np.floating], confidence: float = 0.95) -> tuple[float, float]:
    """Compute standard normal 95% confidence interval for a 1D vector."""
    a = np.asarray(data, dtype=np.float64).ravel()
    if a.size < 2:
        return float(a[0]) if a.size == 1 else 0.0, float(a[0]) if a.size == 1 else 0.0
    m = np.mean(a)
    se = stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2.0, len(a) - 1)
    return float(m - h), float(m + h)


def wilcoxon_paired_test(
    acc_method: NDArray[np.floating],
    acc_baseline: NDArray[np.floating],
    name_method: str,
    name_baseline: str = "SHAP",
) -> Optional[StatisticalResult]:
    """
    Two-sided Wilcoxon signed-rank test on paired experimental units (dataset + classifier + seed).
    """
    a = np.asarray(acc_method, dtype=np.float64).ravel()
    b = np.asarray(acc_baseline, dtype=np.float64).ravel()

    if a.size != b.size or a.size < 2:
        return None

    diffs = a - b
    if np.count_nonzero(diffs) < 1:
        return StatisticalResult(
            test_type="Wilcoxon",
            comparison=f"{name_method} vs {name_baseline}",
            statistic=0.0,
            pvalue=1.0,
            effect_size=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            n_samples=a.size,
        )

    try:
        res = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        stat = float(res.statistic)
        pval = float(res.pvalue)
    except Exception as e:
        logger.warning("Wilcoxon test failed: %s", e)
        return None

    eff = calculate_effect_size_cohen_d(a, b)
    ci_l, ci_u = compute_confidence_interval(diffs)

    return StatisticalResult(
        test_type="Wilcoxon",
        comparison=f"{name_method} vs {name_baseline}",
        statistic=stat,
        pvalue=pval,
        effect_size=eff,
        ci_lower=ci_l,
        ci_upper=ci_u,
        n_samples=a.size,
    )


def friedman_test(
    matrix_data: NDArray[np.floating], method_names: list[str]
) -> tuple[Optional[StatisticalResult], Optional[pd.DataFrame]]:
    """
    Friedman test across multiple methods on matched subjects (rows = paired splits, cols = methods).
    """
    mat = np.asarray(matrix_data, dtype=np.float64)
    n_subjects, n_methods = mat.shape

    if n_subjects < 3 or n_methods < 3:
        return None, None

    try:
        res = stats.friedmanchisquare(*[mat[:, c] for c in range(n_methods)])
        stat = float(res.statistic)
        pval = float(res.pvalue)
    except Exception as e:
        logger.warning("Friedman test failed: %s", e)
        return None, None

    f_res = StatisticalResult(
        test_type="Friedman",
        comparison="All Methods",
        statistic=stat,
        pvalue=pval,
        effect_size=0.0,
        ci_lower=0.0,
        ci_upper=0.0,
        n_samples=n_subjects,
    )

    # Nemenyi post-hoc critical difference
    ranks = np.array([stats.rankdata(-mat[i, :]) for i in range(n_subjects)])
    mean_ranks = np.mean(ranks, axis=0)

    posthoc_rows: list[dict[str, Any]] = []
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            m1, m2 = method_names[i], method_names[j]
            rank_diff = abs(mean_ranks[i] - mean_ranks[j])
            posthoc_rows.append(
                {
                    "comparison": f"{m1} vs {m2}",
                    "method1": m1,
                    "method2": m2,
                    "mean_rank1": float(mean_ranks[i]),
                    "mean_rank2": float(mean_ranks[j]),
                    "rank_diff": float(rank_diff),
                }
            )

    return f_res, pd.DataFrame(posthoc_rows)


def save_results(df: pd.DataFrame, output_dir: Path) -> None:
    """Save the raw runs and a per-(dataset, classifier, method) summary as CSV.

    The summary aggregates whichever of the known metric columns are present, so
    it works for both the full benchmark record and leaner sensitivity records.
    """
    out = Path(output_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "aggregated").mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "raw" / "all_runs.csv", index=False)
    df.to_csv(out / "all_runs.csv", index=False)

    if df.empty or not {"dataset", "classifier", "method"}.issubset(df.columns):
        logger.info("Saved raw results to %s (no summary: missing group columns)", out)
        return

    # (output column, source column, aggregation) — source columns may be absent.
    candidates = [
        ("accuracy_mean", "accuracy", "mean"),
        ("accuracy_std", "accuracy", "std"),
        ("balanced_accuracy_mean", "balanced_accuracy", "mean"),
        ("f1_mean", "f1", "mean"),
        ("roc_auc_mean", "roc_auc", "mean"),
        ("selected_features_mean", "selected_features", "mean"),
        ("feature_ratio_mean", "feature_ratio", "mean"),
        ("fitness_mean", "fitness", "mean"),
        ("runtime_mean", "runtime_seconds", "mean"),
        ("evaluations_mean", "actual_oracle_evaluations", "mean"),
    ]
    agg = {out_col: (src, how) for out_col, src, how in candidates if src in df.columns}
    group = df.groupby(["dataset", "classifier", "method"])
    summary = group.agg(n_runs=("method", "count"), **agg).reset_index()

    summary.to_csv(out / "aggregated" / "summary.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    logger.info("Saved raw and aggregated results to %s", out)


# ============================================================
# Post-hoc summary tables (used by analysis notebooks)
# ============================================================

_DEFAULT_GROUPBY: tuple[str, ...] = ("dataset", "classifier", "method")


def build_descriptive_summary(
    df: pd.DataFrame,
    metrics: list[str] | tuple[str, ...],
    groupby_cols: list[str] | tuple[str, ...] = _DEFAULT_GROUPBY,
) -> pd.DataFrame:
    """Aggregate benchmark metrics by dataset/classifier/method.

    Produces wide columns ``<metric>_count/mean/std/median/min/max``.
    """
    if df.empty:
        return pd.DataFrame(columns=list(groupby_cols))

    result = df.groupby(list(groupby_cols)).agg(
        {metric: ["count", "mean", "std", "median", "min", "max"] for metric in metrics}
    )
    result.columns = ["_".join(col).strip() for col in result.columns.values]
    return result.reset_index()


def prepare_statistical_summary(
    df: pd.DataFrame,
    metric: str = "accuracy",
    groupby_cols: list[str] | tuple[str, ...] = _DEFAULT_GROUPBY,
) -> pd.DataFrame:
    """Mean/std of a single metric per group, for downstream statistical analysis."""
    if df.empty:
        return pd.DataFrame(columns=list(groupby_cols) + [f"{metric}_mean", f"{metric}_std"])

    grouped = df.groupby(list(groupby_cols))[metric].agg(["mean", "std"]).reset_index()
    return grouped.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})


def build_method_metric_summary(
    df: pd.DataFrame,
    metric: str,
    method_col: str = "method",
) -> pd.DataFrame:
    """Mean/std of a single metric per method."""
    if df.empty:
        return pd.DataFrame(columns=[method_col, f"{metric}_mean", f"{metric}_std"])

    result = df.groupby(method_col)[metric].agg(["mean", "std"]).reset_index()
    return result.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})


def prepare_pareto_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compact Pareto summary (front size + mean accuracy/fitness) per group."""
    cols = ["dataset", "classifier", "method", "pareto_front_size", "mean_accuracy", "mean_fitness"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    return (
        df.groupby(["dataset", "classifier", "method"])
        .agg(
            pareto_front_size=("pareto_front_size", "first"),
            mean_accuracy=("accuracy", "mean"),
            mean_fitness=("fitness", "mean"),
        )
        .reset_index()
    )


# ============================================================
# Selection stability across seeds
# ============================================================

def _as_index_set(value: Any) -> set[int]:
    """Coerce a ``selected_indices`` cell to a set of ints.

    Accepts a Python list/tuple/ndarray, a list-literal string ``"[1, 2, 3]"``, or a
    NumPy array-repr string ``"[ 1  2  3\\n  4 ...]"`` (how a list column round-trips
    through parquet then CSV).
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return {int(x) for x in value}
    if isinstance(value, str):
        import re

        return {int(tok) for tok in re.findall(r"-?\d+", value)}
    return set()


def _mean_pairwise_jaccard(sets: list[set[int]]) -> float:
    """Mean Jaccard similarity over all unordered pairs of selected-feature sets."""
    n = len(sets)
    if n < 2:
        return float("nan")
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            union = sets[i] | sets[j]
            total += 1.0 if not union else len(sets[i] & sets[j]) / len(union)
            count += 1
    return total / count


def _nogueira_stability(sets: list[set[int]], d: int) -> float:
    """Nogueira et al. (2018) stability index: 1 = identical selections, ~0 = random.

    Handles variable subset sizes; chance-corrected. Returns nan for the degenerate
    all-features / no-features case unless every selection is identical.
    """
    m = len(sets)
    if m < 2 or d <= 0:
        return float("nan")
    z = np.zeros((m, d), dtype=np.float64)
    for row, s in enumerate(sets):
        idx = [f for f in s if 0 <= f < d]
        z[row, idx] = 1.0
    p = z.mean(axis=0)
    k_bar = z.sum(axis=1).mean()
    denom = (k_bar / d) * (1.0 - k_bar / d)
    if denom <= 1e-12:
        return 1.0 if all(s == sets[0] for s in sets) else float("nan")
    numer = (m / (m - 1.0)) * np.mean(p * (1.0 - p))
    return float(1.0 - numer / denom)


def feature_stability(
    df: pd.DataFrame,
    groupby_cols: list[str] | tuple[str, ...] = _DEFAULT_GROUPBY,
    indices_col: str = "selected_indices",
    n_features_col: str = "search_features",
) -> pd.DataFrame:
    """Per-group selection stability across seeds.

    For each (dataset, classifier, method) group, treats the per-seed
    ``selected_indices`` as repeated feature subsets and reports:

    * ``mean_jaccard``       -- mean pairwise Jaccard similarity (1 = always the same subset)
    * ``nogueira_stability`` -- chance-corrected stability index (1 = identical, ~0 = random)
    * ``mean_selected`` / ``std_selected`` -- subset-size spread
    """
    out_cols = list(groupby_cols) + [
        "n_seeds", "mean_selected", "std_selected", "mean_jaccard", "nogueira_stability",
    ]
    if df.empty or indices_col not in df.columns:
        return pd.DataFrame(columns=out_cols)

    rows: list[dict[str, Any]] = []
    for keys, sub in df.groupby(list(groupby_cols)):
        sets = [_as_index_set(v) for v in sub[indices_col]]
        sizes = [len(s) for s in sets]
        d = int(sub[n_features_col].iloc[0]) if n_features_col in sub.columns else 0
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            {
                **dict(zip(groupby_cols, key_tuple)),
                "n_seeds": len(sets),
                "mean_selected": float(np.mean(sizes)) if sizes else float("nan"),
                "std_selected": float(np.std(sizes, ddof=1)) if len(sizes) > 1 else 0.0,
                "mean_jaccard": _mean_pairwise_jaccard(sets),
                "nogueira_stability": _nogueira_stability(sets, d),
            }
        )
    return pd.DataFrame(rows, columns=out_cols)


# ============================================================
# Benchmark analysis (leakage-controlled result tables + tests)
# ============================================================

_METHOD_ORDER = [
    "ga", "bpso", "sa", "gwo", "nsga2",
    "shap", "lime", "lasso", "rfe", "boruta", "full_features",
]

# Demsar (2006) Table 5: q_alpha for the Nemenyi test at alpha = 0.05, indexed by
# the number of methods k (already divided by sqrt(2)).
_NEMENYI_Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
    9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
    16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
}


def _ordered_methods(df: pd.DataFrame, methods: Optional[list[str]]) -> list[str]:
    present = list(df["method"].unique())
    if methods is not None:
        return [m for m in methods if m in present]
    return [m for m in _METHOD_ORDER if m in present] + [
        m for m in present if m not in _METHOD_ORDER
    ]


def method_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method roll-up: held-out test accuracy, validation accuracy, search
    overfitting gap, subset size, runtime and search budget."""
    if df.empty:
        return pd.DataFrame()
    agg: dict[str, tuple[str, str]] = {"n": ("method", "count")}
    for out, src in [
        ("test_acc", "accuracy"), ("test_acc_sd", "accuracy"),
        ("balanced_acc", "balanced_accuracy"), ("f1", "f1"), ("roc_auc", "roc_auc"),
        ("val_acc", "val_accuracy"), ("gen_gap", "generalization_gap"),
        ("n_features", "selected_features"), ("feature_ratio", "feature_ratio"),
        ("fitness", "fitness"), ("runtime_s", "runtime_seconds"),
        ("oracle_evals", "actual_oracle_evaluations"),
    ]:
        if src in df.columns:
            how = "std" if out.endswith("_sd") else "mean"
            agg[out] = (src, how)
    out = df.groupby("method").agg(**agg)
    if "method_family" in df.columns:
        out.insert(0, "family", df.groupby("method")["method_family"].first())
    return out.reindex(_ordered_methods(df, None)).round(4)


def per_dataset_metric(
    df: pd.DataFrame, metric: str = "accuracy", methods: Optional[list[str]] = None
) -> pd.DataFrame:
    """method x dataset table of a metric (mean over classifiers and seeds)."""
    if df.empty:
        return pd.DataFrame()
    piv = df.pivot_table(index="method", columns="dataset", values=metric)
    return piv.reindex(_ordered_methods(df, methods)).round(4)


def generalization_gap_by_tier(
    df: pd.DataFrame, methods: Optional[list[str]] = None
) -> pd.DataFrame:
    """method x scale_tier table of the mean (val_accuracy - test_accuracy) gap."""
    if df.empty or "generalization_gap" not in df.columns or "scale_tier" not in df.columns:
        return pd.DataFrame()
    piv = df.pivot_table(index="method", columns="scale_tier", values="generalization_gap")
    tiers = [t for t in ["low", "medium", "ultrahigh"] if t in piv.columns]
    return piv[tiers].reindex(_ordered_methods(df, methods)).round(4)


def friedman_nemenyi(
    df: pd.DataFrame,
    metric: str = "accuracy",
    methods: Optional[list[str]] = None,
    group_cols: tuple[str, ...] = ("dataset", "classifier", "seed"),
    alpha: float = 0.05,
) -> SimpleNamespace:
    """Friedman omnibus test over methods on ``metric`` with the Nemenyi
    critical-difference post-hoc (Demsar 2006 protocol).

    Subjects are the matched ``group_cols`` cells; a cell is dropped unless every
    method has a value for it. Lower mean rank = better ``metric``.

    Returns a namespace with ``statistic``, ``pvalue``, ``n_subjects``,
    ``n_methods``, ``mean_ranks`` (Series, ascending), ``cd`` and ``pairs``
    (DataFrame: method_a, method_b, rank_diff, significant).
    """
    ms = _ordered_methods(df, methods)
    piv = df.pivot_table(index=list(group_cols), columns="method", values=metric)[ms].dropna()
    n, k = piv.shape
    if n < 2 or k < 3:
        return SimpleNamespace(
            statistic=float("nan"), pvalue=float("nan"), n_subjects=n, n_methods=k,
            mean_ranks=pd.Series(dtype=float), cd=float("nan"),
            pairs=pd.DataFrame(columns=["method_a", "method_b", "rank_diff", "significant"]),
        )

    stat, pval = stats.friedmanchisquare(*[piv[m].to_numpy() for m in ms])
    ranks = piv.rank(axis=1, ascending=False)          # 1 = best metric value
    mean_ranks = ranks.mean().sort_values()
    q = _NEMENYI_Q05.get(k, 3.544)
    cd = float(q * np.sqrt(k * (k + 1) / (6.0 * n)))

    rows = []
    for i, a in enumerate(mean_ranks.index):
        for b in mean_ranks.index[i + 1:]:
            diff = abs(mean_ranks[a] - mean_ranks[b])
            rows.append({"method_a": a, "method_b": b, "rank_diff": round(diff, 3),
                         "significant": bool(diff > cd)})
    return SimpleNamespace(
        statistic=float(stat), pvalue=float(pval), n_subjects=int(n), n_methods=int(k),
        mean_ranks=mean_ranks.round(3), cd=cd, pairs=pd.DataFrame(rows),
    )


def holm_wilcoxon_vs(
    df: pd.DataFrame,
    baseline: str = "shap",
    metric: str = "accuracy",
    methods: Optional[list[str]] = None,
    group_cols: tuple[str, ...] = ("dataset", "classifier", "seed"),
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank of every method vs ``baseline`` on ``metric``,
    with Holm-corrected p-values and matched-pairs rank-biserial effect size.
    """
    ms = [m for m in _ordered_methods(df, methods) if m != baseline]
    base = df[df["method"] == baseline].set_index(list(group_cols))[metric]
    rows = []
    for m in ms:
        sub = df[df["method"] == m].set_index(list(group_cols))[metric]
        pair = pd.concat({"m": sub, "b": base}, axis=1).dropna()
        diff = (pair["m"] - pair["b"]).to_numpy()
        nz = diff[diff != 0]
        if nz.size == 0:
            p_raw, rbc = 1.0, 0.0
        else:
            try:
                p_raw = float(stats.wilcoxon(pair["m"], pair["b"]).pvalue)
            except ValueError:
                p_raw = 1.0
            ranks = stats.rankdata(np.abs(nz))
            rbc = float(np.sum(np.sign(nz) * ranks) / (nz.size * (nz.size + 1) / 2))
        rows.append({
            "method": m, "n_pairs": int(diff.size),
            f"mean_{metric}_diff": round(float(np.mean(diff)), 4),
            "win_rate": round(float(np.mean(diff > 0)), 3),
            "rank_biserial": round(rbc, 3), "p_raw": p_raw,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        order = np.argsort(out["p_raw"].to_numpy())
        k = len(out)
        holm = np.empty(k)
        running = 0.0
        for i, idx in enumerate(order):
            running = max(running, min(1.0, out["p_raw"].iloc[idx] * (k - i)))
            holm[idx] = running
        out["p_holm"] = holm.round(4)
        out["p_raw"] = out["p_raw"].round(4)
        out = out.sort_values(f"mean_{metric}_diff", ascending=False).reset_index(drop=True)
    return out

