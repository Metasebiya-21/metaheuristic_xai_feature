"""
Plotting and numerical helpers: convergence curves, bar charts, and alignment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)


def resample_to_length(
    y: ArrayLike, target_len: int, last_value_pad: bool = True
) -> NDArray[np.floating]:
    """
    Resample 1D ``y`` to length ``target_len`` by **linear** interpolation
    in index space, or by padding/trimming if ``len(y)`` is wrong.

    * If longer: subsample (equally-spaced) ``target_len`` values.
    * If shorter: pad with the last value at the end (typical for early-stopped
      optimization traces), else pad with the last element once.
    """
    yu = np.asarray(y, dtype=np.float64).ravel()
    n = yu.size
    if n == 0:
        return np.full(int(target_len), np.nan, dtype=np.float64)
    if n == 1 or target_len == 1:
        yp = yu[0] if n else np.nan
        return np.full(int(target_len), yp, dtype=np.float64)
    if n >= target_len:
        idx = np.round(np.linspace(0, n - 1, target_len)).astype(int)
        return yu[idx]
    if not last_value_pad and n < target_len:
        idx = np.linspace(0, n - 1, int(target_len))
        return np.interp(idx, np.arange(n, dtype=np.float64), yu)
    if last_value_pad:
        p = int(target_len) - n
        return np.pad(yu, (0, p), mode="edge")[: int(target_len)]
    return yu  # should not happen


def mean_convergence(
    list_of_runs: Sequence[Sequence[float]], target_len: int = 100
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    For multiple runs, align each to ``target_len`` then take mean and std
    (sample std, ddof=0 for stability) across runs.
    """
    mats: list[NDArray[np.floating]] = [
        resample_to_length(s, target_len) for s in list_of_runs
    ]
    a = np.stack(mats, axis=0)
    return np.mean(a, axis=0), np.std(a, axis=0)


def plot_fitness_convergence(
    history_by_method: dict[str, List[float] | List[List[float]]],
    path: Path,
    title: str = "Convergence: best fitness vs iteration (single run or mean)",
) -> None:
    """
    Line plot of fitness. If a list is ``list of lists`` (per-run), plot mean
    and light std band; if a 1D list, plot a single line.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    for name, hist in history_by_method.items():
        if (
            isinstance(hist, list)
            and len(hist) > 0
            and isinstance(hist[0], (list, tuple, np.ndarray))
        ):
            runs = [list(x) for x in hist]  # type: ignore[assignment]
            if not runs or not any(r for r in runs if r):
                continue
            t_len = int(max(len(x) for x in runs if x is not None))
            if t_len < 1:
                continue
            t_len = min(500, t_len)  # cap for very long SA
            m, s = mean_convergence(runs, t_len)  # type: ignore[arg-type]
            x = np.arange(1, len(m) + 1)
            (line,) = plt.plot(x, m, label=f"{name} (mean of runs)")
            plt.fill_between(
                x,
                m - s,
                m + s,
                alpha=0.15,
                color=line.get_color(),  # type: ignore[union-attr]
            )
        else:
            h1 = [float(x) for x in hist]  # type: ignore[union-attr, arg-type]
            if not h1:
                continue
            (line,) = plt.plot(
                range(1, len(h1) + 1), h1, label=f"{name}"
            )
    plt.xlabel("Step / generation / iteration")
    plt.ylabel("Best fitness (lower = better)")
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        plt.close()
    logger.info("Saved convergence plot: %s", path)


def plot_method_bars(
    series: dict[str, float] | pd.Series,
    ylabel: str,
    path: Path,
    title: str = "",
) -> None:
    """Bar chart: one float per method."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if isinstance(series, dict):
        methods = list(series.keys())
        vals = [float(series[k]) for k in methods]
    else:
        series = series.dropna()  # type: ignore[assignment]
        methods = [str(x) for x in series.index.tolist()]  # type: ignore[union-attr, attr-defined]
        vals = [float(x) for x in series.values]  # type: ignore[union-attr, attr-defined]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 3.2))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(methods)))
    plt.bar(methods, vals, color=colors, edgecolor="0.2")
    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.ylabel(ylabel, fontsize=10)
    if title:
        plt.title(title, fontsize=10)
    plt.tight_layout()
    try:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        plt.close()
    logger.info("Saved bar plot: %s", path)


def build_bar_series_from_summary(
    summary: pd.DataFrame, column: str, method_key: str = "method"
) -> dict[str, float]:
    """``{method: value}`` for a numeric column in a summary frame."""
    if method_key not in summary.columns or column not in summary.columns:
        return {}
    return {str(m): float(v) for m, v in zip(summary[method_key], summary[column])}  # type: ignore[call-overload, arg-type]


__all__ = [
    "resample_to_length",
    "mean_convergence",
    "plot_fitness_convergence",
    "plot_method_bars",
    "build_bar_series_from_summary",
]
