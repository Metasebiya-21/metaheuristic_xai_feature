"""Plotting and numerical helpers: convergence curves, method/metric bar charts, image display.

File-writing plot functions render with the non-interactive ``Agg`` backend.
``display_saved_image`` uses the caller's active backend (for notebooks).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)

# Larger default font sizes for the figures embedded in the report. These are
# tuned relative to each function's (small) figsize below, not in isolation:
# LaTeX rescales the saved PNG to a fixed page width regardless of its native
# pixel size, so on-page legibility depends on the text-to-canvas ratio, not
# the raw font point size -- a bigger font on a proportionally bigger canvas
# renders the same size on the page. Keep figsize small and fonts large.
_REPORT_RC = {
    "font.size": 17,
    "axes.titlesize": 21,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
}


# ============================================================
# Convergence trace alignment
# ============================================================

def resample_to_length(
    y: ArrayLike, target_len: int, last_value_pad: bool = True
) -> NDArray[np.floating]:
    """Resample 1D ``y`` to ``target_len`` by linear interpolation, or pad/trim."""
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
    return yu


def mean_convergence(
    list_of_runs: Sequence[Sequence[float]], target_len: int = 100
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Align each run to ``target_len`` then return per-step mean and std across runs."""
    mats = [resample_to_length(s, target_len) for s in list_of_runs]
    a = np.stack(mats, axis=0)
    return np.mean(a, axis=0), np.std(a, axis=0)


def plot_fitness_convergence(
    history_by_method: dict[str, List[float] | List[List[float]]],
    path: Path,
    title: str = "Convergence: best fitness vs iteration (single run or mean)",
) -> None:
    """Line plot of best fitness per method (mean + std band when given per-run lists)."""
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
            runs = [list(x) for x in hist]
            if not runs or not any(r for r in runs if r):
                continue
            t_len = int(max(len(x) for x in runs if x is not None))
            if t_len < 1:
                continue
            t_len = min(500, t_len)
            m, s = mean_convergence(runs, t_len)
            x = np.arange(1, len(m) + 1)
            (line,) = plt.plot(x, m, label=f"{name} (mean of runs)")
            plt.fill_between(x, m - s, m + s, alpha=0.15, color=line.get_color())
        else:
            h1 = [float(x) for x in hist]
            if not h1:
                continue
            plt.plot(range(1, len(h1) + 1), h1, label=f"{name}")
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


# ============================================================
# Bar charts
# ============================================================

def plot_method_bars(
    series: dict[str, float] | pd.Series,
    ylabel: str,
    path: Path,
    title: str = "",
) -> None:
    """Bar chart of one float per method."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if isinstance(series, dict):
        methods = list(series.keys())
        vals = [float(series[k]) for k in methods]
    else:
        series = series.dropna()
        methods = [str(x) for x in series.index.tolist()]
        vals = [float(x) for x in series.values]

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
    return {str(m): float(v) for m, v in zip(summary[method_key], summary[column])}


def plot_metric_bars(
    df: pd.DataFrame,
    *,
    metric: str,
    output_path: str | Path,
    dataset: str | None = None,
    classifier: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    log_scale: bool | None = None,
    method_col: str = "method",
    color: str = "tab:blue",
) -> Path:
    """Save a bar plot of mean metric (with std error bars) by method."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    summary = df.groupby(method_col)[metric].agg(["mean", "std"]).reset_index()

    if ylabel is None:
        ylabel = metric.replace("_", " ").title()
    if title is None:
        dataset_label = f"{dataset} / " if dataset is not None else ""
        classifier_label = classifier if classifier is not None else ""
        title = f"{metric.replace('_', ' ').title()} by method — {dataset_label}{classifier_label}".rstrip(" — ")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = range(len(summary))
    ax.bar(positions, summary["mean"], yerr=summary["std"], capsize=4, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Method")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(summary[method_col], rotation=15, ha="right")

    if log_scale is None:
        log_scale = bool(summary["mean"].max() / max(summary["mean"].min(), 1e-9) > 20)
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel(f"{ylabel} (log scale)")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================
# Image display (notebooks)
# ============================================================

def display_saved_image(
    image_path: str | Path,
    *,
    figsize: tuple[float, float] = (8, 5),
    title: str | None = None,
    axis_off: bool = True,
) -> None:
    """Render a saved PNG inline in a notebook.

    Uses ``IPython.display.Image`` when available so the figure is embedded as
    output regardless of the matplotlib backend (works under ``nbconvert``'s Agg
    backend, where ``plt.show()`` is a no-op). Falls back to matplotlib otherwise.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    try:
        from IPython.display import Image, display

        if title:
            from IPython.display import Markdown

            display(Markdown(f"**{title}**"))
        display(Image(filename=str(path)))
        return
    except Exception:
        pass

    import matplotlib.pyplot as plt

    img = plt.imread(path)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    if axis_off:
        ax.axis("off")
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ============================================================
# Benchmark analysis figures
# ============================================================

_FAMILY_COLOR = {
    "scalar_metaheuristic": "#d1495b",
    "multiobjective": "#8d3b72",
    "xai": "#2e86ab",
    "classical": "#3c896d",
    "baseline": "#6c757d",
    "unknown": "#6c757d",
}


def _agg_by_method(df: pd.DataFrame, col: str, how: str = "mean") -> "pd.Series":
    s = df.groupby("method")[col].agg(how)
    return s.sort_values()


def plot_accuracy_parsimony(df: pd.DataFrame, path: Path, title: str = "Accuracy vs. subset size") -> None:
    """Scatter of mean held-out test accuracy against mean number of selected
    features, one labelled point per method, coloured by family."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(_REPORT_RC)

    g = df.groupby("method").agg(
        acc=("accuracy", "mean"),
        nfeat=("selected_features", "mean"),
        fam=("method_family", "first"),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Width stays as before (it sets the on-page font-size ratio once LaTeX
    # rescales to a fixed page width); only height grows, so the plot gets
    # taller/roomier on the page without shrinking the text.
    fig, ax = plt.subplots(figsize=(9, 7.5))
    g = g.sort_values("nfeat")
    for j, (method, row) in enumerate(g.iterrows()):
        ax.scatter(row["nfeat"], row["acc"], s=110,
                   color=_FAMILY_COLOR.get(row["fam"], "#6c757d"),
                   edgecolors="0.2", zorder=3)
        dy = 10 if j % 2 == 0 else -18
        ax.annotate(method, (row["nfeat"], row["acc"]), xytext=(7, dy),
                    textcoords="offset points", fontsize=16)
    ax.set_xscale("log")
    # extra headroom on the right so the rightmost point's label doesn't run
    # into the legend anchored just outside the axes
    x_lo, x_hi = ax.get_xlim()
    ax.set_xlim(x_lo, x_hi * 3.2)
    ax.set_xlabel("Mean # selected features (log scale)")
    ax.set_ylabel("Mean held-out test accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, label=name)
        for name, c in [("metaheuristic", _FAMILY_COLOR["scalar_metaheuristic"]),
                        ("multi-objective", _FAMILY_COLOR["multiobjective"]),
                        ("XAI", _FAMILY_COLOR["xai"]),
                        ("classical", _FAMILY_COLOR["classical"]),
                        ("baseline", _FAMILY_COLOR["baseline"])]
    ]
    ax.legend(handles=handles, fontsize=15, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved accuracy/parsimony plot: %s", path)


def plot_generalization_gap_bars(
    df: pd.DataFrame, path: Path, title: str = "Search overfitting: validation - test accuracy"
) -> None:
    """Bar chart of the mean (val_accuracy - test_accuracy) gap per method."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(_REPORT_RC)

    if "generalization_gap" not in df.columns:
        logger.warning("no 'generalization_gap' column; skipping %s", path)
        return
    s = _agg_by_method(df, "generalization_gap")
    fams = df.groupby("method")["method_family"].first()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(s.index, s.values,
           color=[_FAMILY_COLOR.get(fams.get(m, "unknown"), "#6c757d") for m in s.index],
           edgecolor="0.2")
    ax.axhline(0, color="0.3", lw=1)
    ax.set_ylabel("val accuracy - test accuracy")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25, labelsize=16)
    plt.setp(ax.get_xticklabels(), ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved generalization-gap plot: %s", path)


def plot_stability_bars(
    df: pd.DataFrame, path: Path, title: str = "Selection stability across seeds"
) -> None:
    """Bar chart of mean Nogueira stability and mean pairwise Jaccard per method."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(_REPORT_RC)

    from metaheuristic_xai.evaluation import feature_stability

    st = feature_stability(df).groupby("method")[["mean_jaccard", "nogueira_stability"]].mean()
    st = st.sort_values("nogueira_stability")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(st))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - 0.2, st["nogueira_stability"], width=0.4, label="Nogueira index", color="#2e86ab")
    ax.bar(x + 0.2, st["mean_jaccard"], width=0.4, label="mean pairwise Jaccard", color="#8ecae6")
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(st.index, rotation=25, ha="right", fontsize=16)
    ax.set_ylabel("stability (1 = identical, ~0 = random)")
    ax.set_title(title)
    ax.legend(fontsize=15)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved stability plot: %s", path)


def plot_cd_diagram(
    mean_ranks: "pd.Series", cd: float, path: Path,
    title: str = "Friedman + Nemenyi critical-difference diagram",
) -> None:
    """Demsar critical-difference diagram: methods on a rank axis (best rank on the
    left), labels stacked left/right, with cliques of methods that are not
    significantly different (rank gap <= CD) joined by thick bars below the axis."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(_REPORT_RC)

    ranks = mean_ranks.sort_values()
    names = list(ranks.index)
    vals = [float(v) for v in ranks.values]
    k = len(names)
    lo = int(np.floor(min(vals)))
    hi = int(np.ceil(max(vals)))

    half = (k + 1) // 2
    step = 1.3                      # vertical spacing between stacked labels

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the canvas small relative to the (large) font sizes below: this figure
    # gets scaled down to a fixed page width regardless of pixel count, so what
    # matters for on-page legibility is the text-to-canvas ratio, not raw font pt.
    fig, ax = plt.subplots(figsize=(9, 0.42 * k + 2.0))
    ax.set_xlim(lo - 1.3, hi + 1.3)   # lowest (best) rank on the left, extra room for big labels
    ax.set_ylim(-(half * step + 1.4), 2.8)
    ax.axis("off")

    # top rank axis
    ax.plot([lo, hi], [0, 0], color="k", lw=1.6)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [0, 0.14], color="k", lw=1.6)
        ax.text(t, 0.32, str(t), ha="center", va="bottom", fontsize=20)

    # CD ruler
    y_cd = 1.6
    ax.plot([lo, lo + cd], [y_cd, y_cd], color="k", lw=3)
    for xx in (lo, lo + cd):
        ax.plot([xx, xx], [y_cd - 0.12, y_cd + 0.12], color="k", lw=3)
    ax.text(lo + cd / 2, y_cd + 0.26, f"CD = {cd:.2f}", ha="center", fontsize=20)

    for i, (name, r) in enumerate(zip(names, vals)):
        left = i < half
        row = i if left else (k - 1 - i)
        y = -(row + 1) * step
        edge = (lo - 1.3) if left else (hi + 1.3)
        ax.plot([r, r], [0, y], color="0.35", lw=1.3)
        ax.plot([r, edge], [y, y], color="0.35", lw=1.3)
        ax.text(edge, y, f"{name}  ({r:.2f})  ", va="center",
                ha="right" if left else "left", fontsize=21)

    # cliques: maximal runs of consecutive methods within CD
    y_bar = -0.34
    j = 0
    while j < k:
        m = j
        while m + 1 < k and (vals[m + 1] - vals[j]) <= cd:
            m += 1
        if m > j:
            ax.plot([vals[j] - 0.05, vals[m] + 0.05], [y_bar, y_bar],
                    color="crimson", lw=6, solid_capstyle="round")
            y_bar -= 0.28
        j = m + 1 if m > j else j + 1

    ax.set_title(title, fontsize=26, pad=14)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved CD diagram: %s", path)


def plot_leakage_verification(
    df: pd.DataFrame, path: Path,
    title: str = "GA feature-selection accuracy: search scored on test set vs held out",
) -> None:
    """Grouped bars per dataset: GA with the search scored on the test set
    (original), GA leakage-free, and Tree-SHAP -- from
    :func:`metaheuristic_xai.runner.run_leakage_verification` output."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(_REPORT_RC)

    ga = df[df["method"] == "GA"].pivot_table(
        index="dataset", columns="condition", values="test_accuracy", aggfunc="mean"
    )
    shap = df[df["method"] == "Tree-SHAP"].groupby("dataset")["test_accuracy"].mean()
    order = [d for d in ["wdbc", "ionosphere", "madelon", "colon", "leukemia"] if d in ga.index]
    ga, shap = ga.reindex(order), shap.reindex(order)
    orig_col = next(c for c in ga.columns if "test" in c)
    free_col = next(c for c in ga.columns if "free" in c)

    x = np.arange(len(order))
    w = 0.27
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - w, ga[orig_col], w, label="GA — search on test set (original)", color="#d1495b")
    ax.bar(x, ga[free_col], w, label="GA — leakage-free", color="#edae49")
    ax.bar(x + w, shap.values, w, label="Tree-SHAP (top-10)", color="#2e86ab")
    for i in x:
        ax.annotate(f"+{(ga[orig_col].iloc[i] - ga[free_col].iloc[i]) * 100:.1f}",
                    (i - w / 2, max(ga[orig_col].iloc[i], ga[free_col].iloc[i]) + 0.005),
                    ha="center", fontsize=14, color="#7a1f2b")
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=15)
    ax.set_ylabel("held-out test accuracy")
    ax.set_ylim(min(0.6, float(np.nanmin([ga.min().min(), shap.min()])) - 0.05), 1.02)
    ax.set_title(title)
    ax.legend(fontsize=13, loc="lower left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved leakage-verification plot: %s", path)


__all__ = [
    "build_bar_series_from_summary",
    "display_saved_image",
    "plot_accuracy_parsimony",
    "plot_cd_diagram",
    "plot_generalization_gap_bars",
    "plot_leakage_verification",
    "plot_stability_bars",
    "mean_convergence",
    "plot_fitness_convergence",
    "plot_method_bars",
    "plot_metric_bars",
    "resample_to_length",
]
