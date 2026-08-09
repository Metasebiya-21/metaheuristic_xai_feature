"""
Pareto Front Analysis and Visualization module for NSGA-II.

Calculates:
- Hypervolume indicator
- Pareto front cardinality (number of non-dominated solutions)
- Feature-count diversity and Accuracy diversity
- Pareto-front spread / spacing
- Plots: Accuracy vs Number of Features and Pareto evolution across generations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from metaheuristic_xai.algorithms.nsga2 import ParetoSolution

logger = logging.getLogger(__name__)


def compute_hypervolume_2d(
    pareto_front: Sequence[ParetoSolution],
    reference_point: tuple[float, float] = (1.0, 1.0),
) -> float:
    """
    Compute 2D hypervolume bounded by reference_point (ref_error, ref_feature_ratio).
    Minimizing both obj1 (error) and obj2 (feature_ratio).
    """
    if not pareto_front:
        return 0.0

    # Sort solutions by error ascending
    pts = sorted([(sol.error, sol.feature_ratio) for sol in pareto_front], key=lambda p: p[0])

    ref_err, ref_ratio = reference_point
    hv = 0.0
    prev_ratio = ref_ratio

    for err, ratio in pts:
        if err < ref_err and ratio < prev_ratio:
            hv += (ref_err - err) * (prev_ratio - ratio)
            prev_ratio = ratio

    return float(max(0.0, hv))


def compute_pareto_metrics(pareto_front: Sequence[ParetoSolution]) -> dict[str, float]:
    """Calculate quantitative metrics of a Pareto front."""
    n_sols = len(pareto_front)
    if n_sols == 0:
        return {
            "n_pareto_solutions": 0.0,
            "hypervolume": 0.0,
            "accuracy_diversity": 0.0,
            "feature_diversity": 0.0,
            "pareto_spread": 0.0,
        }

    accs = [sol.accuracy for sol in pareto_front]
    feats = [sol.n_features for sol in pareto_front]

    acc_div = float(np.std(accs)) if n_sols > 1 else 0.0
    feat_div = float(np.std(feats)) if n_sols > 1 else 0.0

    # Pareto spread (distance between extreme solutions in normalized space)
    min_err, max_err = min(sol.error for sol in pareto_front), max(sol.error for sol in pareto_front)
    min_rat, max_rat = min(sol.feature_ratio for sol in pareto_front), max(sol.feature_ratio for sol in pareto_front)
    spread = float(np.sqrt((max_err - min_err) ** 2 + (max_rat - min_rat) ** 2))

    hv = compute_hypervolume_2d(pareto_front)

    return {
        "n_pareto_solutions": float(n_sols),
        "hypervolume": hv,
        "accuracy_diversity": acc_div,
        "feature_diversity": feat_div,
        "pareto_spread": spread,
    }


def plot_pareto_front(
    pareto_front: Sequence[ParetoSolution],
    all_solutions: Sequence[ParetoSolution] | None,
    path: Path,
    title: str = "Pareto Front: Accuracy vs Number of Features",
) -> None:
    """Scatter plot of Accuracy vs Number of Features highlighting Pareto front."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))

    if all_solutions:
        all_feats = [s.n_features for s in all_solutions]
        all_accs = [s.accuracy for s in all_solutions]
        plt.scatter(all_feats, all_accs, color="gray", alpha=0.3, label="Evaluated Solutions", s=25)

    p_feats = [s.n_features for s in pareto_front]
    p_accs = [s.accuracy for s in pareto_front]

    # Sort for step line
    pts = sorted(zip(p_feats, p_accs), key=lambda p: p[0])
    sf_x, sf_y = [p[0] for p in pts], [p[1] for p in pts]

    plt.plot(sf_x, sf_y, color="crimson", linestyle="--", linewidth=1.5, zorder=3)
    plt.scatter(p_feats, p_accs, color="crimson", label="Pareto Front", s=50, edgecolors="k", zorder=4)

    plt.xlabel("Number of Selected Features (|S|)")
    plt.ylabel("Test Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        plt.close()
    logger.info("Saved Pareto front plot: %s", path)


def plot_pareto_evolution(
    pareto_history: Sequence[Sequence[ParetoSolution]],
    gen_labels: Sequence[int],
    path: Path,
    title: str = "Pareto Front Evolution Across Generations",
) -> None:
    """Line/Scatter plot showing Pareto front evolution across generations."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(pareto_history)))

    for i, (front, gen) in enumerate(zip(pareto_history, gen_labels)):
        pts = sorted([(s.n_features, s.accuracy) for s in front], key=lambda p: p[0])
        if not pts:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        plt.plot(xs, ys, label=f"Gen {gen}", color=colors[i], marker="o", markersize=4, linewidth=1.5)

    plt.xlabel("Number of Selected Features (|S|)")
    plt.ylabel("Test Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        plt.close()
    logger.info("Saved Pareto evolution plot: %s", path)
