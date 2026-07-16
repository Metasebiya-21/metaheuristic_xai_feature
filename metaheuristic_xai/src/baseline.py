"""
SHAP-based feature selection baseline using a trained random forest and Tree Explainer.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import shap
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.fitness import (
    RF_N_ESTIMATORS,
    RF_SEED,
    evaluate_mask,
    get_rf_n_jobs,
)

logger = logging.getLogger(__name__)


def run_shap_baseline(
    X_train: NDArray[np.floating],
    X_test: NDArray[np.floating],
    y_train: NDArray,
    y_test: NDArray,
    feature_names: Optional[NDArray] = None,
    top_k: int = 10,
    shap_max_samples: int = 200,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Select the top-``k`` features by mean absolute SHAP value (class 1 if binary list).

    Trains a ``RandomForestClassifier`` (``random_state=42``), computes SHAP on a
    stratified subset of the training set for speed, and builds a binary mask with
    exactly ``top_k`` ones. Evaluates with ``evaluate_mask`` for a consistent
    comparison with metaheuristics.

    Parameters
    ----------
    X_train, X_test, y_train, y_test
        Scaled data as used elsewhere in the project.
    feature_names
        Optional 1D array of names; used only for the importance plot labels.
    top_k
        Number of features to keep.
    shap_max_samples
        Subsample of training points for SHAP to limit runtime and memory.
    output_dir
        If set, a bar plot of the top selected features' mean |SHAP| is written under
        ``<output_dir>/shap_baseline_top_features.png`` (or similar).

    Returns
    -------
    dict
        Keys: ``"mask"``, ``"fitness"``, ``"accuracy"``, ``"n_features"``,
        ``"runtime"``, ``"importance"`` (full vector length ``d`` or empty),
        ``"shap_time"`` (time spent in SHAP only, approximate).
    """
    n_rows = int(X_train.shape[0])
    n_dim = int(X_train.shape[1])
    if top_k < 1 or top_k > n_dim:
        msg = f"top_k must be in [1, n_features], got {top_k} and n={n_dim}."
        raise ValueError(msg)

    t0 = time.perf_counter()
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RF_SEED,
        n_jobs=get_rf_n_jobs(),
    )
    try:
        rf.fit(X_train, y_train)
    except ValueError as e:
        logger.error("SHAP baseline: RandomForest training failed.")
        raise

    # Stratified small subsample for SHAP
    n_sub = min(n_rows, shap_max_samples)
    if n_sub < n_rows:
        try:
            X_sub, _, _, _ = train_test_split(
                X_train,
                y_train,
                train_size=n_sub,
                stratify=np.ravel(y_train),
                random_state=RF_SEED,
            )
        except (ValueError, TypeError) as e:
            logger.error("Subsample for SHAP failed: %s", e)
            raise
    else:
        X_sub = X_train

    t_shap0 = time.perf_counter()
    try:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sub)
    except (Exception,) as e:
        # TreeExplainer can miss edge cases; surface as a clear project error
        logger.exception("SHAP TreeExplainer failed.")
        raise RuntimeError("TreeExplainer.shap_values failed") from e

    t_shap1 = time.perf_counter()
    shap_time = t_shap1 - t_shap0

    importance = _abs_mean_shap_for_binary(shap_values, n_dim)

    # Top-k (largest first); tie order by index for determinism
    order = np.lexsort((np.arange(n_dim, dtype=np.int64), -importance))
    top_idx = order[:top_k].astype(np.int64, copy=False)

    mask = np.zeros(n_dim, dtype=np.int8)
    mask[top_idx] = 1

    fitness, acc, nfeats, _ = evaluate_mask(
        mask, X_train, X_test, y_train, y_test
    )
    t1 = time.perf_counter()
    runtime = t1 - t0

    if output_dir is not None:
        try:
            _plot_top_importance(
                importance,
                top_idx,
                feature_names,
                Path(output_dir) / "shap_baseline_top_features.png",
            )
        except (OSError, ValueError) as e:
            logger.warning("SHAP bar plot not saved: %s", e)

    return {
        "mask": mask,
        "fitness": float(fitness),
        "accuracy": acc,
        "n_features": int(nfeats),
        "runtime": float(runtime),
        "importance": importance,
        "shap_time": float(shap_time),
    }


def _abs_mean_shap_for_binary(
    shap_values: Any, n_dim: int
) -> NDArray[np.floating]:
    """Map SHAP return value to a 1D importance vector of length n_dim."""
    if isinstance(shap_values, list):
        m = np.asarray(
            shap_values[1] if len(shap_values) > 1 else shap_values[0],
            dtype=np.float64,
        )
    else:
        m = np.asarray(shap_values, dtype=np.float64)
    if m.ndim == 3 and m.shape[1] == n_dim and m.shape[2] >= 2:
        m = m[:, :, 1]
    if m.ndim == 2 and m.shape[1] == n_dim:
        return np.mean(np.abs(m), axis=0)
    msg = (
        f"Unexpected shap values shape, got {type(shap_values)} / "
        f"{getattr(m, 'shape', None)} with n_dim={n_dim}."
    )
    raise TypeError(msg)


def _plot_top_importance(
    full_importance: NDArray[np.floating],
    top_idx: NDArray[np.signedinteger],
    feature_names: Optional[NDArray],
    path: Path,
) -> None:
    """Bar chart: mean |SHAP| for selected top features."""
    import matplotlib

    matplotlib.use("Agg", force=True)  # headless
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    vals = full_importance[top_idx]
    if feature_names is not None and len(feature_names) == full_importance.size:
        names = [str(feature_names[i]) for i in top_idx]
    else:
        names = [f"feat_{i}" for i in top_idx]

    order = np.argsort(-vals)
    y_pos = np.arange(len(order), dtype=np.float32)
    plt.figure(figsize=(8, 0.35 * len(order) + 1.0))
    plt.barh(
        y_pos, vals[order], align="center", color="steelblue", ecolor="steelblue"
    )
    plt.yticks(
        y_pos, [names[i] for i in order], fontsize=8
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |SHAP| (training subsample)")
    plt.title("SHAP baseline: selected top features")
    plt.tight_layout()
    try:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        plt.close()
    logger.info("Wrote SHAP feature importance bar chart to %s", path)
