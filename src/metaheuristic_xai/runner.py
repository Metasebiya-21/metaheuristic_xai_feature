"""Benchmark and sensitivity runners.

One code path for a single dataset/classifier/method/seed evaluation
(:func:`run_feature_selection`) that both the CLI and the notebooks call, plus
matrix drivers (:func:`run_full_benchmark`, :func:`run_paper_matrix`) and
sensitivity sweeps.
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import pandas as pd

from metaheuristic_xai.classifiers import get_classifier_adapter
from metaheuristic_xai.config import (
    ALPHA_VALUES,
    EXPANDED_SENSITIVITY_GRID,
    METHOD_FAMILY,
    PAPER_CONFIG,
    SENSITIVITY_GRID,
    BenchmarkConfig,
    get_nominal_budget,
)
from metaheuristic_xai.datasets import DATASET_REGISTRY, load_dataset
from metaheuristic_xai.oracle import build_oracles
from metaheuristic_xai.selectors import get_feature_selector

SEARCH_VAL_SIZE = 0.25

# ============================================================
# Persistence helper
# ============================================================

def _safe_save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a DataFrame as parquet, falling back to CSV if parquet is unavailable."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path, index=False)
        return out_path
    except Exception as exc:  # pragma: no cover - fallback path
        fallback = out_path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        print(f"Parquet save failed ({type(exc).__name__}: {exc}); saved CSV instead: {fallback}")
        return fallback


def _expand_method_kwargs(
    method_name: str,
    user_kwargs: dict[str, Any] | None,
    defaults: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Merge default settings with any per-run overrides."""
    config = defaults.get(method_name, {}).copy()
    if user_kwargs:
        config.update(user_kwargs)
    return config


# ============================================================
# Single run
# ============================================================

def run_feature_selection(
    dataset_bundle: Any,
    classifier_name: str,
    method_name: str,
    seed: int = 42,
    alpha: float = 0.8,
    method_kwargs: dict[str, Any] | None = None,
    defaults: dict[str, dict[str, Any]] | None = None,
    val_size: float = SEARCH_VAL_SIZE,
) -> dict[str, Any]:
    """Evaluate one dataset/classifier/method/seed configuration and return a record.

    Leakage protocol: the feature-selection search optimises against ``search_oracle``
    (an inner validation split of the training set); the reported metrics come from
    a single ``test_oracle`` evaluation of the final mask on the held-out test set,
    identical for every method.
    """
    defaults = defaults or {}
    search_oracle, test_oracle = build_oracles(
        classifier=get_classifier_adapter(classifier_name, random_state=seed),
        X_train=dataset_bundle.X_train,
        y_train=dataset_bundle.y_train,
        X_test=dataset_bundle.X_test,
        y_test=dataset_bundle.y_test,
        alpha=alpha,
        dataset_name=dataset_bundle.name,
        split_id=seed,
        val_size=val_size,
        enable_cache=True,
    )
    search_features = int(dataset_bundle.X_train.shape[1])

    if method_name == "full_features":
        mask = np.ones(search_features, dtype=np.int8)
        conv = None
        pareto_front = None
        selector = None
    else:
        kws = _expand_method_kwargs(method_name, method_kwargs, defaults)
        selector = get_feature_selector(method_name, oracle=search_oracle, random_state=seed, **kws)
        selector.fit(dataset_bundle.X_train, dataset_bundle.y_train)
        mask = np.asarray(selector.get_support(), dtype=np.int8)
        conv = getattr(selector, "convergence_", None)
        pareto_front = getattr(selector, "pareto_front_", None)

    res = test_oracle.evaluate(mask)              # reported metrics: held-out test
    val_res = search_oracle.evaluate(mask)        # search/validation metrics
    selected_indices = np.where(np.asarray(mask) == 1)[0].astype(int).tolist()

    runtime_value = 0.0 if selector is None else float(getattr(selector, "runtime_", 0.0))
    nominal = get_nominal_budget(
        method_name, _expand_method_kwargs(method_name, method_kwargs, defaults)
    )
    search_evals = int(search_oracle.total_evaluations)

    record = {
        "dataset": dataset_bundle.name,
        "source": DATASET_REGISTRY[dataset_bundle.name].source,
        "scale_tier": dataset_bundle.config.scale_tier,
        "classifier": classifier_name,
        "method": method_name,
        "method_family": METHOD_FAMILY.get(method_name, "unknown"),
        "seed": seed,
        "alpha": alpha if method_name != "full_features" else None,
        "raw_features": int(dataset_bundle.raw_d),
        "preprocessed_features": int(dataset_bundle.preprocessed_d),
        "search_features": search_features,
        "selected_features": int(res.n_features),
        "selected_indices": selected_indices,
        "feature_ratio": float(res.feature_ratio),
        "accuracy": float(res.accuracy),
        "balanced_accuracy": float(res.balanced_accuracy),
        "precision": float(res.precision),
        "recall": float(res.recall),
        "f1": float(res.f1),
        "roc_auc": float(res.roc_auc),
        "pr_auc": float(res.pr_auc),
        "fitness": float(res.fitness),
        "val_accuracy": float(val_res.accuracy),
        "val_fitness": float(val_res.fitness),
        "generalization_gap": float(val_res.accuracy - res.accuracy),
        "runtime_seconds": runtime_value,
        "nominal_evaluation_budget": nominal,
        "actual_oracle_evaluations": search_evals,
        "cache_hits": int(search_oracle.cache_hits),
        "cache_hit_rate": (
            float(search_oracle.cache_hits / search_evals) if search_evals else 0.0
        ),
        "convergence_history": conv,
        "pareto_front_size": len(pareto_front) if pareto_front is not None else None,
        "pareto_hypervolume": (
            float(getattr(selector, "hypervolume_", np.nan)) if pareto_front else np.nan
        ),
        "pareto_front": [
            {
                "selected_features": int(sol.n_features),
                "feature_ratio": float(sol.feature_ratio),
                "accuracy": float(sol.accuracy),
                "fitness": float(getattr(sol, "fitness", np.nan)),
            }
            for sol in pareto_front
        ]
        if pareto_front is not None
        else None,
    }
    return record


# ============================================================
# Matrix drivers
# ============================================================

@lru_cache(maxsize=32)
def _load_dataset_cached(dataset_name: str, seed: int) -> Any:
    """Per-worker dataset cache so repeated (dataset, seed) tasks skip re-preprocessing."""
    return load_dataset(dataset_name, random_state=seed)


_TIER_WEIGHT = {"ultrahigh": 3, "medium": 2, "low": 1}


def _finalize(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-run frames and normalise list columns that round-tripped
    through parquet as ndarrays (keeps ``all_runs.csv`` readable)."""
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "selected_indices" in df.columns:
        df["selected_indices"] = df["selected_indices"].map(
            lambda v: list(v) if isinstance(v, np.ndarray) else v
        )
    return df


def load_results(results_dir: str | Path) -> pd.DataFrame:
    """Load *every* per-run ``*.parquet`` under ``results_dir`` into one DataFrame.

    Use this in analysis instead of reading ``all_runs.csv`` when a benchmark was
    run in several passes (different classifiers/datasets each time) -- it always
    reflects the full set of completed runs on disk, not just the last config.
    """
    out_dir = Path(results_dir)
    files = sorted(p for p in out_dir.glob("*.parquet") if p.is_file())
    return _finalize([pd.read_parquet(p) for p in files])


def _run_one(
    dataset_name: str,
    classifier_name: str,
    method_name: str,
    seed: int,
    method_kwargs: dict[str, Any] | None,
    defaults: dict[str, dict[str, Any]],
    output_path: Path,
    val_size: float,
) -> Path:
    """Run a single (dataset, classifier, method, seed) cell and write its parquet."""
    bundle = _load_dataset_cached(dataset_name, seed)
    record = run_feature_selection(
        bundle,
        classifier_name,
        method_name,
        seed=seed,
        alpha=0.8,
        method_kwargs=method_kwargs,
        defaults=defaults,
        val_size=val_size,
    )
    _safe_save_dataframe(pd.DataFrame([record]), output_path)
    return output_path


def run_full_benchmark(
    config: BenchmarkConfig,
    *,
    results_dir: Path | None = None,
    resume: bool = True,
    max_runtime_seconds: float | None = None,
    defaults: dict[str, dict[str, Any]] | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run the benchmark matrix over dataset/classifier/method/seed, saving per-run artifacts.

    ``n_jobs != 1`` fans every individual run out across worker processes (heaviest
    first, each classifier still single-threaded to avoid oversubscription); pass
    ``-1`` for all cores. ``n_jobs=1`` keeps the serial path, which is the only one
    that honours ``max_runtime_seconds``.
    """
    out_dir = Path(results_dir) if results_dir is not None else config.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_defaults = defaults or config.method_kwargs
    val_size = getattr(config, "search_val_size", SEARCH_VAL_SIZE)

    def _result(frames: list[pd.DataFrame]) -> pd.DataFrame:
        # resume=True: return everything on disk so earlier passes accumulate;
        # resume=False: return only this config's runs.
        return load_results(out_dir) if resume else _finalize(frames)

    if n_jobs != 1:
        from joblib import Parallel, delayed

        if max_runtime_seconds is not None:
            print("run_full_benchmark: max_runtime_seconds is ignored when n_jobs != 1.")

        tasks = []
        for d_name in config.datasets:
            tier = _TIER_WEIGHT.get(DATASET_REGISTRY[d_name].scale_tier, 1)
            for c_name in config.classifiers:
                for m_name in config.methods:
                    weight = tier * max(
                        1, get_nominal_budget(m_name, run_defaults.get(m_name, {}))
                    )
                    for seed in config.seeds:
                        path = out_dir / f"{d_name}_{c_name}_{m_name}_seed{seed}.parquet"
                        tasks.append((weight, d_name, c_name, m_name, seed, path))
        tasks.sort(key=lambda t: t[0], reverse=True)  # slowest cells dispatched first

        pending = [t for t in tasks if not (resume and t[5].exists())]
        Parallel(n_jobs=n_jobs)(
            delayed(_run_one)(
                d_name, c_name, m_name, seed,
                config.method_kwargs.get(m_name), run_defaults, path, val_size,
            )
            for _, d_name, c_name, m_name, seed, path in pending
        )
        return _result([pd.read_parquet(t[5]) for t in tasks if t[5].exists()])

    start_time = time()
    frames = []
    for dataset_name in config.datasets:
        for seed in config.seeds:
            bundle = None
            for classifier_name in config.classifiers:
                for method_name in config.methods:
                    if (
                        max_runtime_seconds is not None
                        and time() - start_time >= max_runtime_seconds
                    ):
                        return _result(frames)

                    output_path = (
                        out_dir / f"{dataset_name}_{classifier_name}_{method_name}_seed{seed}.parquet"
                    )
                    if resume and output_path.exists():
                        frames.append(pd.read_parquet(output_path))
                        continue

                    if bundle is None:
                        bundle = load_dataset(dataset_name, random_state=seed)
                    record = run_feature_selection(
                        bundle,
                        classifier_name,
                        method_name,
                        seed=seed,
                        alpha=0.8,
                        method_kwargs=config.method_kwargs.get(method_name),
                        defaults=run_defaults,
                        val_size=val_size,
                    )
                    _safe_save_dataframe(pd.DataFrame([record]), output_path)
                    frames.append(pd.DataFrame([record]))

    return _result(frames)


def run_paper_matrix(
    *,
    resume: bool = True,
    max_runtime_seconds: float | None = 120 * 60,
    results_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the paper-grade benchmark matrix (see ``config.PAPER_CONFIG``)."""
    return run_full_benchmark(
        PAPER_CONFIG,
        results_dir=results_dir or PAPER_CONFIG.results_dir,
        resume=resume,
        max_runtime_seconds=max_runtime_seconds,
    )


# ============================================================
# Sensitivity sweeps
# ============================================================

def run_alpha_sensitivity(
    dataset_names: list[str] | None = None,
    alpha_values: list[float] | None = None,
    method_name: str = "ga",
    classifier_name: str = "random_forest",
    seeds: list[int] | None = None,
    method_kwargs: dict[str, Any] | None = None,
    results_dir: str | Path = "results/alpha_sensitivity",
) -> pd.DataFrame:
    """Sweep the fitness accuracy weight ``alpha`` for one method/classifier pair."""
    dataset_list = dataset_names or ["wdbc", "ionosphere"]
    alpha_list = alpha_values or ALPHA_VALUES
    seed_list = seeds or [0, 1, 2]
    rows: list[dict[str, Any]] = []

    for dataset_name in dataset_list:
        for seed in seed_list:
            bundle = load_dataset(dataset_name, random_state=seed)
            for alpha in alpha_list:
                result = run_feature_selection(
                    bundle,
                    classifier_name,
                    method_name,
                    seed=seed,
                    alpha=alpha,
                    method_kwargs=method_kwargs or {},
                    defaults={method_name: method_kwargs or {}},
                )
                result["alpha"] = alpha
                rows.append(result)

    output_df = pd.DataFrame(rows)
    _safe_save_dataframe(output_df, Path(results_dir) / "alpha_sensitivity.parquet")
    return output_df


def _run_grid_sensitivity(
    method_name: str,
    param_grid: dict[str, list[Any]],
    dataset_name: str,
    classifier_name: str,
    seed_list: list[int],
    alpha_list: list[float],
    out_path: Path,
) -> pd.DataFrame:
    keys = list(param_grid.keys())
    rows: list[dict[str, Any]] = []
    for combo in itertools.product(*[param_grid[k] for k in keys]):
        method_kwargs = dict(zip(keys, combo))
        for alpha in alpha_list:
            for seed in seed_list:
                bundle = load_dataset(dataset_name, random_state=seed)
                result = run_feature_selection(
                    bundle,
                    classifier_name,
                    method_name,
                    seed=seed,
                    alpha=alpha,
                    method_kwargs=method_kwargs,
                    defaults={method_name: method_kwargs},
                )
                result["method_kwargs"] = method_kwargs
                result["alpha"] = alpha
                rows.append(result)

    output_df = pd.DataFrame(rows)
    _safe_save_dataframe(output_df, out_path)
    return output_df


def run_hyperparameter_sensitivity(
    method_name: str,
    *,
    dataset_name: str = "wdbc",
    classifier_name: str = "random_forest",
    seeds: list[int] | None = None,
    param_grid: dict[str, list[Any]] | None = None,
    alpha_values: list[float] | None = None,
    results_dir: str | Path = "results/sensitivity",
) -> pd.DataFrame:
    """Run a hyper-parameter grid sweep for ``method_name`` and save the results."""
    return _run_grid_sensitivity(
        method_name,
        param_grid or SENSITIVITY_GRID.get(method_name, {}),
        dataset_name,
        classifier_name,
        seeds or [42],
        alpha_values or [0.5, 0.8],
        Path(results_dir) / f"{method_name}_sensitivity.parquet",
    )


def run_expanded_sensitivity(
    *,
    dataset_name: str = "wdbc",
    classifier_name: str = "random_forest",
    method_name: str = "ga",
    alpha_values: list[float] | None = None,
    param_grid: dict[str, list[Any]] | None = None,
    seeds: list[int] | None = None,
    results_dir: str | Path = "results/expanded_sensitivity",
) -> pd.DataFrame:
    """Run the wider exploratory sensitivity sweep."""
    return _run_grid_sensitivity(
        method_name,
        param_grid or EXPANDED_SENSITIVITY_GRID.get(method_name, {}),
        dataset_name,
        classifier_name,
        seeds or [42, 43, 44],
        alpha_values or [0.3, 0.5, 0.8],
        Path(results_dir) / f"{method_name}_expanded_sensitivity.parquet",
    )


# ============================================================
# Data-leakage verification
# ============================================================

def _leakage_cell(
    dataset: str, seed: int, alpha: float, ga_kwargs: dict[str, Any], n_estimators: int
) -> list[dict[str, Any]]:
    """One (dataset, seed): GA scored on the test set (original) vs GA scored on an
    inner validation split (leakage-free), plus Tree-SHAP top-10."""
    from metaheuristic_xai.algorithms.ga import run_genetic_algorithm
    from metaheuristic_xai.classifiers import RandomForestAdapter
    from metaheuristic_xai.oracle import EvaluationOracle
    from metaheuristic_xai.selectors.classical import SHAPSelector

    b = _load_dataset_cached(dataset, seed)
    rows: list[dict[str, Any]] = []

    def row(method: str, condition: str, mask: Any, test_oracle: Any, val_oracle: Any) -> None:
        m = np.asarray(mask, dtype=np.int8)
        rt, rv = test_oracle.evaluate(m), val_oracle.evaluate(m)
        rows.append({
            "dataset": dataset, "seed": seed, "method": method, "condition": condition,
            "test_accuracy": float(rt.accuracy), "val_accuracy": float(rv.accuracy),
            "generalization_gap": float(rv.accuracy - rt.accuracy),
            "n_features": int(rt.n_features),
        })

    # ORIGINAL: the search oracle IS the held-out test set
    leaky = EvaluationOracle(
        RandomForestAdapter(n_estimators=n_estimators, random_state=seed),
        b.X_train, b.y_train, b.X_test, b.y_test, alpha=alpha, split_id=seed,
    )
    g = run_genetic_algorithm(
        leaky.X_train, leaky.X_test, leaky.y_train, leaky.y_test,
        base_seed=seed, oracle=leaky, **ga_kwargs,
    )
    row("GA", "original (search on test)", g["best_mask"], leaky, leaky)

    # LEAKAGE-FREE: search on an inner validation split; final subset scored once on test
    search, test = build_oracles(
        RandomForestAdapter(n_estimators=n_estimators, random_state=seed),
        b.X_train, b.y_train, b.X_test, b.y_test, alpha=alpha, split_id=seed,
    )
    g2 = run_genetic_algorithm(
        search.X_train, search.X_test, search.y_train, search.y_test,
        base_seed=seed, oracle=search, **ga_kwargs,
    )
    row("GA", "leakage-free", g2["best_mask"], test, search)

    sh = SHAPSelector(top_k=10, random_state=seed)
    sh.fit(b.X_train, b.y_train)
    row("Tree-SHAP", "(no oracle)", sh.get_support(), test, search)
    return rows


def run_leakage_verification(
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
    *,
    alpha: float = 0.9,
    ga_kwargs: dict[str, Any] | None = None,
    n_estimators: int = 100,
    n_jobs: int = 1,
    results_path: str | Path | None = "results/leakage_verification.csv",
) -> pd.DataFrame:
    """Reproduce the wrapper evaluation with the search scored on the held-out test
    set (the original defect) vs on an inner validation split, side by side.

    Returns tidy rows (dataset, seed, method, condition, test_accuracy,
    val_accuracy, generalization_gap, n_features). GA hyper-parameters default to
    pop 50 / 50 generations / alpha 0.9 to match the original setup.
    """
    ds = datasets or list(DATASET_REGISTRY.keys())
    sd = seeds or list(range(42, 52))
    kw = {"population_size": 50, "generations": 50, "crossover_rate": 0.8, "mutation_rate": 0.2}
    if ga_kwargs:
        kw.update(ga_kwargs)

    tasks = [(d, s) for d in ds for s in sd]
    if n_jobs != 1:
        from joblib import Parallel, delayed

        out = Parallel(n_jobs=n_jobs)(
            delayed(_leakage_cell)(d, s, alpha, kw, n_estimators) for d, s in tasks
        )
    else:
        out = [_leakage_cell(d, s, alpha, kw, n_estimators) for d, s in tasks]

    df = pd.DataFrame([r for sub in out for r in sub])
    if results_path is not None:
        p = Path(results_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
    return df


__all__ = [
    "load_results",
    "run_alpha_sensitivity",
    "run_expanded_sensitivity",
    "run_feature_selection",
    "run_full_benchmark",
    "run_hyperparameter_sensitivity",
    "run_leakage_verification",
    "run_paper_matrix",
]
