"""Central benchmark configuration: experiment matrices, method presets, and budget estimates.

This is the single source of truth for:
- which datasets / classifiers / methods / seeds an experiment covers (``BenchmarkConfig``),
- default and paper-grade hyper-parameters per method,
- the nominal classifier-evaluation budget each method costs,
- alpha and sensitivity grids,
- the pre-run computational budget estimate (``--estimate``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from metaheuristic_xai.datasets import DATASET_REGISTRY

logger = logging.getLogger(__name__)


# ============================================================
# Experiment matrix
# ============================================================

@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a benchmark execution matrix."""

    datasets: list[str]
    classifiers: list[str]
    methods: list[str]
    seeds: list[int]
    method_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    alpha_values: list[float] = field(default_factory=lambda: [0.5, 0.7, 0.8, 0.9, 1.0])
    results_dir: Path = Path("results")
    output_prefix: str = "benchmark"
    search_val_size: float = 0.25  # inner validation fraction the search optimises against
    search_cv: int | None = None   # if set, the search scores masks by k-fold CV on the training partition


METHOD_FAMILY = {
    "shap": "xai",
    "lime": "xai",
    "lasso": "classical",
    "rfe": "classical",
    "boruta": "classical",
    "ga": "scalar_metaheuristic",
    "bpso": "scalar_metaheuristic",
    "sa": "scalar_metaheuristic",
    "gwo": "scalar_metaheuristic",
    "nsga2": "multiobjective",
    "full_features": "baseline",
}

MAIN_EVALUATION_BUDGET = 2000


# ============================================================
# Per-method hyper-parameters
# ============================================================

DEFAULT_METHOD_KWARGS: dict[str, dict[str, Any]] = {
    "ga": {"population_size": 50, "generations": 40},
    "bpso": {"swarm_size": 50, "iters": 40},
    "sa": {"initial_temp": 100.0, "cooling": 0.95, "max_steps": 2000},
    "gwo": {"population_size": 50, "iterations": 40},
    "nsga2": {"population_size": 50, "generations": 40},
    "shap": {"top_k": 10, "shap_max_samples": 200},
    "lime": {"top_k": 10, "num_samples": 100},
    "lasso": {},
    "rfe": {"n_features_to_select": 10},
    "boruta": {"max_iter": 100},
    "full_features": {},
}

# Lighter search budgets for laptops (roughly 4x fewer evaluations). Feature
# subsets are usually close to the full-budget ones on these datasets.
FAST_METHOD_KWARGS: dict[str, dict[str, Any]] = {
    **DEFAULT_METHOD_KWARGS,
    "ga": {"population_size": 25, "generations": 20},
    "bpso": {"swarm_size": 25, "iters": 20},
    "sa": {"initial_temp": 100.0, "cooling": 0.9, "max_steps": 500},
    "gwo": {"population_size": 25, "iterations": 20},
    "nsga2": {"population_size": 25, "generations": 20},
    "boruta": {"max_iter": 40},
}

# GA with the competent-selector operators (sparse init, uniform crossover,
# random immigrants) at the full evaluation budget. Pair with
# ``BenchmarkConfig(search_cv=5)`` for the k-fold search objective.
GA_IMPROVED_KWARGS: dict[str, Any] = {
    "population_size": 50,
    "generations": 40,
    "crossover_rate": 0.8,
    "mutation_rate": 0.2,
    "init_density": 0.1,
    "crossover": "uniform",
    "n_elites": 2,
    "immigrants": 5,
}


# ============================================================
# Nominal evaluation budget per method
# ============================================================

NOMINAL_EVALUATION_BUDGET: dict[str, Any] = {
    "full_features": 1,
    "shap": 1,
    "lime": 1,
    "lasso": 1,
    "rfe": 1,
    "boruta": 1,
    "ga": lambda cfg: cfg.get("population_size", 50) * cfg.get("generations", 100),
    "bpso": lambda cfg: cfg.get("swarm_size", 50) * cfg.get("iters", 100),
    "sa": lambda cfg: cfg.get("max_steps", 1500),
    "gwo": lambda cfg: cfg.get("population_size", 50) * cfg.get("iterations", 100),
    "nsga2": lambda cfg: cfg.get("population_size", 50) * cfg.get("generations", 100),
}


def get_nominal_budget(method_name: str, method_kwargs: dict[str, Any]) -> int:
    """Expected number of classifier evaluations for one run of ``method_name``."""
    budget = NOMINAL_EVALUATION_BUDGET.get(method_name, 1)
    if callable(budget):
        return int(budget(method_kwargs))
    return int(budget)


# ============================================================
# Default and paper-grade experiment matrices
# ============================================================

BENCHMARK_CONFIG = BenchmarkConfig(
    datasets=["wdbc", "ionosphere", "madelon", "colon", "leukemia"],
    classifiers=["random_forest", "xgboost", "lightgbm", "svm", "mlp"],
    methods=[
        "ga",
        "bpso",
        "sa",
        "gwo",
        "nsga2",
        "shap",
        "lime",
        "lasso",
        "rfe",
        "boruta",
        "full_features",
    ],
    seeds=[42, 43, 44],
    method_kwargs=DEFAULT_METHOD_KWARGS,
    alpha_values=[0.5, 0.7, 0.8, 0.9, 1.0],
    results_dir=Path("results"),
    output_prefix="benchmark",
)


PAPER_METHOD_KWARGS: dict[str, dict[str, Any]] = {
    "ga": {"population_size": 40, "generations": 40},
    "gwo": {"population_size": 40, "iterations": 40},
    "bpso": {"swarm_size": 40, "iters": 40},
    "nsga2": {"population_size": 40, "generations": 40},
    "boruta": {"max_iter": 100},
    "rfe": {"n_features_to_select": 10},
    "shap": {"top_k": 10, "shap_max_samples": 200},
    "full_features": {},
}

PAPER_RESULTS_DIR = Path("results") / "paper_matrix"

PAPER_CONFIG = BenchmarkConfig(
    datasets=["wdbc", "ionosphere", "colon"],
    classifiers=["random_forest", "xgboost", "lightgbm"],
    methods=["ga", "gwo", "bpso", "nsga2", "boruta", "rfe", "shap", "full_features"],
    seeds=list(range(42, 57)),
    method_kwargs=PAPER_METHOD_KWARGS,
    results_dir=PAPER_RESULTS_DIR,
    output_prefix="paper",
)


# ============================================================
# Sensitivity grids
# ============================================================

ALPHA_VALUES = [0.3, 0.5, 0.8]

SENSITIVITY_GRID = {
    "ga": {"population_size": [10, 20], "generations": [5, 10]},
    "bpso": {"swarm_size": [10, 20], "iters": [5, 10]},
    "sa": {"max_steps": [20, 50], "initial_temp": [50.0]},
}

EXPANDED_SENSITIVITY_GRID = {
    "ga": {"population_size": [20, 50], "generations": [10, 20], "crossover_rate": [0.7, 0.9]},
    "bpso": {"swarm_size": [20, 50], "iters": [10, 20]},
    "sa": {"max_steps": [100, 500], "initial_temp": [50.0, 100.0], "cooling": [0.9, 0.95]},
    "gwo": {"population_size": [20, 50], "iterations": [10, 20]},
}


# ============================================================
# Pre-run computational budget estimate
# ============================================================

@dataclass
class BudgetEstimate:
    """Estimated evaluations and resource costs for an experiment plan."""

    n_datasets: int
    n_classifiers: int
    n_methods: int
    n_seeds: int
    total_runs: int
    total_evaluations: int
    evaluations_per_method: dict[str, int]
    details_df: pd.DataFrame


def estimate_benchmark_budget(
    datasets: Sequence[str],
    classifiers: Sequence[str],
    methods: Sequence[str],
    seeds: Sequence[int],
    method_params: dict[str, dict[str, Any]] | None = None,
) -> BudgetEstimate:
    """Calculate the full benchmark computational budget estimate."""
    method_params = method_params or {}
    evals_per_method = {
        m: get_nominal_budget(m, method_params.get(m, {})) for m in methods
    }

    n_d, n_c, n_m, n_s = len(datasets), len(classifiers), len(methods), len(seeds)
    total_runs = n_d * n_c * n_m * n_s
    total_evaluations = n_d * n_c * n_s * sum(evals_per_method.values())

    rows: list[dict[str, Any]] = []
    for d in datasets:
        d_cfg = DATASET_REGISTRY.get(d)
        d_name = d_cfg.name if d_cfg else d
        scale_tier = d_cfg.scale_tier if d_cfg else "unknown"
        for c in classifiers:
            for m in methods:
                e_single = evals_per_method[m]
                rows.append(
                    {
                        "dataset": d_name,
                        "scale_tier": scale_tier,
                        "classifier": c,
                        "method": m,
                        "evals_per_run": e_single,
                        "n_seeds": n_s,
                        "total_evaluations": e_single * n_s,
                    }
                )

    return BudgetEstimate(
        n_datasets=n_d,
        n_classifiers=n_c,
        n_methods=n_m,
        n_seeds=n_s,
        total_runs=total_runs,
        total_evaluations=total_evaluations,
        evaluations_per_method=evals_per_method,
        details_df=pd.DataFrame(rows),
    )


def print_budget_report(estimate: BudgetEstimate) -> None:
    """Print a terminal report for the computational budget dry-run."""
    print("=" * 78)
    print(" EXPERIMENTAL COMPUTATIONAL BUDGET ESTIMATION (DRY-RUN)")
    print("=" * 78)
    print(f" Datasets   : {estimate.n_datasets}")
    print(f" Classifiers: {estimate.n_classifiers}")
    print(f" Methods    : {estimate.n_methods}")
    print(f" Seeds      : {estimate.n_seeds}")
    print(f" Total experiment runs      : {estimate.total_runs:,}")
    print(f" Total classifier evaluations: {estimate.total_evaluations:,}")
    print("-" * 78)
    print(" Evaluations per method (single run):")
    for m, e in estimate.evaluations_per_method.items():
        print(f"   - {m:<12}: {e:,}")
    print("=" * 78)


__all__ = [
    "ALPHA_VALUES",
    "BENCHMARK_CONFIG",
    "BudgetEstimate",
    "BenchmarkConfig",
    "DEFAULT_METHOD_KWARGS",
    "EXPANDED_SENSITIVITY_GRID",
    "FAST_METHOD_KWARGS",
    "MAIN_EVALUATION_BUDGET",
    "METHOD_FAMILY",
    "NOMINAL_EVALUATION_BUDGET",
    "PAPER_CONFIG",
    "PAPER_METHOD_KWARGS",
    "PAPER_RESULTS_DIR",
    "SENSITIVITY_GRID",
    "estimate_benchmark_budget",
    "get_nominal_budget",
    "print_budget_report",
]
