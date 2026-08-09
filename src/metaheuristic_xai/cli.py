"""Command-line entry point for the feature-selection benchmark.

Thin wrapper: parse flags -> build a :class:`BenchmarkConfig` -> hand off to
:func:`metaheuristic_xai.runner.run_full_benchmark` -> write summaries, paired
statistics, and the convergence plot.
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("SKLEARN_N_JOBS", "1")
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

from metaheuristic_xai.classifiers import CLASSIFIER_REGISTRY
from metaheuristic_xai.complexity import get_complexity_summary
from metaheuristic_xai.config import (
    BENCHMARK_CONFIG,
    DEFAULT_METHOD_KWARGS,
    BenchmarkConfig,
    estimate_benchmark_budget,
    print_budget_report,
)
from metaheuristic_xai.datasets import DATASET_REGISTRY, list_datasets
from metaheuristic_xai.evaluation import feature_stability, save_results, wilcoxon_paired_test
from metaheuristic_xai.plots import plot_fitness_convergence
from metaheuristic_xai.selectors import SELECTOR_REGISTRY

logger = logging.getLogger("metaheuristic_xai")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

_PROJ = Path(__file__).resolve().parents[1]

QUICK_METHOD_KWARGS: dict[str, dict[str, object]] = {
    "ga": {"population_size": 10, "generations": 5},
    "bpso": {"swarm_size": 10, "iters": 5},
    "sa": {"initial_temp": 10.0, "cooling": 0.9, "max_steps": 20},
    "gwo": {"population_size": 10, "iterations": 5},
    "nsga2": {"population_size": 10, "generations": 5},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature Selection Benchmark Framework CLI")
    p.add_argument("--list-datasets", action="store_true", help="List registered datasets and exit.")
    p.add_argument("--complexity", action="store_true", help="Show algorithm asymptotic complexity and exit.")
    p.add_argument("--estimate", "--estimate-only", action="store_true", help="Print the computational budget estimate and exit.")
    p.add_argument("--datasets", nargs="+", default=["all"], help="Datasets to include (or 'all').")
    p.add_argument("--classifiers", nargs="+", default=["random_forest"], help="Classifiers to evaluate (or 'all').")
    p.add_argument("--methods", nargs="+", default=["all"], help="Feature selection methods (or 'all').")
    p.add_argument("--seeds", type=int, default=5, help="Number of seeds (splits), starting at 42.")
    p.add_argument("--quick", action="store_true", help="Smoke test: 1 seed, short GA/PSO/SA/GWO/NSGA-II schedules.")
    p.add_argument("--jobs", type=int, default=1, help="Worker processes; every run is dispatched independently, heaviest first. -1 = all cores.")
    p.add_argument("--fresh", action="store_true", help="Recompute all runs instead of reusing per-run parquet files in --results-dir.")
    p.add_argument("--results-dir", type=str, default=str(_PROJ / "results"), help="Output directory for CSVs.")
    p.add_argument("--plots-dir", type=str, default=str(_PROJ / "plots"), help="Output directory for figures.")
    return p.parse_args()


def _resolve(selected: list[str], registry_keys: list[str], kind: str) -> list[str]:
    if "all" in selected:
        return list(registry_keys)
    unknown = [s for s in selected if s not in registry_keys]
    if unknown:
        raise ValueError(f"Unknown {kind}: {unknown}. Choose from {registry_keys}.")
    return selected


def main() -> None:
    args = parse_args()

    if args.list_datasets:
        print("REGISTERED DATASETS:")
        for d in list_datasets():
            print(f"  - {d['name']:<12} Tier: {d['scale_tier']:<10} Source: {d['source']}")
        return

    if args.complexity:
        print(get_complexity_summary())
        return

    datasets = _resolve(args.datasets, list(DATASET_REGISTRY.keys()), "dataset")
    classifiers = _resolve(args.classifiers, list(CLASSIFIER_REGISTRY.keys()), "classifier")
    # "full_features" is a baseline the runner handles directly, not a registered selector.
    methods = _resolve(args.methods, list(SELECTOR_REGISTRY.keys()) + ["full_features"], "method")
    seeds = [42 + i for i in range(1 if args.quick else args.seeds)]

    method_kwargs = dict(DEFAULT_METHOD_KWARGS)
    if args.quick:
        method_kwargs.update(QUICK_METHOD_KWARGS)

    estimate = estimate_benchmark_budget(datasets, classifiers, methods, seeds, method_kwargs)
    print_budget_report(estimate)
    if args.estimate:
        return

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    config = BenchmarkConfig(
        datasets=datasets,
        classifiers=classifiers,
        methods=methods,
        seeds=seeds,
        method_kwargs=method_kwargs,
        alpha_values=BENCHMARK_CONFIG.alpha_values,
        results_dir=results_dir,
    )

    from metaheuristic_xai.runner import run_full_benchmark

    df = run_full_benchmark(
        config, results_dir=results_dir, resume=not args.fresh, n_jobs=args.jobs
    )
    save_results(df, results_dir)

    if not df.empty and len(seeds) >= 2:
        stab = feature_stability(df)
        stab.to_csv(results_dir / "feature_stability.csv", index=False)
        print("\nSELECTION STABILITY ACROSS SEEDS:")
        print(stab.to_string(index=False))

    if not df.empty and len(seeds) >= 2 and "shap" in methods:
        _paired_tests_vs_shap(df, methods, results_dir)

    _convergence_plot(df, plots_dir / "convergence.png")

    logger.info("Benchmark complete. Artifacts in %s and %s", results_dir, plots_dir)


def _paired_tests_vs_shap(df: pd.DataFrame, methods: list[str], results_dir: Path) -> None:
    logger.info("Computing Wilcoxon signed-rank paired tests vs SHAP...")
    keys = ["dataset", "classifier", "seed"]
    base = df[df["method"] == "shap"].set_index(keys)["accuracy"]
    rows: list[dict[str, object]] = []
    for m in methods:
        if m == "shap":
            continue
        sub = df[df["method"] == m].set_index(keys)["accuracy"]
        paired = pd.concat({"m": sub, "base": base}, axis=1).dropna()
        if len(paired) < 2:
            continue
        res = wilcoxon_paired_test(
            paired["m"].to_numpy(), paired["base"].to_numpy(),
            name_method=m, name_baseline="shap",
        )
        if res:
            rows.append(asdict(res))
    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(results_dir / "wilcoxon_results.csv", index=False)
        print("\nWILCOXON SIGNED-RANK TEST RESULTS (vs SHAP):")
        print(out.to_string(index=False))


def _convergence_plot(df: pd.DataFrame, path: Path) -> None:
    if df.empty or "convergence_history" not in df.columns:
        return
    by_method: dict[str, list[list[float]]] = {}
    for method, hist in zip(df["method"], df["convergence_history"]):
        if isinstance(hist, (list, tuple)) and len(hist) > 0:
            by_method.setdefault(str(method), []).append([float(x) for x in hist])
    if by_method:
        plot_fitness_convergence(by_method, path)


if __name__ == "__main__":
    main()
