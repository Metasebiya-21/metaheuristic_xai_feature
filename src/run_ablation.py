#!/usr/bin/env python3
"""Fitness-weight x search-budget ablation for GA (Random Forest, leakage-free oracles).

Sweeps alpha and the number of GA generations (budget = population x generations)
over several datasets and seeds, using the same build_oracles protocol as the
main benchmark: search on an inner validation split, one final test evaluation.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("SKLEARN_N_JOBS", "1")

import pandas as pd
from joblib import Parallel, delayed

from metaheuristic_xai.algorithms.ga import run_genetic_algorithm
from metaheuristic_xai.classifiers import RandomForestAdapter
from metaheuristic_xai.datasets import load_dataset
from metaheuristic_xai.oracle import build_oracles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ablation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["wdbc", "ionosphere", "colon"])
    p.add_argument("--alphas", nargs="+", type=float, default=[0.7, 0.8, 0.9])
    p.add_argument("--generations", nargs="+", type=int, default=[40],
                   help="GA generations to sweep; budget = population x generations.")
    p.add_argument("--population-size", type=int, default=50, help="Matches the main benchmark.")
    p.add_argument("--seeds", type=int, default=5, help="Splits 42 .. 42+N-1, as in the benchmark.")
    p.add_argument("--n-jobs", type=int, default=1, help="-1 = all cores.")
    p.add_argument("--out-dir", type=Path, default=_SRC.parent / "results" / "ablation")
    p.add_argument("--quick", action="store_true", help="Smoke: wdbc, 1 seed, 5 generations.")
    return p.parse_args()


def one_run(dataset: str, alpha: float, gens: int, pop: int, seed: int) -> dict:
    bundle = load_dataset(dataset, random_state=seed)
    search, test = build_oracles(
        classifier=RandomForestAdapter(random_state=seed),
        X_train=bundle.X_train, y_train=bundle.y_train,
        X_test=bundle.X_test, y_test=bundle.y_test,
        alpha=alpha, dataset_name=dataset, split_id=seed,
    )
    g = run_genetic_algorithm(
        search.X_train, search.X_test, search.y_train, search.y_test,
        population_size=pop, generations=gens, base_seed=seed, oracle=search,
    )
    res, val = test.evaluate(g["best_mask"]), search.evaluate(g["best_mask"])
    return {
        "dataset": dataset, "alpha": alpha, "generations": gens,
        "nominal_budget": pop * gens, "seed": seed,
        "accuracy": float(res.accuracy), "val_accuracy": float(val.accuracy),
        "generalization_gap": float(val.accuracy - res.accuracy),
        "n_features": int(res.n_features), "feature_ratio": float(res.feature_ratio),
        "fitness": float(res.fitness), "oracle_evaluations": int(search.total_evaluations),
        "runtime": float(g["runtime"]),
        "selected_indices": [int(i) for i in res.used_indices],
    }


def main() -> None:
    args = parse_args()
    if args.quick:
        args.datasets, args.generations, args.seeds = ["wdbc"], [5], 1
    seeds = [42 + i for i in range(args.seeds)]
    tasks = [(d, a, g, args.population_size, s)
             for d in args.datasets for a in args.alphas for g in args.generations for s in seeds]
    logger.info("Running %d GA runs (n_jobs=%d).", len(tasks), args.n_jobs)

    rows = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(one_run)(*t) for t in tasks)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "ablation_runs.csv", index=False)

    summary = (
        df.groupby(["dataset", "alpha", "generations"])
        .agg(n=("seed", "count"),
             accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
             n_features_mean=("n_features", "mean"), gen_gap_mean=("generalization_gap", "mean"),
             runtime_mean=("runtime", "mean"))
        .round(4).reset_index()
    )
    summary.to_csv(args.out_dir / "ablation_summary.csv", index=False)
    print("\nAblation summary (GA + Random Forest, mean over seeds):\n")
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.out_dir}/ablation_runs.csv and ablation_summary.csv")


if __name__ == "__main__":
    main()
