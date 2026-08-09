#!/usr/bin/env python3
"""Fitness-weight ablation: GA with accuracy weight alpha in {0.7, 0.8, 0.9}."""

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

from metaheuristic_xai.algorithms.ga import run_genetic_algorithm
from metaheuristic_xai.classifiers import RandomForestAdapter
from metaheuristic_xai.datasets import load_dataset
from metaheuristic_xai.oracle import build_oracles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ablation")

RESULTS = _SRC.parent / "results"
ALPHAS = (0.7, 0.8, 0.9)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-runs", type=int, default=10, help="Independent splits per alpha.")
    p.add_argument("--generations", type=int, default=100, help="GA generations.")
    p.add_argument("--quick", action="store_true", help="Smoke: 3 splits, 20 GA generations.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_runs = 3 if args.quick else int(args.n_runs)
    gens = 20 if args.quick else int(args.generations)
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for alpha in ALPHAS:
        for r in range(n_runs):
            rs = 42 + r
            bundle = load_dataset("wdbc", random_state=rs)
            search_oracle, test_oracle = build_oracles(
                classifier=RandomForestAdapter(random_state=rs),
                X_train=bundle.X_train,
                y_train=bundle.y_train,
                X_test=bundle.X_test,
                y_test=bundle.y_test,
                alpha=alpha,
                dataset_name="wdbc",
                split_id=rs,
            )
            g = run_genetic_algorithm(
                search_oracle.X_train,
                search_oracle.X_test,
                search_oracle.y_train,
                search_oracle.y_test,
                generations=gens,
                base_seed=rs,
                oracle=search_oracle,
            )
            res = test_oracle.evaluate(g["best_mask"])
            rows.append(
                {
                    "alpha": alpha,
                    "run_id": r,
                    "accuracy": float(res.accuracy),
                    "n_features": int(res.n_features),
                    "fitness": float(res.fitness),
                    "runtime": float(g["runtime"]),
                }
            )
        logger.info("Finished alpha=%.1f (%d splits).", alpha, n_runs)

    df = pd.DataFrame(rows)
    out = RESULTS / "ablation_alpha.csv"
    df.to_csv(out, index=False)

    summary = (
        df.groupby("alpha")
        .agg(
            n_runs=("run_id", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            n_features_mean=("n_features", "mean"),
            n_features_std=("n_features", "std"),
            fitness_mean=("fitness", "mean"),
            fitness_std=("fitness", "std"),
        )
        .reset_index()
    )
    summary_path = RESULTS / "ablation_alpha_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nAblation summary (GA, mean +/- std over splits):\n")
    print(summary.to_string(index=False))
    print(f"\nSaved: {out} and {summary_path}")


if __name__ == "__main__":
    main()
