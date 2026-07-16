#!/usr/bin/env python3
"""Fitness-weight ablation: GA with accuracy weight alpha in {0.7, 0.8, 0.9}."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("SKLEARN_N_JOBS", "1")

import pandas as pd

from src.data_loader import load_breast_cancer_data
from src.fitness import evaluate_mask
from src.ga import run_genetic_algorithm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ablation")

RESULTS = _PROJ / "results"
ALPHAS = (0.7, 0.8, 0.9)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-runs", type=int, default=10, help="Independent splits per alpha.")
    p.add_argument("--generations", type=int, default=100, help="GA generations.")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke: 3 splits, 20 GA generations.",
    )
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
            X_train, X_test, y_train, y_test, _, _ = load_breast_cancer_data(
                random_state=rs
            )
            g = run_genetic_algorithm(
                X_train,
                X_test,
                y_train,
                y_test,
                generations=gens,
                base_seed=rs,
                acc_weight=alpha,
            )
            f, acc, n_feat, _ = evaluate_mask(
                g["best_mask"],
                X_train,
                X_test,
                y_train,
                y_test,
                acc_weight=alpha,
            )
            rows.append(
                {
                    "alpha": alpha,
                    "run_id": r,
                    "accuracy": float(acc),
                    "n_features": int(n_feat),
                    "fitness": float(f),
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
    print("\nAblation summary (GA, mean ± std over splits):\n")
    print(summary.to_string(index=False))
    print(f"\nSaved: {out} and {summary_path}")


if __name__ == "__main__":
    main()
