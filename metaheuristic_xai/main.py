#!/usr/bin/env python3
"""
Benchmark: GA, binary PSO, and simulated annealing vs a SHAP baseline
on the Breast Cancer Wisconsin (scaled) dataset.

Reproducibility: each ``run i`` uses ``random_state=42 + i`` for the stratified
split, and metaheuristics are seeded with ``42 + i`` in their respective modules.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

# Allow ``python3 main.py`` without installing as a package
_PROJ = Path(__file__).resolve().parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

# Reduce SHAP/Numba chatter when workers spin up; headless-friendly Matplotlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# One tree-parallel job per RF fit avoids oversubscription and sklearn/joblib noise.
os.environ.setdefault("SKLEARN_N_JOBS", "1")
warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)
_mpl = _PROJ / ".mpl"
_mpl.mkdir(exist_ok=True, parents=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl))

import dataclasses

import pandas as pd

from src.baseline import run_shap_baseline
from src.data_loader import load_breast_cancer_data
from src.evaluation import (
    RunRecord,
    all_wilcoxon_vs_shap,
    build_summary,
    records_to_dataframe,
    save_results_csv,
    save_summary_table,
)
from src.fitness import evaluate_mask
from src.ga import run_genetic_algorithm
from src.pso import run_binary_pso
from src.sa import run_simulated_annealing
from src.utils import build_bar_series_from_summary, plot_fitness_convergence, plot_method_bars

logger = logging.getLogger("metaheuristic_xai")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

RESULTS: Path = _PROJ / "results"
PLOTS: Path = _PROJ / "plots"
TOP_K: int = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
    )
    p.add_argument(
        "--n-runs",
        type=int,
        default=30,
        help="Independent stratified holdouts; default 30 for publication stats.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Smoke test: 1 data split; GA/PSO use 5 gen/iters, SA 150 steps. "
            "Faster than the full 100×100 schedule; default run uses 30 splits × full schedule."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_runs = 1 if args.quick else int(args.n_runs)
    ga_gens: int = 5 if args.quick else 100
    pso_iters: int = 5 if args.quick else 100
    sa_max_steps: int | None = 150 if args.quick else None
    if n_runs < 1:
        raise SystemExit("n_runs must be >= 1")
    if args.quick:
        logger.info(
            "--quick: 1 data split, GA/PSO=%d steps, SA cap=%s.",
            ga_gens,
            sa_max_steps,
        )

    RESULTS.mkdir(exist_ok=True, parents=True)
    PLOTS.mkdir(exist_ok=True, parents=True)
    (RESULTS / ".gitkeep").write_text("", encoding="utf-8")
    (PLOTS / ".gitkeep").write_text("", encoding="utf-8")

    records: list[RunRecord] = []
    ga_hists: list[list[float]] = []
    pso_hists: list[list[float]] = []
    sa_hists: list[list[float]] = []

    for r in range(n_runs):
        try:
            rs = 42 + r
            X_train, X_test, y_train, y_test, f_names, _s = load_breast_cancer_data(
                test_size=0.2, random_state=rs
            )
        except (RuntimeError, ValueError) as e:
            logger.critical("Data load failed: %s", e)
            raise

        shap_again = run_shap_baseline(
            X_train,
            X_test,
            y_train,
            y_test,
            feature_names=f_names,
            top_k=TOP_K,
            shap_max_samples=200,
            output_dir=(PLOTS if r == 0 else None),
        )
        try:
            rec_s = RunRecord(
                r,
                "SHAP",
                float(shap_again["accuracy"]),
                int(shap_again["n_features"]),
                float(shap_again["fitness"]),
                float(shap_again["runtime"]),
                float(shap_again.get("shap_time", 0.0)),
            )
        except (KeyError, TypeError) as e:
            logger.critical("Bad SHAP result: %s", e)
            raise
        records.append(rec_s)
        try:
            g = run_genetic_algorithm(
                X_train, X_test, y_train, y_test,
                population_size=50,
                generations=ga_gens,
                base_seed=rs,
            )
            f_g, a_g, n_g, _ = evaluate_mask(
                g["best_mask"], X_train, X_test, y_train, y_test
            )
            records.append(
                RunRecord(r, "GA", float(a_g), int(n_g), float(f_g), float(g["runtime"]))
            )
            ga_hists.append([float(x) for x in g["convergence"]])  # type: ignore[union-attr, arg-type]
        except (ValueError, TypeError) as e:
            logger.critical("GA run %d failed: %s", r, e)
            raise

        try:
            p = run_binary_pso(
                X_train, X_test, y_train, y_test,
                swarm_size=50,
                iters=pso_iters,
                base_seed=rs,
            )
            f_p, a_p, n_p, _ = evaluate_mask(
                p["best_mask"], X_train, X_test, y_train, y_test
            )
            records.append(
                RunRecord(
                    r,
                    "PSO",
                    float(a_p),
                    int(n_p),
                    float(f_p),
                    float(p["runtime"]),
                )
            )
            pso_hists.append([float(x) for x in p["convergence"]])  # type: ignore[union-attr, arg-type]
        except (ValueError, TypeError) as e:
            logger.critical("PSO run %d failed: %s", r, e)
            raise

        try:
            s = run_simulated_annealing(
                X_train, X_test, y_train, y_test,
                base_seed=rs,
                max_steps=sa_max_steps,
            )
            f_s, a_s, n_s, _ = evaluate_mask(
                s["best_mask"], X_train, X_test, y_train, y_test
            )
            records.append(
                RunRecord(
                    r,
                    "SA",
                    float(a_s),
                    int(n_s),
                    float(f_s),
                    float(s["runtime"]),
                )
            )
            sa_hists.append([float(x) for x in s["convergence"]])  # type: ignore[union-attr, arg-type]
        except (ValueError, TypeError) as e:
            logger.critical("SA run %d failed: %s", r, e)
            raise
        if (r + 1) % max(1, n_runs // 3) == 0 or (r + 1) == n_runs:
            logger.info("Finished %d / %d independent data splits.", r + 1, n_runs)

    rec_dicts: list[dict] = [dataclasses.asdict(x) for x in records] if records else []
    save_results_csv(rec_dicts, RESULTS / "all_runs.csv")
    sm = build_summary(rec_dicts)
    save_summary_table(sm, RESULTS / "summary.csv")

    raw_df = records_to_dataframe(records)
    w_list = all_wilcoxon_vs_shap(raw_df, baseline_name="SHAP")
    w_rows: list[dict] = [
        {
            "comparison": w.comparison,
            "statistic": w.statistic,
            "pvalue": w.pvalue,
            "n_a": w.n_a,
            "n_b": w.n_b,
        }
        for w in w_list
    ]
    if w_rows:
        pd.DataFrame(w_rows).to_csv(RESULTS / "wilcoxon_vs_shap.csv", index=False)
        logger.info("Saved Wilcoxon (rank-sum) test results to results/wilcoxon_vs_shap.csv")

    plot_fitness_convergence(
        {
            "GA (mean)": ga_hists,
            "PSO (mean)": pso_hists,
            "SA (mean)": sa_hists,
        },
        PLOTS / "convergence_fitness.png",
        title="Best fitness: mean ± std over independent runs (lower = better)",
    )
    for col, fname, t, ylab in (
        (
            "accuracy_mean",
            "bar_accuracy.png",
            "Test accuracy (mean by method)",
            "mean test accuracy",
        ),
        (
            "n_features_mean",
            "bar_n_features.png",
            "Selected feature count (mean)",
            "mean |S| (features)",
        ),
        (
            "runtime_mean",
            "bar_runtime.png",
            "Wall-clock runtime per run (mean)",
            "mean runtime (s)",
        ),
    ):
        ser = build_bar_series_from_summary(sm, col)
        if ser:
            plot_method_bars(ser, ylab, PLOTS / fname, title=t)

    print(
        "\n" + "-" * 72
        + "\nSUMMARY (mean ± std) — n_runs="
        + str(n_runs)
        + " independent stratified splits (random_state=42+i)\n"
        + "-" * 72
    )
    print(sm.to_string(index=False))
    print("-" * 72)
    if w_list:
        print("Wilcoxon rank-sum (two-sided) on test accuracy — vs SHAP:\n")
        for w in w_list:
            line = f"  {w.comparison}:  statistic={w.statistic:.5g}  p={w.pvalue:.4g}  n={w.n_a} vs n={w.n_b}"
            print(line)
    else:
        print("No Wilcoxon table: run with --n-runs >=2 (default 30) to rank-sum test.")
    print("-" * 72)
    print(f"Artifacts:  {RESULTS!s}  and  {PLOTS!s}\n")
    logger.info("Benchmark complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130) from None