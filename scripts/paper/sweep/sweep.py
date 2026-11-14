#!/usr/bin/env python3
"""Parameter-robustness sweep for GA-XAI (reviewer request).

Reproduces the main benchmark's call path exactly (runner.run_feature_selection with
DEFAULT_METHOD_KWARGS, alpha 0.8, val_size 0.25, load_dataset(name, random_state=seed),
Random Forest) and changes ONE factor per configuration. Writes only to results_sweep/.

Modes:
  verify  - rerun the baseline for seed 42 on each dataset x 5 methods; compare to results_5seed
  probe   - time every configuration once (WDBC, seed 42)
  run     - the full design (3 datasets x 5 seeds x all configurations)
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[3]  # repo root (was .parent.parent when this lived in results_sweep/)
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

OUT = ROOT / "results_sweep"
METHODS = ["ga", "bpso", "sa", "gwo", "nsga2"]
DATASETS = ["wdbc", "ionosphere", "colon", "leukemia", "madelon"]  # all 5 (user request)
SEEDS = [42, 43, 44, 45, 46]  # full design chosen by the user (projected 8.4-9.9 h on 6 workers)
POP = 50

# F4: one method-specific hyperparameter (name, default); GWO has none.
F4_PARAM = {"ga": ("mutation_rate", 0.2), "bpso": ("w", 0.9), "sa": ("initial_temp", 100.0),
            "nsga2": ("mutation_rate", 0.05), "gwo": None}


def budget_kwargs(method: str, budget: int, pop: int = POP) -> dict:
    if method == "sa":
        return {"max_steps": budget}
    gens = budget // pop
    return {"ga": {"population_size": pop, "generations": gens},
            "bpso": {"swarm_size": pop, "iters": gens},
            "gwo": {"population_size": pop, "iterations": gens},
            "nsga2": {"population_size": pop, "generations": gens}}[method]


def configurations(method: str) -> list[tuple[str, str, dict, float, dict | None]]:
    """(factor, level, method_kwargs_override, alpha, bpso_options)."""
    c = [("baseline", "default", {}, 0.8, None)]
    for b in (1000, 5000, 10000):
        c.append(("F1_budget", str(b), budget_kwargs(method, b), 0.8, None))
    if method != "sa":
        for p in (25, 100):
            c.append(("F2_population", str(p), budget_kwargs(method, 2000, p), 0.8, None))
    for a in (0.7, 0.9):
        c.append(("F3_alpha", str(a), {}, a, None))
    if F4_PARAM[method]:
        name, default = F4_PARAM[method]
        for mult in (0.5, 2.0):
            val = default * mult
            if method == "bpso":
                c.append(("F4_param", f"w={val:g}", {}, 0.8, {"c1": 0.5, "c2": 0.3, "w": val, "p": 2}))
            else:
                c.append(("F4_param", f"{name}={val:g}", {name: val}, 0.8, None))
    return c


def run_cell(method, dataset, factor, level, override, alpha, bpso_options, seed):
    from metaheuristic_xai.config import DEFAULT_METHOD_KWARGS
    from metaheuristic_xai.datasets import load_dataset
    from metaheuristic_xai import runner

    bundle = load_dataset(dataset, random_state=seed)
    kw = {**DEFAULT_METHOD_KWARGS[method], **override}
    # BPSO c1/c2/w go through BPSOSelector's constructor arguments (k and p are fixed there);
    # they are kept out of `kw` so the recorded method_kwargs match the original sweep rows.
    pso = {k: bpso_options[k] for k in ("c1", "c2", "w")} if bpso_options is not None else {}
    t0 = time.perf_counter()
    rec = runner.run_feature_selection(bundle, "random_forest", method, seed=seed, alpha=alpha,
                                       method_kwargs={**kw, **pso}, defaults=DEFAULT_METHOD_KWARGS, val_size=0.25)
    wall = time.perf_counter() - t0
    conv = rec.get("convergence_history")
    return {"method": method, "dataset": dataset, "factor": factor, "level": level, "seed": seed,
            "alpha": alpha, "method_kwargs": json.dumps(kw), "bpso_options": json.dumps(bpso_options),
            "test_accuracy": float(rec["accuracy"]), "n_selected": int(rec["selected_features"]),
            "generalization_gap": float(rec["generalization_gap"]), "runtime_seconds": float(rec["runtime_seconds"]),
            "wall_seconds": wall, "selected_indices": json.dumps([int(i) for i in rec["selected_indices"]]),
            "final_test_fitness": float(rec["fitness"]), "final_val_fitness": float(rec["val_fitness"]),
            "nominal_budget": int(rec["nominal_evaluation_budget"]),
            "oracle_lookups": int(rec["actual_oracle_evaluations"]), "cache_hits": int(rec["cache_hits"]),
            "convergence_history": json.dumps([float(x) for x in conv]) if conv is not None and len(conv) else "[]"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["verify", "probe", "run"])
    ap.add_argument("--n-jobs", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.mode == "verify":
        tasks = [(m, d, "baseline", "default", {}, 0.8, None, 42) for d in DATASETS for m in METHODS]
    elif args.mode == "probe":
        tasks = [(m, "wdbc", f, l, o, a, b, 42) for m in METHODS for (f, l, o, a, b) in configurations(m)]
    else:
        tasks = [(m, d, f, l, o, a, b, s) for m in METHODS for (f, l, o, a, b) in configurations(m)
                 if f != "baseline" for d in DATASETS for s in SEEDS]
        # longest first (rough cost: dataset weight x budget), so the expensive runs do not form the tail
        dw = {"madelon": 15, "leukemia": 2, "colon": 2, "ionosphere": 1, "wdbc": 1}
        def cost(t):
            bud = int(t[3]) if t[2] == "F1_budget" else 2000
            return dw[t[1]] * bud
        tasks.sort(key=cost, reverse=True)
    t0 = time.perf_counter()
    if args.mode == "run":
        # one JSON per run, so a crash loses nothing and a restart resumes
        rdir = OUT / "runs"; rdir.mkdir(exist_ok=True)
        def key(t): return f"{t[0]}_{t[1]}_{t[2]}_{t[3]}_seed{t[7]}".replace("=", "-")
        def run_and_save(*t):
            f = rdir / f"{key(t)}.json"
            if f.exists():
                return json.loads(f.read_text())
            r = run_cell(*t); f.write_text(json.dumps(r)); return r
        rows = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(run_and_save)(*t) for t in tasks)
    else:
        rows = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(run_cell)(*t) for t in tasks)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{args.mode}_runs.csv", index=False)
    print(f"{args.mode}: {len(df)} runs in {time.perf_counter() - t0:.0f}s wall")


if __name__ == "__main__":
    main()
