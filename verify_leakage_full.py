"""
Leakage verification across ALL five datasets.

For every (dataset, seed): GA with the search scored on the held-out test set
(original bug) vs GA with the search scored on an inner-validation split
(leakage-free); plus Tree-SHAP top-10 (never uses the oracle). alpha = 0.9,
10 stratified splits, Random Forest (100 trees). Final subset scored once on the
untouched test set in every condition.
"""
from __future__ import annotations
import sys, time
sys.path.insert(0, "src")
import numpy as np
import pandas as pd

from metaheuristic_xai.datasets import load_dataset
from metaheuristic_xai.classifiers import RandomForestAdapter
from metaheuristic_xai.oracle import EvaluationOracle, build_oracles
from metaheuristic_xai.algorithms.ga import run_genetic_algorithm
from metaheuristic_xai.selectors.classical import SHAPSelector

ALPHA = 0.9
GA_KW = dict(population_size=50, generations=50, crossover_rate=0.8, mutation_rate=0.2)
DATASETS = ["wdbc", "ionosphere", "madelon", "colon", "leukemia"]
SEEDS = list(range(42, 52))


def one_cell(dataset: str, seed: int) -> list[dict]:
    b = load_dataset(dataset, random_state=seed)
    out = []

    leaky = EvaluationOracle(
        RandomForestAdapter(n_estimators=100, random_state=seed),
        b.X_train, b.y_train, b.X_test, b.y_test, alpha=ALPHA, split_id=seed,
    )
    g = run_genetic_algorithm(leaky.X_train, leaky.X_test, leaky.y_train, leaky.y_test,
                              base_seed=seed, oracle=leaky, **GA_KW)
    r = leaky.evaluate(g["best_mask"])
    out.append(dict(dataset=dataset, method="GA", condition="original (leaky)",
                    seed=seed, accuracy=r.accuracy, n_features=r.n_features))

    search, test = build_oracles(
        RandomForestAdapter(n_estimators=100, random_state=seed),
        b.X_train, b.y_train, b.X_test, b.y_test, alpha=ALPHA, split_id=seed,
    )
    g2 = run_genetic_algorithm(search.X_train, search.X_test, search.y_train, search.y_test,
                               base_seed=seed, oracle=search, **GA_KW)
    r2 = test.evaluate(g2["best_mask"])
    out.append(dict(dataset=dataset, method="GA", condition="leakage-free",
                    seed=seed, accuracy=r2.accuracy, n_features=r2.n_features))

    sh = SHAPSelector(top_k=10, random_state=seed)
    sh.fit(b.X_train, b.y_train)
    rs = test.evaluate(sh.get_support())
    out.append(dict(dataset=dataset, method="Tree-SHAP", condition="(no oracle)",
                    seed=seed, accuracy=rs.accuracy, n_features=rs.n_features))
    print(f"  {dataset:11s} seed {seed} done", flush=True)
    return out


if __name__ == "__main__":
    from joblib import Parallel, delayed
    t0 = time.time()
    tasks = [(d, s) for d in DATASETS for s in SEEDS]
    results = Parallel(n_jobs=6)(delayed(one_cell)(d, s) for d, s in tasks)
    rows = [r for sub in results for r in sub]
    df = pd.DataFrame(rows)
    df.to_csv("results/verify_leakage_full.csv", index=False)

    summ = df.groupby(["dataset", "method", "condition"]).agg(
        acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"),
        feat_mean=("n_features", "mean"),
    ).round(4)
    print("\n" + "=" * 90)
    print("LEAKAGE VERIFICATION  (all datasets, alpha=0.9, 10 splits, GA pop50/gen50)")
    print("=" * 90)
    print(summ.to_string())

    piv = df.pivot_table(index="dataset", columns=["method", "condition"], values="accuracy", aggfunc="mean").round(4)
    print("\n--- test accuracy: GA original vs GA leakage-free vs SHAP ---")
    print(piv.to_string())
    ga = df[df.method == "GA"].pivot_table(index="dataset", columns="condition", values="accuracy", aggfunc="mean")
    print("\n--- GA inflation from leakage (original - leakage-free), accuracy points ---")
    print(((ga["original (leaky)"] - ga["leakage-free"]) * 100).round(2).to_string())
    print(f"\nelapsed {(time.time()-t0)/60:.1f} min")
    print("VERIFY_FULL_DONE", flush=True)
