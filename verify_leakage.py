"""
Isolated verification: WDBC only, alpha=0.9, 10 stratified splits, GA vs Tree-SHAP.
Only ONE thing differs between the two GA conditions -- whether the search is
scored on the held-out test set (original) or on an inner validation split (fixed).
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np, pandas as pd

from metaheuristic_xai.datasets import load_dataset
from metaheuristic_xai.classifiers import RandomForestAdapter
from metaheuristic_xai.oracle import EvaluationOracle, build_oracles
from metaheuristic_xai.algorithms.ga import run_genetic_algorithm
from metaheuristic_xai.selectors.classical import SHAPSelector

ALPHA = 0.9
GA_KW = dict(population_size=50, generations=50, crossover_rate=0.8, mutation_rate=0.2)
SEEDS = list(range(42, 52))          # 10 stratified splits

rows = []
t0 = time.time()
for seed in SEEDS:
    b = load_dataset("wdbc", random_state=seed)

    # ---- ORIGINAL (leaky): the search oracle IS the test set ----
    leaky = EvaluationOracle(
        RandomForestAdapter(n_estimators=100, random_state=seed),
        b.X_train, b.y_train, b.X_test, b.y_test, alpha=ALPHA, split_id=seed,
    )
    g = run_genetic_algorithm(leaky.X_train, leaky.X_test, leaky.y_train, leaky.y_test,
                              base_seed=seed, oracle=leaky, **GA_KW)
    r = leaky.evaluate(g["best_mask"])
    rows.append(dict(method="GA", condition="original (leaky)", seed=seed,
                     accuracy=r.accuracy, n_features=r.n_features))

    # ---- FIXED: search on inner val split of train, report once on held-out test ----
    search, test = build_oracles(
        RandomForestAdapter(n_estimators=100, random_state=seed),
        b.X_train, b.y_train, b.X_test, b.y_test, alpha=ALPHA, split_id=seed,
    )
    g2 = run_genetic_algorithm(search.X_train, search.X_test, search.y_train, search.y_test,
                               base_seed=seed, oracle=search, **GA_KW)
    r2 = test.evaluate(g2["best_mask"])
    rows.append(dict(method="GA", condition="leakage-free", seed=seed,
                     accuracy=r2.accuracy, n_features=r2.n_features))

    # ---- Tree-SHAP top-10 (never uses the oracle; identical under both) ----
    sh = SHAPSelector(top_k=10, random_state=seed)
    sh.fit(b.X_train, b.y_train)
    rs = test.evaluate(sh.get_support())
    rows.append(dict(method="Tree-SHAP", condition="(no oracle)", seed=seed,
                     accuracy=rs.accuracy, n_features=rs.n_features))
    print(f"seed {seed} done  ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/verify_leakage_wdbc.csv", index=False)
summary = df.groupby(["method", "condition"]).agg(
    accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
    n_features_mean=("n_features", "mean"), n_features_std=("n_features", "std"),
    n=("seed", "count"),
).round(4)
print("\n==================== VERIFICATION SUMMARY (WDBC, alpha=0.9, 10 splits) ====================")
print(summary.to_string())
print("\nabstract as submitted:  GA 0.9851 +/- 0.0083 (6.6 +/- 1.6 feat)   vs   SHAP 0.9456 +/- 0.0123 (10 feat)")
print("VERIFY_DONE", flush=True)
