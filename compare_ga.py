"""
Can a *properly built* GA beat Tree-SHAP under a leakage-free evaluation?

Nested / disjoint-fold design (tightened after the first pass showed a subtle
cross-seed leak risk in a repeated-holdout consensus):

  - 5 OUTER folds per dataset via `load_dataset_folds` -- every row is in
    exactly one outer test fold, so outer folds never overlap (unlike repeated
    independent train_test_split "seeds").
  - `ga_free / ga_k10 / ga_k10_cv / ga_k10_cv_shapseed / shap / boruta`: one run
    per outer fold, scored on that fold's own held-out test partition.
  - `ga_consensus`: for EACH outer fold, run 4 INNER GA searches using only that
    fold's own training partition (never its test partition), take the features
    chosen by >= 3/4 inner runs, then score that per-fold consensus mask once on
    the fold's held-out test partition. No fold's consensus is ever informed by
    another fold's data, so there is no cross-fold leakage of any kind.
"""
from __future__ import annotations
import sys, time
sys.path.insert(0, "src")
import numpy as np
import pandas as pd

from metaheuristic_xai.datasets import load_dataset_folds
from metaheuristic_xai.classifiers import RandomForestAdapter
from metaheuristic_xai.oracle import build_oracles
from metaheuristic_xai.algorithms.ga import run_genetic_algorithm
from metaheuristic_xai.selectors.classical import SHAPSelector, BorutaSelector

DATASETS = ["wdbc", "ionosphere", "colon", "leukemia"]  # madelon run separately
K_OUTER = 5
M_INNER = 4
CONSENSUS_MIN = 3  # >= 3/4 inner runs
ALPHA = 0.8
K = 10
BUD = dict(population_size=35, generations=25, crossover_rate=0.8, mutation_rate=0.2)
IMPROVED = dict(crossover="uniform", n_elites=2, immigrants=5)


def _clf(seed):
    return RandomForestAdapter(n_estimators=100, random_state=seed)


def outer_fold(dataset: str, fold_id: int, bundle) -> list[dict]:
    rows = []

    def rec(method, mask, test_oracle, val_oracle):
        m = np.asarray(mask, np.int8)
        rt, rv = test_oracle.evaluate(m), val_oracle.evaluate(m)
        rows.append(dict(dataset=dataset, fold=fold_id, method=method,
                         test_acc=rt.accuracy, val_acc=rv.accuracy,
                         gap=rv.accuracy - rt.accuracy, n_features=int(rt.n_features),
                         selected=list(np.where(m == 1)[0])))

    s1, t1 = build_oracles(_clf(fold_id), bundle.X_train, bundle.y_train,
                           bundle.X_test, bundle.y_test, alpha=ALPHA, split_id=fold_id)
    s2, _ = build_oracles(_clf(fold_id), bundle.X_train, bundle.y_train,
                          bundle.X_test, bundle.y_test, alpha=ALPHA, split_id=fold_id,
                          cv=5, cv_repeats=1)

    g = run_genetic_algorithm(s1.X_train, s1.X_test, s1.y_train, s1.y_test,
                              base_seed=fold_id, oracle=s1, **BUD)
    rec("ga_free", g["best_mask"], t1, s1)

    g = run_genetic_algorithm(s1.X_train, s1.X_test, s1.y_train, s1.y_test,
                              base_seed=fold_id, oracle=s1, target_k=K, **BUD, **IMPROVED)
    rec("ga_k10", g["best_mask"], t1, s1)

    g = run_genetic_algorithm(s2.X_train, s2.X_test, s2.y_train, s2.y_test,
                              base_seed=fold_id, oracle=s2, target_k=K, **BUD, **IMPROVED)
    rec("ga_k10_cv", g["best_mask"], t1, s2)

    sh = SHAPSelector(top_k=K, random_state=fold_id)
    sh.fit(bundle.X_train, bundle.y_train)
    shap_mask = sh.get_support()
    rec("shap", shap_mask, t1, s1)

    g = run_genetic_algorithm(s2.X_train, s2.X_test, s2.y_train, s2.y_test,
                              base_seed=fold_id, oracle=s2, target_k=K,
                              seed_masks=[shap_mask], **BUD, **IMPROVED)
    rec("ga_k10_cv_shapseed", g["best_mask"], t1, s2)

    bo = BorutaSelector(max_iter=100, random_state=fold_id)
    bo.fit(bundle.X_train, bundle.y_train)
    rec("boruta", bo.get_support(), t1, s1)

    # --- ga_consensus: M_INNER searches using ONLY this fold's training partition ---
    cnt: dict[int, int] = {}
    for j in range(M_INNER):
        inner_search, _ = build_oracles(
            _clf(1000 * fold_id + j), bundle.X_train, bundle.y_train,
            bundle.X_train, bundle.y_train,  # dummy "test" -- never scored, inner-only
            alpha=ALPHA, split_id=1000 * fold_id + j, cv=5,
        )
        gi = run_genetic_algorithm(inner_search.X_train, inner_search.X_test,
                                   inner_search.y_train, inner_search.y_test,
                                   base_seed=1000 * fold_id + j, oracle=inner_search,
                                   target_k=K, **BUD, **IMPROVED)
        for f in np.where(np.asarray(gi["best_mask"]) == 1)[0]:
            cnt[int(f)] = cnt.get(int(f), 0) + 1
    consensus = sorted(f for f, c in cnt.items() if c >= CONSENSUS_MIN)
    d_ = bundle.X_train.shape[1]
    cmask = np.zeros(d_, np.int8)
    cmask[[f for f in consensus if f < d_]] = 1
    if cmask.sum() > 0:
        rec("ga_consensus", cmask, t1, s1)

    print(f"  {dataset:11s} fold {fold_id} done", flush=True)
    return rows


def cell(dataset: str, fold_id: int, bundle) -> list[dict]:
    return outer_fold(dataset, fold_id, bundle)


if __name__ == "__main__":
    from joblib import Parallel, delayed
    t0 = time.time()

    tasks = []
    for ds in DATASETS:
        folds = load_dataset_folds(ds, n_splits=K_OUTER, random_state=0)
        for i, b in enumerate(folds):
            tasks.append((ds, i, b))

    res = Parallel(n_jobs=6)(delayed(cell)(ds, i, b) for ds, i, b in tasks)
    df = pd.DataFrame([r for sub in res for r in sub])
    df.to_csv("results/compare_ga_nested.csv", index=False)

    pd.set_option("display.width", 200)
    order = ["ga_free", "ga_k10", "ga_k10_cv", "ga_k10_cv_shapseed", "ga_consensus", "shap", "boruta"]
    piv = df.pivot_table(index="method", columns="dataset", values="test_acc", aggfunc="mean").reindex(order).round(4)
    print("\n================ TEST ACCURACY (mean over 5 disjoint outer folds) ================")
    print(piv.to_string())
    ov = df.groupby("method").agg(test_acc=("test_acc", "mean"), test_sd=("test_acc", "std"),
                                  n_feat=("n_features", "mean"), gap=("gap", "mean")).reindex(order).round(4)
    print("\n================ POOLED ================"); print(ov.to_string())

    shap_by_ds = df[df.method == "shap"].groupby("dataset").test_acc.mean()
    print("\n================ (method - SHAP) test-accuracy points, per dataset ================")
    for m in order:
        if m == "shap":
            continue
        row = ((df[df.method == m].groupby("dataset").test_acc.mean() - shap_by_ds) * 100).round(2)
        print(f"{m:20s} " + "  ".join(f"{ds}:{row.get(ds, float('nan')):+.2f}" for ds in DATASETS)
              + f"   | pooled {(df[df.method==m].test_acc.mean()-df[df.method=='shap'].test_acc.mean())*100:+.2f}")

    def jacc(sets):
        s = [set(x) for x in sets]
        return float(np.mean([len(s[i] & s[j]) / max(1, len(s[i] | s[j]))
                              for i in range(len(s)) for j in range(i + 1, len(s))])) if len(s) > 1 else np.nan
    stab = df.groupby(["method", "dataset"]).selected.apply(lambda c: jacc(list(c))).groupby("method").mean().reindex(order).round(3)
    print("\n================ mean cross-fold Jaccard (selection stability) ================"); print(stab.to_string())
    print(f"\nelapsed {(time.time()-t0)/60:.1f} min")
    print("COMPARE_GA_NESTED_DONE", flush=True)
