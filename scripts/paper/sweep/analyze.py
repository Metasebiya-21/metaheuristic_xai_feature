#!/usr/bin/env python3
"""Pre-specified analysis of the parameter sweep (written before the sweep finished).

(a) per method x factor level: mean test accuracy, |S|, generalization gap over 25 runs
    (5 datasets x 5 seeds); paired Wilcoxon vs the same method's baseline cells in
    results_5seed (Random Forest, same dataset/seed), Holm-corrected within each factor
    (across all method x level tests of that factor).
(b) vs SHAP and Boruta on the same 25 cells: paired differences and Wilcoxon p-values;
    flags: mean accuracy above SHAP's, or within 0.005 of Boruta's.
(c) Nogueira stability across the 5 seeds per (method, configuration, dataset), averaged over datasets.
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]  # repo root (was .parent.parent when this lived in results_sweep/)
sys.path.insert(0, str(ROOT / "src"))
from metaheuristic_xai.evaluation import _nogueira_stability  # noqa: E402

OUT = ROOT / "results_sweep"
LABEL = {"ga": "GA", "bpso": "BPSO", "sa": "SA", "gwo": "GWO", "nsga2": "NSGA-II"}
D = {"wdbc": 30, "ionosphere": 33, "colon": 300, "leukemia": 300, "madelon": 500}  # searched feature count
KEY = ["dataset", "seed"]


def parse_idx(x):
    if isinstance(x, (list, np.ndarray)):
        return sorted(int(i) for i in x)
    s = str(x)
    v = re.findall(r"np\.int64\((\d+)\)", s) or re.findall(r"(?<![\w.])(\d+)(?![\w.(])", s)
    return sorted(int(i) for i in v)


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (n - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def wil(x, y):
    d = np.asarray(x) - np.asarray(y)
    if np.all(np.abs(d) < 1e-12):
        return 1.0
    return float(wilcoxon(x, y).pvalue)


def main():
    files = sorted(Path(os.environ.get("SWEEP_RUNS_DIR", OUT / "runs")).glob("*.json"))
    S = pd.DataFrame([json.loads(f.read_text()) for f in files])
    S["idx"] = S.selected_indices.map(lambda s: json.loads(s))
    base = pd.read_csv(ROOT / "results_5seed/all_runs.csv")
    base = base[(base.classifier == "random_forest") & base.dataset.isin(D)].copy()
    base["idx"] = base.selected_indices.map(parse_idx)
    B = base[base.method.isin(LABEL)].rename(columns={"accuracy": "test_accuracy", "selected_features": "n_selected"})
    B = B.assign(factor="baseline", level="default")[
        ["method", "dataset", "seed", "factor", "level", "test_accuracy", "n_selected", "generalization_gap", "idx"]]
    ref = {m: base[base.method == m].set_index(KEY).sort_index() for m in ("shap", "boruta")}
    ALL = pd.concat([B, S[B.columns]], ignore_index=True)
    print("sweep runs:", len(S), "| per (method,factor,level):", S.groupby(["method", "factor", "level"]).size().unique())

    # (a)
    rows = []
    for (m, f, lvl), g in ALL.groupby(["method", "factor", "level"]):
        g = g.set_index(KEY).sort_index()
        b = B[B.method == m].set_index(KEY).sort_index().loc[g.index]
        r = dict(method=LABEL[m], factor=f, level=lvl, n=len(g), acc=g.test_accuracy.mean(), S=g.n_selected.mean(),
                 gap=g.generalization_gap.mean(), d_acc_vs_base=(g.test_accuracy - b.test_accuracy).mean(),
                 p_vs_base=np.nan if f == "baseline" else wil(g.test_accuracy, b.test_accuracy))
        for refname in ("shap", "boruta"):
            rr = ref[refname].loc[g.index]
            r[f"{refname}_acc"] = rr.accuracy.mean()
            r[f"d_vs_{refname}"] = (g.test_accuracy - rr.accuracy).mean()
            r[f"p_vs_{refname}"] = wil(g.test_accuracy, rr.accuracy)
        # (c) Nogueira per dataset, averaged
        st = [_nogueira_stability([set(x) for x in gg.idx], D[ds]) for ds, gg in g.reset_index().groupby("dataset")]
        r["nogueira"] = float(np.nanmean(st))
        rows.append(r)
    T = pd.DataFrame(rows)
    T["p_vs_base_holm"] = np.nan
    for f in T.factor.unique():
        if f == "baseline":
            continue
        k = T.factor == f
        T.loc[k, "p_vs_base_holm"] = holm(T.loc[k, "p_vs_base"].values)
    T["above_SHAP"] = T.acc > T.shap_acc
    T["within_0.005_of_Boruta"] = (T.boruta_acc - T.acc) <= 0.005
    order = {"baseline": 0, "F1_budget": 1, "F2_population": 2, "F3_alpha": 3, "F4_param": 4}
    T["_lv"] = T.level.map(lambda v: float(re.sub(r"^[a-z_]+=", "", v)) if re.sub(r"^[a-z_]+=", "", v).replace(".", "", 1).isdigit() else -1.0)
    T["_f"] = T.factor.map(order)
    T = T.sort_values(["method", "_f", "_lv"]).drop(columns=["_lv", "_f"])
    T.to_csv(os.environ.get("SWEEP_TABLE_OUT", OUT / "analysis_table.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(T.round(4).to_string(index=False))
    print("\nSHAP mean acc on these 25 cells (Random Forest, 5 datasets x 5 seeds):", round(ref["shap"].accuracy.mean(), 4),
          "| Boruta:", round(ref["boruta"].accuracy.mean(), 4))
    print("\nconfigs above SHAP:", T[T.above_SHAP][["method", "factor", "level", "acc", "d_vs_shap", "p_vs_shap"]].to_string(index=False))
    print("\nconfigs within 0.005 of Boruta:", T[T["within_0.005_of_Boruta"]][["method", "factor", "level", "acc", "d_vs_boruta", "p_vs_boruta"]].to_string(index=False))
    print("\nNogueira of SHAP/Boruta on the same cells:",
          {k: round(float(np.nanmean([_nogueira_stability([set(x) for x in g.idx], D[ds])
                                      for ds, g in base[base.method == k].groupby("dataset")])), 3)
           for k in ("shap", "boruta", "rfe", "lime", "lasso")})


if __name__ == "__main__":
    main()
