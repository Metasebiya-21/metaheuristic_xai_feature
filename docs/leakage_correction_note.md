# Technical note: data-leakage defect in the GA-XAI evaluation pipeline

**Status:** internal draft for authors / editor communication.
**Date:** 2026-09-10.
**Scope:** the wrapper feature-selection evaluation used for the paper's headline
comparison (Genetic Algorithm vs Tree-SHAP).

---

## 1. Summary

The evaluation pipeline scored every candidate feature subset produced by the
metaheuristic search (GA, BPSO, SA) **on the held-out test set**, and then reported
the accuracy of the final subset **on that same test set**. The search was
therefore optimising the exact quantity later reported as the result.

The Tree-SHAP baseline was **not** affected: SHAP ranks features from a model
trained on the training partition only and never queries the evaluation oracle
during selection.

Consequently the reported GA accuracy is inflated by test-set optimisation, while
the reported SHAP accuracy is a valid held-out estimate. After removing the leak,
the GA advantage disappears.

## 2. The defect

The relevant call path in the submitted/benchmark code (`src/main.py`,
`ga.py`, `oracle.py`, `metaheuristic_selectors.py`):

1. The evaluation oracle is constructed with the held-out test partition:

   ```python
   # src/main.py
   oracle = EvaluationOracle(
       classifier=clf_adapter,
       X_train=bundle.X_train, y_train=bundle.y_train,
       X_test=bundle.X_test,   y_test=bundle.y_test,   # <-- held-out test set
       ...
   )
   ```

2. `oracle.evaluate(mask)` fits a classifier on `X_train[:, mask]` and computes
   accuracy against `y_test`:

   ```python
   # oracle.py
   clf.fit(self.X_train[:, idx], self.y_train)
   y_pred = clf.predict(self.X_test[:, idx])
   acc = accuracy_score(self.y_test, y_pred)
   fitness = alpha * (1 - acc) + (1 - alpha) * (|mask| / d)
   ```

3. The GA minimises this `fitness` over the whole search (population x generations
   candidate masks). Because `acc` is test-set accuracy, the GA is directly
   maximising held-out test accuracy.

4. The final subset is then scored with the **same oracle / same test set** and
   that value is reported:

   ```python
   # src/main.py
   eval_res = selector.evaluate()          # oracle.evaluate(support) -> same X_test/y_test
   record.accuracy = eval_res.accuracy     # <-- the quantity the GA maximised
   ```

`SHAPSelector.fit` trains a Random Forest on `X_train` only, ranks features by mean
`|SHAP|`, and takes the top-*k*; it does not call the oracle. Its reported accuracy
is a genuine held-out estimate.

## 3. Isolated verification

`alpha = 0.9`, 10 stratified splits (seeds 42-51), GA (pop 50, 50 generations)
vs Tree-SHAP top-10, Random Forest (100 trees). The **only** difference between
the two GA conditions is whether the search is scored on the held-out test set
(original) or on a stratified inner-validation split carved from the training
partition (leakage-free); the final subset is scored once on the untouched test
set in both. Tree-SHAP never queries the oracle, so it is identical under both.

Function: `metaheuristic_xai.runner.run_leakage_verification`; raw output:
`results/leakage_verification.csv` (150 rows).

### 3.1 WDBC (the dataset behind the abstract's headline numbers)

| method | condition | accuracy (mean +/- sd, n=10) | # features |
|---|---|---|---|
| GA | **original** (search scored on test) | **0.9851 +/- 0.0102** | 6.2 |
| GA | **leakage-free** (search on inner val) | **0.9509 +/- 0.0212** | 5.3 |
| Tree-SHAP | (never uses the oracle) | 0.9474 +/- 0.0155 | 10 |

Abstract as submitted: GA **0.9851 +/- 0.0083** (6.6 +/- 1.6 features) vs SHAP
**0.9456 +/- 0.0123** (10 features). Both figures reproduce exactly under the leaky
pipeline. Removing the leak drops GA **0.9851 -> 0.9509** (-3.4 points) and
collapses the GA-over-SHAP gap from ~0.040 to **0.0035** (not significant).

### 3.2 All five datasets (10 splits each)

| dataset | GA original | GA leakage-free | inflation (pts) | Tree-SHAP | GA(free) - SHAP (pts) |
|---|---|---|---|---|---|
| colon | 0.9867 | 0.9667 | **+2.00** | 0.9333 | +3.3 (67 vs 10 features) |
| ionosphere | 0.9803 | 0.9085 | **+7.18** | 0.9211 | **-1.27** |
| leukemia | 0.9867 | 0.9733 | **+1.33** | 0.9400 | +3.3 (75 vs 10 features) |
| madelon | 0.8035 | 0.7587 | **+4.48** | 0.8646 | **-10.60** |
| wdbc | 0.9851 | 0.9509 | **+3.42** | 0.9474 | +0.35 |
| **pooled** | **0.9484** | **0.9116** | **+3.68** | **0.9213** | **-0.97** |

The leak inflates GA by **1.3 to 7.2 accuracy points on every dataset**. Once
removed:

- pooled, leakage-free GA (0.912) is **below** Tree-SHAP (0.921);
- GA loses outright on ionosphere and madelon;
- where it edges ahead (colon, leukemia) it uses ~70 features vs SHAP's 10, on
  n = 62 / 72 datasets where a 3-point difference is ~2 test samples.

Also: in the original setup GA's validation accuracy **equals** its test accuracy
by construction (they are the same held-out set), so the "no generalization gap"
in the abstract is an artifact of there being no separate validation partition.
The leakage-free GA shows real generalization gaps of up to +0.066.

The reported GA advantage over Tree-SHAP is an artifact of evaluating the search
on the test set, not a property of the method -- confirmed across all five
datasets, not just WDBC.

## 4. Corrected multi-dataset result

Leakage-free protocol (inner-validation search, single held-out test evaluation
identical for every method), 5 datasets x 5 classifiers x 3 seeds = 825 runs,
`alpha = 0.8`, ~500 search evaluations per metaheuristic:

- Friedman across 11 methods on test accuracy: chi-square = 95.9, p = 3.6e-16.
  Mean ranks (1 = best): **boruta 3.81**, rfe 4.99, full-feature set 5.23,
  lime 5.39, shap 5.85, lasso 6.43, bpso 6.48, gwo 6.53, nsga2 7.04, sa 7.11,
  **ga 7.15**.
- Holm-corrected paired Wilcoxon vs SHAP: only **Boruta** significantly beats SHAP
  (p = 1e-4). GA, BPSO, SA, GWO, NSGA-II do not; NSGA-II is significantly *worse*.
- vs the full feature set: GA, NSGA-II, SA and LASSO are **significantly worse than
  using every feature** (Holm p = 0.013 / 0.011 / 0.004 / 0.007) - their selection
  costs accuracy.
- Selection stability across seeds (Nogueira index, chance-corrected): GA 0.003,
  SA 0.012, NSGA-II 0.027 vs SHAP 0.49, RFE 0.47, Boruta 0.41 - the metaheuristic
  subsets are close to run-to-run random.
- Search overfitting (validation minus test accuracy): +0.06 to +0.11 on the
  ultra-high-dimensional / small-sample datasets for the metaheuristics; ~0 for
  SHAP / LASSO / RFE / Boruta.

### 4.1 Classifier-reliability check (found during review; does not change the result)

Section 4's numbers pool 5 classifiers (Random Forest, XGBoost, LightGBM, SVM,
MLP). A per-classifier breakdown, not previously checked, shows **MLP is an
outlier**:

| classifier | mean accuracy | mean generalization gap | runs at/near chance (<=0.6 acc) |
|---|---|---|---|
| lightgbm | 0.906 | 0.015 | 1.2% |
| xgboost | 0.904 | 0.011 | 1.2% |
| random_forest | 0.899 | 0.023 | 0.6% |
| svm | 0.885 | 0.032 | 11.5% |
| **mlp** | **0.805** | **0.041** | **17.0%** (63.6% on madelon, 15.2% on leukemia) |

Likely mechanism: `MLPAdapter` uses `early_stopping=True`, which carves an
additional internal validation split on top of the leakage-fix's own inner-val
split -- on n=57-72 datasets this leaves very few samples to fit; on madelon
(non-linear synthetic benchmark) plain MLP may not converge within `max_iter=500`
regardless of sample size.

**This does not change the conclusion.** Re-running the Friedman test with MLP
excluded (4 classifiers, n=60 subjects) gives the same ranking, GA/NSGA-II/SA
still last, Boruta still first:

| | with MLP (n=75) | without MLP (n=60) |
|---|---|---|
| chi-square, p | 95.9, 3.6e-16 | 82.0, 2.1e-13 |
| mean ranks | boruta 3.81 ... ga 7.15 | boruta 3.89 ... ga **7.30** |

The section 5 leakage verification (the paper's core finding) used Random Forest
only and is unaffected. Recommendation: report the classifier-reliability table
alongside the main result, and either exclude MLP or increase `max_iter` /
disable `early_stopping` in `MLPAdapter` before a final run, rather than silently
pooling an unconverged classifier into the headline numbers.

Two additional bugs identified and fixed during the correction (both had
disadvantaged the metaheuristics, and fixing them did **not** change the
conclusion):

- Simulated annealing terminated on the temperature schedule after ~100-220
  candidate evaluations regardless of the configured budget.
- The NSGA-II "knee" selection compared objectives on incompatible scales
  (error in [0, ~0.3] vs feature-ratio in [0, 1]), so it always returned the
  sparsest point on the Pareto front.

## 5. Impact on the paper's claims

| abstract claim | corrected finding |
|---|---|
| "the Genetic Algorithm ... outperform[s] the SHAP baseline" (0.9851 vs 0.9456) | does not hold; GA ties SHAP on WDBC and ranks last of 11 methods pooled |
| "using only 6.6 +/- 1.6 selected features" | subset size is a function of `alpha`; at the reported operating point GA does not beat SHAP |
| "statistically reliable" / "statistical robustness" | the GA feature selections are run-to-run unstable (Nogueira ~= 0) |
| "Wilcoxon rank-sum testing" | rank-sum is for unpaired samples; the splits are paired (signed-rank) |

The paper's central empirical claim is an artifact of the evaluation procedure and
is not reproducible under a leakage-free protocol.

## 6. Reproduction

- Corrected pipeline: `metaheuristic_xai.oracle.build_oracles` (search vs test
  oracle split), `metaheuristic_xai.runner.run_feature_selection`.
- Leakage regression test: `tests/test_leakage.py`
  (`test_build_oracles_search_never_sees_test`).
- Isolated verification: `verify_leakage.py` -> `results/verify_leakage_wdbc.csv`.
- Full corrected matrix: `metaheuristic_xai.runner.run_full_benchmark` with
  `BENCHMARK_CONFIG`; analysis via `metaheuristic_xai.evaluation`
  (`method_summary`, `friedman_nemenyi`, `holm_wilcoxon_vs`, `feature_stability`).

## 6. Can GA be fixed to beat SHAP? Outcome of the investigation

Four variants were tried, all leakage-free, culminating in a fully nested
disjoint-fold design (`compare_ga.py`, `metaheuristic_xai.datasets.load_dataset_folds`
-- 5 outer folds per dataset, every row in exactly one test fold; for
`ga_consensus`, 4 inner GA searches per outer fold using only that fold's own
training partition, never its test rows):

| method | colon | ionosphere | leukemia | wdbc | pooled | vs SHAP (pts) |
|---|---|---|---|---|---|---|
| shap | 0.958 | 0.935 | 0.973 | 0.946 | 0.953 | -- |
| **boruta** | 0.972 | 0.935 | 0.987 | 0.961 | **0.964** | **+1.09 (positive on all 4)** |
| ga_free | 0.944 | 0.915 | 0.959 | 0.951 | 0.942 | -1.08 |
| ga_k10 (matched cardinality) | 0.944 | 0.923 | 0.917 | 0.953 | 0.934 | -1.87 |
| ga_k10_cv (5-fold CV objective) | 0.904 | 0.923 | 0.931 | 0.963 | 0.930 | -2.28 |
| ga_k10_cv_shapseed (SHAP warm-start) | 0.931 | 0.920 | 0.904 | 0.970 | 0.931 | -2.15 |
| ga_consensus (stability selection) | 0.821 | 0.909 | 0.864 | 0.960 | 0.905 | **-4.78 (worst)** |

An earlier checkpoint on WDBC+ionosphere only (repeated-holdout "seeds", not
disjoint folds) suggested `ga_k10` slightly beat SHAP pooled (+0.22) and a
resampling-based consensus beat it by +1.19 after a leave-one-seed-out check.
Extending to colon and leukemia under the fully nested design **reverses this**:
`ga_consensus` collapses to the worst-performing method (colon -13.7 pts,
leukemia -11.0 pts vs SHAP). Mechanism: with ~50-58 training samples per outer
fold, 4 inner GA searches rarely agree; requiring >=3/4 agreement produces a
1-2 feature consensus mask (3/5 colon folds and 2/5 leukemia folds produced an
**empty** consensus entirely), versus Boruta's ~86 features on the same data.

**Conclusion: none of matched cardinality, a cross-validated search objective,
SHAP warm-starting, or stability-selection consensus makes GA competitive with
SHAP once small-sample/high-dimensional datasets are included.** The WDBC-only
positive signal does not generalise and should not be reported as a general
result. Boruta remains the only method in the study that beats SHAP on every
dataset, without requiring any of this additional machinery. Recommendation:
report this as a documented negative result and recommend Boruta rather than
continue engineering GA variants.
