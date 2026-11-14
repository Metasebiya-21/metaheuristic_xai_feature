# WDBC feature-agreement analysis (GA-XAI, reviewer request)

Generated 2026-11-11.

- `results/`, `results_5seed/`, the code and the manuscripts were only read, not modified.
- Everything was written to `results_wdbc_features/`. No benchmark cell was rerun.
- This is an agreement check against features named in prior literature. **It is not clinical validation.**

**Reference.** Wolberg, Street and Mangasarian (The Breast Journal 3(2):77–80, 1997) and the UCI WDBC documentation name the best-separating features as worst area, worst smoothness and mean texture.

## 1. Verification (step 1)

- The selected masks are **stored**. They are in the `selected_indices` column of `results_5seed/all_runs.csv`, written as `np.int64(k)` reprs and parsed by regex. Each per-cell parquet file (`results_5seed/wdbc_<clf>_<method>_seed<s>.parquet`) holds the same masks as an array.
- WDBC has **275 rows**: 11 methods × 5 classifiers × seeds 42–46, with 25 runs per method.
- **Verified: 275 / 275 rows match. 0 mismatches and 0 missing parquet files.** For every row:
  - CSV accuracy equals parquet accuracy (|diff| ≤ 1e-9);
  - `selected_features` is equal in both;
  - the length of the index list equals `selected_features`;
  - the index sets are identical in both.
- All indices fall in 0–29.
- **No rerun was needed**, so the 2-hour timing probe did not apply.

Logs: `step1_verify.txt`. Masks used: `wdbc_masks.pkl`.

## 2. Feature names and indices (step 0)

Names come from `load_dataset("wdbc", random_state=s)` and its `bundle.feature_names`. For all five seeds (42–46), the bundle has 30 features in exactly the order of `sklearn.datasets.load_breast_cancer().feature_names`, so preprocessing drops or reorders nothing. The full list of 30 is in `step0_names.txt`.

**Confirmed by name:**

| Feature | Index | Role |
|---|---|---|
| mean texture | **1** | reference |
| worst area | **23** | reference |
| worst smoothness | **24** | reference |
| worst radius | 20 | relaxed groups only |
| worst texture | 21 | relaxed groups only |
| worst perimeter | 22 | relaxed groups only |
| mean smoothness | 4 | relaxed groups only |

## 3. Metrics (step 2, fixed in advance)

**Definitions**
- **s:** mean subset size.
- **Strict:** the mean number of {mean texture, worst area, worst smoothness} in the subset.
- **Strict chance:** 3·s/30 per run, averaged over runs.
- **Relaxed:** the mean number of groups hit, out of three: size = {worst radius, worst perimeter, worst area}; texture = {mean texture, worst texture}; smoothness = {mean smoothness, worst smoothness}.
- **Relaxed chance:** a hypergeometric expectation, Σ_g [1 − C(30−g, s)/C(30, s)] per run, averaged.
- **CI:** a 95% percentile bootstrap over the 25 runs (10,000 resamples), for observed − expected.

### Main table

| Method | s | Strict obs. | Strict exp. | Strict Δ [95% CI] | All 3 | Relaxed obs. | Relaxed exp. | Relaxed Δ [95% CI] |
|---|---|---|---|---|---|---|---|---|
| GA | 5.04 | 0.60 | 0.50 | +0.10 [−0.17, 0.38] | 0% | 1.68 | 1.06 | **+0.63 [0.44, 0.80]** |
| BPSO | 7.48 | 1.32 | 0.75 | **+0.57 [0.20, 0.92]** | 4% | 1.96 | 1.47 | **+0.49 [0.16, 0.82]** |
| SA | 5.48 | 1.00 | 0.55 | **+0.45 [0.12, 0.77]** | 0% | 2.04 | 1.09 | **+0.95 [0.70, 1.20]** |
| GWO | 6.32 | 0.92 | 0.63 | +0.29 [−0.01, 0.61] | 4% | 2.12 | 1.28 | **+0.85 [0.61, 1.09]** |
| NSGA-II | 3.84 | 0.84 | 0.38 | **+0.46 [0.16, 0.77]** | 4% | 1.88 | 0.83 | **+1.05 [0.75, 1.35]** |
| SHAP | 10.00 | 1.00 | 1.00 | 0.00 [0.00, 0.00] | 0% | 1.00 | 1.85 | **−0.85 [−0.85, −0.85]** |
| LIME | 10.00 | 1.80 | 1.00 | **+0.80 [0.64, 0.96]** | 0% | 2.00 | 1.85 | **+0.15 [0.15, 0.15]** |
| RFE | 10.00 | 1.20 | 1.00 | **+0.20 [0.04, 0.36]** | 0% | 2.00 | 1.85 | **+0.15 [0.15, 0.15]** |
| LASSO | 17.40 | 2.20 | 1.74 | **+0.46 [0.29, 0.62]** | 20% | 3.00 | 2.53 | **+0.47 [0.34, 0.60]** |
| Boruta | 25.00 | 3.00 | 2.50 | **+0.50 [0.44, 0.56]** | 100% | 3.00 | 2.95 | **+0.05 [0.04, 0.07]** |
| Full feature set | 30 | 3.00 (trivial) | 3.00 | 0 | 100% | 3.00 | 3.00 | 0 |

Bold marks an interval that excludes 0. **Full feature set is trivially 3/3 and is excluded from every ranking statement below.**

### Inclusion rate of each reference feature

| Method | mean texture | worst area | worst smoothness |
|---|---|---|---|
| GA | 0.24 | 0.20 | 0.16 |
| BPSO | 0.64 | 0.36 | 0.32 |
| SA | 0.48 | 0.40 | 0.12 |
| GWO | 0.40 | 0.24 | 0.28 |
| NSGA-II | 0.36 | 0.16 | 0.32 |
| SHAP | **0.00** | 1.00 | **0.00** |
| LIME | 0.80 | 1.00 | 0.00 |
| RFE | 0.20 | 1.00 | 0.00 |
| LASSO | 0.60 | 0.60 | 1.00 |
| Boruta | 1.00 | 1.00 | 1.00 |

### Relaxed group hit rates

| Method | size | texture | smoothness |
|---|---|---|---|
| GA | 0.52 | 0.92 | 0.24 |
| BPSO | 0.48 | 0.92 | 0.56 |
| SA | 0.80 | 0.92 | 0.32 |
| GWO | 0.64 | 1.00 | 0.48 |
| NSGA-II | 0.60 | 0.88 | 0.40 |
| SHAP | 1.00 | **0.00** | **0.00** |
| LIME | 1.00 | 1.00 | 0.00 |
| RFE | 1.00 | 1.00 | 0.00 |
| LASSO | 1.00 | 1.00 | 1.00 |
| Boruta | 1.00 | 1.00 | 1.00 |

Full numbers: `metrics_by_method.csv` and `step2_metrics.txt`.

### Important caveat on the confidence intervals

SHAP, LIME, RFE, LASSO and Boruta fit their own internal models, so they produce **one identical mask per seed for all 5 evaluation classifiers** (checked: 1 distinct mask per method–seed). For them, the 25 "runs" are 5 independent masks repeated five times. The pre-specified run-level bootstrap therefore understates their uncertainty. The zero-width intervals for SHAP, LIME and RFE show this: every run has the same counts.

As a **supplementary** analysis (added after seeing this structure, and labelled as such), the bootstrap was repeated by resampling the 5 seeds (file `metrics_seed_cluster_ci.csv`). Two conclusions change:
- **Strict:** the intervals for SA [−0.08, 0.90] and RFE [0.00, 0.60] now touch or include 0.
- **Relaxed:** BPSO's interval [−0.03, 1.01] now includes 0.

Every other "above chance" or "below chance" call below is unchanged.

## 4. Top-10 most frequently selected features (inclusion rate over 25 runs)

| Rank | GA | BPSO | Boruta | SHAP | RFE |
|---|---|---|---|---|---|
| 1 | worst texture 0.72 | mean texture 0.64 | mean radius 1.00 | mean perimeter 1.00 | mean concavity 1.00 |
| 2 | worst concave points 0.60 | worst concave points 0.48 | mean texture 1.00 | mean concavity 1.00 | mean concave points 1.00 |
| 3 | mean concave points 0.36 | worst concavity 0.48 | mean perimeter 1.00 | mean concave points 1.00 | worst radius 1.00 |
| 4 | mean area 0.32 | mean area 0.44 | mean area 1.00 | worst radius 1.00 | worst perimeter 1.00 |
| 5 | mean texture 0.24 | worst area 0.36 | mean smoothness 1.00 | worst perimeter 1.00 | worst area 1.00 |
| 6 | worst concavity 0.24 | worst texture 0.32 | mean compactness 1.00 | worst area 1.00 | worst concave points 1.00 |
| 7 | texture error 0.24 | mean perimeter 0.32 | mean concavity 1.00 | worst concavity 1.00 | mean perimeter 0.80 |
| 8 | fractal dimension error 0.24 | fractal dimension error 0.32 | mean concave points 1.00 | worst concave points 1.00 | worst concavity 0.80 |
| 9 | worst radius 0.20 | worst smoothness 0.32 | radius error 1.00 | area error 0.80 | worst texture 0.80 |
| 10 | mean concavity 0.20 | worst symmetry 0.32 | perimeter error 1.00 | mean area 0.80 | mean radius 0.60 |

**Notes on ties**
- Boruta keeps about 25 of 30 features, and many tie at 1.00. Its top 10 are simply the first ten of those, in index order.
- Ties at the 10th place for GA and BPSO are broken by feature index. See `top10_features.csv`.

## 5. Findings

**(i) Which methods include the reference features above chance** (run-level CIs as pre-specified, Full feature set excluded):

- **Strict, above chance:** BPSO, SA, NSGA-II, LIME, RFE, LASSO and Boruta.
- **Strict, not above chance:**
  - **GA:** +0.10, CI [−0.17, 0.38].
  - **GWO:** +0.29, CI [−0.01, 0.61], which is borderline.
  - **SHAP:** exactly at chance (+0.00). Every SHAP subset has 10 features and contains worst area but never mean texture or worst smoothness.
- **Relaxed, above chance:** all five metaheuristics, LIME, RFE, LASSO and Boruta (Boruta only by +0.05).
- **Relaxed, below chance:** SHAP (−0.85). In 25 of 25 runs it hits no texture-group and no smoothness-group feature.

Under the supplementary seed-level intervals, SA and RFE (strict) and BPSO (relaxed) are no longer clearly above chance.

**(ii) Do strict and relaxed differ?** Yes, in three ways:
- **GA** is at chance under the strict metric but clearly above chance under the relaxed one (+0.63). It prefers *worst* texture (0.72) to *mean* texture (0.24), and the two correlate at r = 0.91.
- **GWO** follows the same pattern: borderline strict, clearly above chance relaxed.
- **SHAP** goes from exactly chance (strict) to clearly below chance (relaxed). Its 10 features fall entirely in the size and concavity clusters.
- **Boruta's** apparent strict advantage (+0.50) is mostly an artefact of keeping 25 of 30 features. Under the relaxed metric, its excess over chance is only +0.05.

**(iii) Correlations** (Pearson, all 569 WDBC samples, raw features):

| Feature pair | r |
|---|---|
| worst area – worst radius | 0.984 |
| worst area – worst perimeter | 0.978 |
| (worst radius – worst perimeter) | (0.994) |
| mean texture – worst texture | 0.912 |
| mean smoothness – worst smoothness | 0.805 |

These are high enough that exact-name matching can understate agreement. Selecting worst radius or worst perimeter carries almost the same information as worst area, and so on. The relaxed metric is the fairer test of whether a selector captured the three *kinds* of information named in the 1997 work.

## 6. Flags (surprising, or relevant to the paper's current claims)

1. **SHAP's 10 features are redundant and miss two of the three named kinds of information.**
   - Every SHAP subset contains worst radius, worst perimeter and worst area together (pairwise r ≥ 0.98), plus mean perimeter and concavity features.
   - No SHAP subset ever contains a texture or smoothness feature.
   - This does not contradict the paper's accuracy results. It does matter for the paper's interpretability framing ("a smaller subset gives a smaller explanation surface"), because SHAP's compact subset is compact in count but not in information.
2. **The paper's pooled "large subsets" claim does not hold on WDBC.** On WDBC the metaheuristics choose small subsets (s = 3.8–7.5 of 30). The pooled 71–102 features (paper, Table 3) are driven by the high-dimensional datasets. The claim is correct as a pooled statement but should not be read as applying to WDBC.
3. **The metaheuristics' instability is partly substitution among correlated features.** The paper reports Nogueira stability near 0 for the metaheuristics, measured on exact feature indices. On WDBC, the texture group is hit in 88–100% of metaheuristic runs, and the size group in 48–80%, with mean versus worst texture swapping from run to run. This does not contradict the Nogueira numbers, which are correct for exact indices. It does suggest that "their subsets agree across seeds no more than random subsets" overstates the disagreement at the level of feature *groups*, at least on WDBC.
4. **LASSO is the only method that always includes worst smoothness** (1.00). LIME, RFE and SHAP never include it, and Boruta always does only because it keeps nearly everything.
5. **Scope.**
   - The metric counts overlap with three features named by one 1997 study and the UCI notes. It does not measure clinical correctness.
   - WDBC is one dataset.
   - For the five classifier-independent selectors, the effective sample is 5 seeds, not 25 runs.

## Files in `results_wdbc_features/`

| File | Contents |
|---|---|
| `step0_names.txt` | the 30 names and indices, and the per-seed order check |
| `step1_verify.txt` | the 275-row verification |
| `wdbc_masks.pkl` | the verified masks used in this analysis |
| `step2_metrics.txt` | the full metric output and correlations |
| `metrics_by_method.csv` | the main table |
| `metrics_seed_cluster_ci.csv` | the supplementary seed-cluster CIs |
| `step2_seed_cluster.txt` | the distinct-mask check and the seed-cluster output |
| `top10_features.csv` | the top-10 lists |
| `REPORT.md` | this report |
