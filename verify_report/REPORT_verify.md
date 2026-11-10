# GA-XAI camera-ready verification report

Read-only check of the claims in `report/panafricon2026/main.tex` against files on disk (2026-11-10).
No experiment was rerun and no file other than this report was written. The statistics below were
recomputed in Python from the CSV/parquet files listed (pandas, scipy 1.18.0).

Manuscript used: `report/panafricon2026/main.tex`. Bibliography used: `report/panafricon2026/sn-bibliography.bib`.
Table numbers refer to the compiled PDF: the nested-CV table you call "Table 8" is **Table 6**
(`\label{tab:ga-nested}`, main.tex:310), the sweep table is **Table 5** (main.tex:283), the cost table is **Table 4** (main.tex:261),
and the dataset table is **Table 2** (main.tex:169).

---

## 1. Sweep wording

**Source.** `results/ablation/ablation_runs.csv` (90 data rows). Columns used: `alpha` (col 2), `generations` (col 3),
`nominal_budget` (col 4), `seed` (col 5), `accuracy` (col 6, the test accuracy), `generalization_gap` (col 8),
`n_features` (col 9) and `selected_indices` (col 14). There are 45 rows at 40 generations (budget 2,000) and 45 at
100 generations (budget 5,000); each dataset × α × budget cell has 5 seeds (42–46). The producer is `src/run_ablation.py`
(GA + `RandomForestAdapter`, `build_oracles`, lines 51–73). Paper text: main.tex:300; paper table: main.tex:289–294.

**(a) Pooled mean test accuracy** (3 datasets × 5 seeds = 15 runs per configuration; unrounded):

| Budget | α | Accuracy | Mean gap | Mean \|S\| | Table 5 (paper) |
|---|---|---|---|---|---|
| 2,000 | 0.7 | 0.904473 | 0.043237 | 32.73 | **0.905** ← should be 0.904 |
| 2,000 | 0.8 | 0.903518 | 0.047117 | 33.20 | 0.904 ✓ |
| 2,000 | 0.9 | 0.910552 | 0.043742 | 34.20 | 0.911 ✓ |
| 5,000 | 0.7 | 0.895589 | 0.057249 | 28.00 | 0.896 ✓ |
| 5,000 | 0.8 | 0.899559 | 0.060230 | 28.60 | 0.900 ✓ |
| 5,000 | 0.9 | 0.906362 | 0.054012 | 29.47 | 0.906 ✓ |

All |S| and gap cells in Table 5 match after rounding. One accuracy cell does not: 0.904473 rounds to **0.904**, not 0.905
(main.tex:289).

**(b) Largest differences in pooled accuracy**
- Across α within budget 2,000: 0.910552 − 0.903518 = **0.0070** (α=0.9 vs 0.8).
- Across α within budget 5,000: 0.906362 − 0.895589 = **0.0108** (α=0.9 vs 0.7).
- Across all six configurations: 0.910552 − 0.895589 = **0.0150** (2,000/α=0.9 vs 5,000/α=0.7). This mixes α and budget.

**(c) Paired Wilcoxon tests.** These use `scipy.stats.wilcoxon` defaults (two-sided; zero differences dropped). Runs are
paired on (dataset, seed), giving 15 pairs; "nonzero" is the number of pairs whose accuracies differ.

Across α within a budget (mean difference = second α minus first):

| Budget | Comparison | Mean diff | Nonzero pairs | p |
|---|---|---|---|---|
| 2,000 | 0.7 vs 0.8 | −0.00096 | 9 | 0.813 |
| 2,000 | 0.7 vs 0.9 | +0.00608 | 9 | 0.374 |
| 2,000 | 0.8 vs 0.9 | +0.00703 | 8 | **0.058** |
| 5,000 | 0.7 vs 0.8 | +0.00397 | 8 | 0.575 |
| 5,000 | 0.7 vs 0.9 | +0.01077 | 8 | 0.141 |
| 5,000 | 0.8 vs 0.9 | +0.00680 | 8 | 0.106 |

Between budgets at fixed α (5,000 minus 2,000):

| α | Accuracy diff | Nonzero pairs | p | Gap diff | \|S\| diff |
|---|---|---|---|---|---|
| 0.7 | −0.00888 | 5 | 0.343 | +0.01401 | −4.73 |
| 0.8 | −0.00396 | 6 | 0.753 | +0.01311 | −4.60 |
| 0.9 | −0.00419 | 4 | 0.715 | +0.01027 | −4.73 |

**(d) Change in mean generalization gap, 2,000 → 5,000:** +0.0140 (α=0.7), +0.0131 (α=0.8), +0.0103 (α=0.9).

**Verdict on the paper's numbers (main.tex:300)**
- "changed by at most 0.011": **right** (0.0108, the within-budget maximum, at budget 5,000).
- "(paired Wilcoxon p ≥ 0.14)": **wrong**. The smallest across-α p is 0.058 (budget 2,000, α 0.8 vs 0.9). The value 0.14 is the
  smallest p at budget 5,000 only. No comparison is significant at 0.05.
- "accuracy … −0.004 to −0.009": **right** (−0.0040, −0.0042, −0.0089).
- "(p ≥ 0.34)": **right** (smallest p = 0.343).
- "gap is widened by 0.010–0.014": **right** (+0.0103 to +0.0140).
- "about five features are removed": **right** (4.60–4.73).
- "90 runs": **right**.

**Colon masks.** The `selected_indices` strings were compared across the three α values for every (budget, seed) group of
`dataset == "colon"`. The masks are **identical in all 10 of 10 groups** (5 seeds × 2 budgets). Subset sizes per seed at budget
2,000: 79, 129, 71, 98, 71; at budget 5,000: 59, 126, 69, 81, 45. For comparison, no WDBC or Ionosphere group (0 of 10 each) has
identical masks across α. Verdict: "On Colon, identical subsets are obtained for the three values of α" is **CONFIRMED**.

---

## 2. Consensus folds (Table 6, nested CV)

**Source.** Code: `compare_ga.py` at the repo root. Results: `results/compare_ga_nested.csv` (columns `dataset`, `fold`, `method`,
`test_acc`, `n_features`). Paper: main.tex:308 and main.tex:321.

**Code behaviour** (`compare_ga.py`):
- Lines 30–37 set the design: 4 datasets, `K_OUTER = 5`, `M_INNER = 4`, `CONSENSUS_MIN = 3`, `ALPHA = 0.8`, `K = 10`, and
  `BUD = population_size 35, generations 25`.
- Lines 88–99: each outer fold runs 4 inner GA searches (`cv=5`, `target_k=K`, `**IMPROVED`) on that fold's training partition only.
  This is the variant (ii) configuration, as the paper states.
- Line 101: the consensus keeps features chosen in at least 3 of the 4 inner runs.
- Lines 102–106: if the consensus mask is empty, **`rec()` is not called**. No row is written: no accuracy of 0 and no
  fallback mask. The fold is simply absent from the CSV.
- Outer folds come from `load_dataset_folds(..., n_splits=5, random_state=0)` (compare_ga.py:122;
  `StratifiedKFold`, src/metaheuristic_xai/datasets.py:1077).

**Per-fold results for `ga_consensus`** (`results/compare_ga_nested.csv`):

| Dataset | Fold | test_acc | \|S\| |
|---|---|---|---|
| colon | 0 | 0.692308 | 1 |
| colon | 3 | 0.666667 | 2 |
| colon | 1, 2, 4 | — (empty mask, no row) | 0 |
| leukemia | 0 | 0.866667 | 1 |
| leukemia | 1 | 0.866667 | 1 |
| leukemia | 2 | 0.857143 | 1 |
| leukemia | 3, 4 | — (empty mask, no row) | 0 |

- Colon mean = 0.679487 over **2 folds** (Table 6 shows 0.6795 ✓).
- Leukemia mean = 0.863492 over **3 folds** (Table 6 shows 0.8635 ✓).
- WDBC (0.959603) and Ionosphere (0.908934) have all 5 folds.
- The pooled value 0.886142 is over **15 folds** (Table 6 shows 0.8861 ✓). Every other method has 20 rows.

**Verdict.** "its mask is empty on 3/5 Colon and 2/5 Leukemia folds, which are excluded from its averages" is **CONFIRMED**.

The text before the table, "means over five outer folds are reported in Table 6" (main.tex:308), is not exact for the
consensus row: its Colon value is over 2 folds, its Leukemia value over 3, and its pooled value over 15 folds rather than 20.
Because the dropped folds are the hardest ones, its pooled value is not directly comparable with the other rows.

**Not stated in the paper.** This experiment uses GA population 35 × 25 generations (compare_ga.py:36), not the 50 × 40 of
the main benchmark.

---

## 3. Selector details

All selectors are in `src/metaheuristic_xai/selectors/classical.py`; their hyperparameters are in
`src/metaheuristic_xai/config.py:67–78` (`DEFAULT_METHOD_KWARGS`).

**(a) Internal models**

| Selector | Internal model | Location |
|---|---|---|
| Tree-SHAP | `RandomForestClassifier(n_estimators=100, random_state=seed)` + `shap.TreeExplainer` | classical.py:55, 66 |
| LIME | `RandomForestClassifier(n_estimators=100, random_state=seed)` + `LimeTabularExplainer` | classical.py:115–124 |
| RFE | `RFE(RandomForestClassifier(n_estimators=100), n_features_to_select=10, step=0.1)` | classical.py:220–221 |
| Boruta | `BorutaPy(RandomForestClassifier(n_estimators=100, max_depth=5), n_estimators="auto", max_iter=100)` | classical.py:251–258 |
| LASSO | `LogisticRegressionCV(penalty="l1", solver="saga", cv=3, max_iter=1000)`, not a Random Forest | classical.py:171–178 |

The paper's "an internal Random Forest regardless of the evaluation classifier" for SHAP, LIME, RFE and Boruta (main.tex:160) is
**CONFIRMED**. One nuance: Boruta's forest has `max_depth=5`, and BorutaPy's `n_estimators="auto"` overrides the 100 trees.

**(b) LIME aggregation** (classical.py:126–143):
1. Up to `num_samples = 100` training rows are drawn without replacement with `np.random.default_rng(seed)` (lines 126–128).
2. Each row is explained with `explain_instance(..., num_features=n_dim)`, using LIME's default number of perturbations
   (not overridden).
3. For each feature, the **absolute weights for class label 1 are summed** across rows (lines 131–137). Rows that raise an
   exception are skipped silently.
4. The top k by summed |weight| are kept, with ties broken by the lower feature index (`np.lexsort`, lines 141–142).

For comparison, SHAP uses the **mean |SHAP value| for class 1** over up to 200 stratified training rows
(`shap_max_samples = 200`; classical.py:58–77).

**(c) k per selector**
- SHAP k = 10, LIME k = 10, RFE k = 10 (config.py:72–75). Observed in `results_5seed/all_runs.csv`, column `selected_features`:
  min = max = 10 for all three.
- LASSO: data-driven; features with |coef| > 1e-5, with a fallback of 1 feature (classical.py:187–191).
- Boruta: data-driven; confirmed features, falling back to tentative ones, then to the single top-ranked feature
  (classical.py:263–272).

**(d) Splits**
- **Outer split:** `train_test_split(test_size=0.2, stratify=y, random_state=seed)` (src/metaheuristic_xai/datasets.py:723–728;
  default `test_size=0.2`, datasets.py:1027). This is **80/20, stratified ✓**. With integer rounding, the realized test fractions
  are 0.200–0.210.
- **Inner split:** `train_test_split(X_tr, y_tr, test_size=val_size, random_state=split_id, stratify=...)`
  (src/metaheuristic_xai/oracle.py:454–456), with `val_size = SEARCH_VAL_SIZE = 0.25` (src/metaheuristic_xai/runner.py:34, 78, 97).
  This is **75/25, stratified ✓**.

**(e) Search budgets** (config.py:67–71)
- GA: population 50, 40 generations.
- BPSO: swarm 50, 40 iterations.
- GWO: population 50, 40 iterations.
- NSGA-II: population 50, 40 generations.
- SA: `max_steps = 2000`.
- `nominal_evaluation_budget` in `results_5seed/all_runs.csv` is 2,000 for all five metaheuristics in every row.

Verdict: **CONFIRMED** (main.tex:186).

**(f) What the seed controls.** The seed changes **both** the data split and the search randomness:
- Outer split: `load_dataset(name, random_state=seed)` (runner.py:185, 318).
- Inner validation split: `split_id=seed` (runner.py:96 → oracle.py:456).
- Evaluation classifier: `random_state=seed` (runner.py:89).
- Selector and search RNG: `get_feature_selector(..., random_state=seed)` (runner.py:109), which is passed on as `base_seed`
  (selectors/metaheuristic.py:108).

Seeds 42–46 are confirmed in `results_5seed/all_runs.csv` (1,375 rows).

---

## 4. Convergence claim

**Source.** `results/alpha_sensitivity/alpha_sensitivity.parquet` (8 rows) is the only convergence data on disk. Nothing else in
the repo mentions "first 20 generations" or "convergence traces".

It was produced by `src/notebook/benchmark.ipynb` (source line ~1018): `run_alpha_sensitivity(dataset_names=["wdbc","ionosphere"],
alpha_values=[0.5, 0.8], method_name="ga", classifier_name="random_forest", seeds=[0, 1])`.

Each row has `nominal_evaluation_budget = 5000` and a `convergence_history` of length 100 (100 generations). So these traces come
from seeds 0–1 and α ∈ {0.5, 0.8}, not from the benchmark's seeds 42–46, α = 0.8 or 40 generations.

**Index convention.** `conv[i]` is the best fitness after generation i+1 (src/metaheuristic_xai/algorithms/ga.py:186–208). The
initial population is not recorded. Improvement is therefore measured as `conv[0] − conv[-1]`, and "within the first 20
generations" means index ≤ 19.

**Generation at which each fraction of the total improvement is first reached** (index + 1):

| Trace | 50% | 80% | 90% |
|---|---|---|---|
| wdbc s0 α0.5 | 6 | 7 | 8 |
| wdbc s0 α0.8 | 5 | 5 | 8 |
| wdbc s1 α0.5 | 4 | 5 | 9 |
| wdbc s1 α0.8 | 9 | 14 | **21** |
| iono s0 α0.5 | 4 | 7 | 10 |
| iono s0 α0.8 | 3 | 10 | 17 |
| iono s1 α0.5 | 5 | 7 | 7 |
| iono s1 α0.8 | 8 | 16 | **38** |

**Traces reaching the threshold within 20 generations:** 50% → 8 of 8; 80% → **8 of 8**; 90% → **6 of 8**.

The paper defines no threshold. The earlier internal note `paper_update_5seed/REPORT.md:262–266` used 90% and counted 7 of 8.
It treated index 20 (generation 21) as "by generation 20", which is an off-by-one.

**Verdict**
- "eight convergence traces": **CONFIRMED**.
- "seven traces": **WRONG as written**. It is 6 of 8 at 90% and 8 of 8 at 80%, and no threshold is stated.
- The trace settings (100 generations, seeds 0–1, α ∈ {0.5, 0.8}) differ from the benchmark and are not stated.

---

## 5. Dataset table

Each dataset was loaded with `load_raw_dataset(name)` and `load_dataset(name, random_state=42)`
(src/metaheuristic_xai/datasets.py:367, 1025; WDBC from scikit-learn, the others through the local OpenML cache). The searched D
is `X_train.shape[1]`. The ANOVA prefilter keeps k = 300 features for Colon and Leukemia (datasets.py:149, 163).

| Dataset | N | Raw D | Searched D | Class counts | Paper Table 2 (main.tex:175–179) |
|---|---|---|---|---|---|
| WDBC | 569 | 30 | 30 | 357 / 212 | ✓ |
| Ionosphere | 351 | 34 | 33 (one column dropped in preprocessing) | 225 (g) / 126 (b) | ✓ |
| Madelon | 2600 | 500 | 500 | 1300 / 1300 | ✓ |
| Colon | 62 | 2000 | 300 | 40 / 22 | ✓ |
| Leukemia | 72 | 7129 | 300 | 47 (ALL) / 25 (AML) | ✓ |

**No mismatches.** Colon has 49 training rows, so its inner validation split has about 12–13 rows, consistent with "about 13
validation samples" (main.tex:233).

---

## 6. Computational table (Table 4)

**Evaluations and cache-hit rates** come from `results_5seed/all_runs.csv`:
- Evals = mean of `actual_oracle_evaluations` over the 125 runs per method.
- Cache = **unweighted mean of per-run `cache_hit_rate`**, where the rate is cache_hits / search_evals
  (src/metaheuristic_xai/runner.py:154–157).

| Method | Evals | Cache (mean of per-run rates) | Cache (pooled hits/lookups) | Paper |
|---|---|---|---|---|
| GA | 1728.44 | 47.77% | 32.4% | 1728 / 47.8% ✓ |
| BPSO | 2002.00 | 14.20% | 12.4% | 2002 / 14.2% ✓ |
| SA | 2003.00 | 43.50% | 30.3% | 2003 / 43.5% ✓ |
| GWO | 2051.00 | 0.76% | 0.8% | 2051 / 0.8% ✓ |
| NSGA-II | 2051.00 | 8.07% | 7.5% | 2051 / 8.1% ✓ |

**Runtimes** come from `results/all_runs.csv`: seeds 42–44, 825 rows. Each value is the mean of `runtime_seconds` over the 75 runs
per method (5 datasets × 5 classifiers × 3 seeds). The design is balanced, so this equals the mean of per-dataset means.

| Method | Runtime from results/ | Paper | 5-seed runtime (for reference) |
|---|---|---|---|
| GA | 114.04 | 114.0 ✓ | 339.56 |
| BPSO | 245.84 | 245.8 ✓ | 763.18 |
| SA | 207.51 | 207.5 ✓ | 520.40 |
| GWO | 243.25 | 243.3 ✓ | 648.53 |
| NSGA-II | 228.49 | 228.5 ✓ | 549.59 |
| SHAP | 1.50 | 1.5 ✓ | 1.15 |
| RFE | 5.97 | 6.0 ✓ | 4.60 |
| LASSO | 25.33 | 25.3 ✓ | 20.48 |
| Boruta | 41.34 | 41.3 ✓ | 47.31 |
| LIME | 51.87 | 51.9 ✓ | 39.68 |

**Provenance problem.** In `results/all_runs.csv`, every metaheuristic row has `nominal_evaluation_budget = 800`, with mean actual
evaluations of 694 (GA) to 831 (GWO, NSGA-II). The 5-seed run used 2,000. The paper calls the runtime source "an earlier three-seed
run of the same benchmark (seeds 42–44)" (main.tex:255).

So the metaheuristic runtimes in Table 4 were measured at **about 40% of the evaluation budget** behind the Evals column in the same
table. The ratios built on them ("about three times the cost of Boruta and 75 times the cost of SHAP", main.tex:255) therefore
understate the cost at 2,000 evaluations. The non-iterative selectors are unaffected, because their runtime does not depend on the
budget.

---

## 7. Bibliography

Parsed `report/panafricon2026/main.tex` (with `%` comments removed) for `\cite{...}` keys, and
`report/panafricon2026/sn-bibliography.bib` for entries.

- **Cited keys:** 41 unique keys across 63 key occurrences.
- **Bib entries:** 46.
- **Cited but missing from the .bib:** none.
- **Unused entries:** `saeys2007` (bib line 236), `cawley2010` (275), `lipton2018` (324), `garcia2008` (368),
  `benavoli2016` (377).
- **Entries with an empty author, title, year or venue:** none. The venue field checked is `journal` for `@article`,
  `booktitle` for `@inproceedings`, and `publisher` for `@book`; `doshivelez2017` is a `@misc` entry with a venue field present.
- A clean BibTeX run on this pair gives 0 warnings and 41 `\bibitem`s (`report/panafricon2026/main.bbl`).

**Caveats**
- This checks the local copy. The `sn-bibliography.bib` in Overleaf was edited by hand and may differ.
- Field values (volume, pages, years) were not checked against the publishers. Several entries were added from memory during
  revision; `bernett2024` and `alshalif2024` were confirmed against the author-supplied bib.

---

## Summary

| Item | Claim in paper | Verdict | Corrected wording if wrong |
|---|---|---|---|
| 1 | Table 5 row α=0.7, budget 2,000: Acc. 0.905 | WRONG | 0.904 (unrounded 0.904473) |
| 1 | "pooled test accuracy is changed by at most 0.011" | CONFIRMED | — (0.0108, within budget 5,000) |
| 1 | "(paired Wilcoxon p ≥ 0.14)" across α | WRONG | "(paired Wilcoxon $p \ge 0.058$)" — smallest p is 2,000, α 0.8 vs 0.9 |
| 1 | budget effect "−0.004 to −0.009 (p ≥ 0.34)" | CONFIRMED | — |
| 1 | "generalization gap is widened by 0.010–0.014" | CONFIRMED | — |
| 1 | "about five features are removed"; "90 runs" | CONFIRMED | — |
| 1 | "On Colon, identical subsets are obtained for the three values of α" | CONFIRMED | — (10/10 seed × budget groups) |
| 2 | Consensus mask empty on 3/5 Colon, 2/5 Leukemia; excluded | CONFIRMED | — (empty: Colon folds 1, 2, 4; Leukemia 3, 4; no row is written) |
| 2 | Colon 0.6795, Leukemia 0.8635, pooled 0.8861 | CONFIRMED | — (over 2, 3 and 15 folds) |
| 2 | "means over five outer folds are reported in Table 6" | WRONG for the consensus row | "…means over five outer folds are reported in Table 6 (for the consensus variant, over the folds with a non-empty mask)." |
| 3a | SHAP, LIME, RFE, Boruta use an internal Random Forest | CONFIRMED | — (Boruta: max_depth 5, auto tree count) |
| 3b | LIME aggregation into top-k | CANNOT VERIFY in the paper (not stated) | Code: summed \|weight\| for class 1 over 100 sampled training rows, then top 10 |
| 3c | k = 10 for SHAP, LIME, RFE | CONFIRMED | — (LASSO, Boruta data-driven) |
| 3d | Inner 75/25; outer stratified 80/20 | CONFIRMED | — |
| 3e | Population 50, 40 generations; SA 2,000 steps | CONFIRMED | — (BPSO swarm 50 × 40, GWO 50 × 40) |
| 3f | Seed role | CONFIRMED (both) | — (outer split, inner split, classifier and search RNG) |
| 4 | "eight convergence traces (GA, Random Forest, WDBC and Ionosphere)" | CONFIRMED | — |
| 4 | "most of the improvement … within the first 20 generations in seven traces" | WRONG | "…80% of the fitness improvement was achieved within the first 20 of 100 generations in all eight traces." (or: "90% … in six of eight") |
| 5 | Table 2 N, raw D, searched D, class balance | CONFIRMED | — (no mismatch) |
| 6 | Evals and cache rates from the five-seed run | CONFIRMED | — (means over 125 runs; cache = mean of per-run rates) |
| 6 | Runtimes, including SHAP 1.5, RFE 6.0, LASSO 25.3, Boruta 41.3, LIME 51.9 | CONFIRMED numerically | — (means over 75 runs in `results/all_runs.csv`) |
| 6 | "an earlier three-seed run of the same benchmark (seeds 42–44)" | WRONG | "…an earlier three-seed run (seeds 42–44) with a nominal budget of 800 evaluations, so the metaheuristic runtimes understate the cost at 2,000 evaluations." |
| 7 | All citations resolve | CONFIRMED | — (41 cited, 0 missing, 5 unused, no empty fields) |
