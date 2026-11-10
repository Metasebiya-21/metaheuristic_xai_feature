# Runtime estimate for the cost table (Table 4)

This is a read-only analysis of files on disk (2026-11-10). Nothing was rerun, and no manuscript, results or code file was changed.

- **Inputs:** `results/all_runs.csv` (seeds 42–44, 825 rows) and `results_5seed/all_runs.csv` (seeds 42–46, 1,375 rows).
- **Rows used:** GA, BPSO, SA, GWO and NSGA-II only, which gives 125 (dataset, classifier, method) cells.
- **Per-cell values:** written to the scratchpad file `runtime_cells.csv`, which is not part of the repository.

## 0. Correction to `verify_report/REPORT_verify.md` (item 6)

REPORT_verify.md says that every metaheuristic row in `results/all_runs.csv` has `nominal_evaluation_budget = 800`. **That is wrong.** The value 800 is the mean over datasets. The per-dataset budgets are:

| Dataset | Nominal budget (results/, 3 seeds) | Nominal budget (results_5seed/) |
|---|---|---|
| Colon | 2,000 | 2,000 |
| Ionosphere, Leukemia, Madelon, WDBC | 500 | 2,000 |

Source: `results/all_runs.csv`, columns `dataset`, `method` and `nominal_evaluation_budget`, grouped by dataset and method. Colon also has identical lookup counts in both runs (for example GWO 2,051, BPSO 2,002 and SA 2,003).

The main conclusion stands: the runtime column was measured at a smaller budget than the Evals column. But the phrase "a nominal budget of 800 evaluations", inserted into `main_v2.tex` by edit 1f, is also inexact. CHANGES.md proposes replacement wording.

## 1. What the counters count

- **Lookups:** `actual_oracle_evaluations` is `search_oracle.total_evaluations` (src/metaheuristic_xai/runner.py:123, 153). That counter is incremented on **every** call to `evaluate()`, including cache hits (src/metaheuristic_xai/oracle.py:121–123 for a cache hit, 145 for a fresh fit). It therefore counts fitness **lookups**, not classifier fits.
- **Cache hits:** `cache_hits` counts lookups served from the memo cache (oracle.py:122).
- **Cache-hit rate:** `cache_hit_rate = cache_hits / actual_oracle_evaluations` (runner.py:155–157). It is a fraction of lookups.
- **Fresh fits:** fresh fits = `actual_oracle_evaluations − cache_hits`.
- **Runtime:** `runtime_seconds` is the selector's own `runtime_` (runner.py:119). It covers the search only. The two final evaluations on the test and validation oracles (runner.py:115–116) come after it and are not counted.

So the Evals column of Table 4 (for example GA 1,728) counts lookups. GA performed only about 901 fresh fits on average in the five-seed run.

## 2. Method

For each cell:

- `sec_per_fresh` = the mean over the 3 seeds in `results/` of `runtime_seconds / fresh`.
- `est_runtime` = `sec_per_fresh` × the mean `fresh` over the 5 seeds in `results_5seed/`.

A ratio-of-means variant, Σruntime / Σfresh, gives the same per-method means to within 0.5 s. Per-method values are equal-weight means over the 25 cells (5 datasets × 5 classifiers).

## 3. Results (seconds per run)

| Method | 3-seed measured (Table 4 now) | Fresh fits, 3-seed | Fresh fits, 5-seed | **Estimate at 5-seed fresh fits** | 5-seed measured |
|---|---|---|---|---|---|
| GA | 114.0 | 427 | 901 | **400.6** | 339.6 |
| BPSO | 245.8 | 747 | 1,718 | **905.0** | 763.2 |
| SA | 207.5 | 623 | 1,132 | **677.6** | 520.4 |
| GWO | 243.2 | 830 | 2,035 | **876.4** | 648.5 |
| NSGA-II | 228.5 | 805 | 1,886 | **815.5** | 549.6 |

**Estimate per dataset** (mean over 5 classifiers):

| Method | Colon | Ionosphere | Leukemia | Madelon | WDBC |
|---|---|---|---|---|---|
| GA | 67.3 | 46.1 | 57.2 | 1,786.3 | 46.0 |
| BPSO | 117.6 | 105.2 | 100.8 | 4,100.0 | 101.5 |
| SA | 103.8 | 23.0 | 81.4 | 3,145.6 | 34.3 |
| GWO | 124.6 | 149.3 | 105.3 | 3,829.8 | 173.1 |
| NSGA-II | 122.5 | 121.0 | 100.3 | 3,602.6 | 131.2 |

**3-seed measured per dataset**, for comparison:

| Method | Colon | Ionosphere | Leukemia | Madelon | WDBC |
|---|---|---|---|---|---|
| GA | 67.9 | 13.2 | 15.7 | 457.5 | 15.8 |
| BPSO | 117.6 | 29.5 | 25.2 | 1,025.0 | 31.9 |
| SA | 101.5 | 11.9 | 24.4 | 876.5 | 23.2 |
| GWO | 124.6 | 38.3 | 27.0 | 980.8 | 45.5 |
| NSGA-II | 122.5 | 34.1 | 25.7 | 922.6 | 37.6 |

**5-seed measured per dataset:**

| Method | Colon | Ionosphere | Leukemia | Madelon | WDBC |
|---|---|---|---|---|---|
| GA | 55.4 | 29.7 | 56.2 | 1,527.2 | 29.3 |
| BPSO | 90.5 | 71.1 | 94.5 | 3,491.6 | 68.2 |
| SA | 81.1 | 16.1 | 78.0 | 2,404.0 | 22.8 |
| GWO | 102.5 | 101.8 | 100.1 | 2,822.3 | 115.9 |
| NSGA-II | 97.9 | 80.5 | 88.3 | 2,394.4 | 86.8 |

Madelon dominates every mean, as the paper already says.

## 4. Checks

**(i) Is runtime proportional to fresh fits?**
- **Across all 125 cells:** Pearson r(runtime, fresh) = −0.06 (p = 0.52) and Spearman ρ = 0.43. Within single methods across their 25 cells, Pearson ranges from −0.13 to −0.03 and Spearman from 0.18 to 0.52. This pooled correlation is uninformative: fit cost varies by orders of magnitude between datasets and classifiers (Madelon versus the rest), and the 500-versus-2,000 budget split runs across datasets.
- **The relevant test is within a (dataset, classifier) pair**, where fit cost is roughly fixed. Across its 15 runs (5 methods × 3 seeds), Pearson r(runtime, fresh) has a median of **0.98**. It ranges from 0.58 (Madelon/LightGBM) to 0.999, and 23 of 25 pairs exceed 0.93. The coefficient of variation of `sec_per_fresh` within a pair has a median of **0.05** (maximum 0.23, Madelon/LightGBM).
- So within a dataset–classifier pair, runtime is close to linear in fresh fits and nearly method-independent. That is the assumption the scaling needs.

**(ii) Estimate against measured, where fresh fits match.** In all 25 Colon cells, both runs used the same 2,000 budget, and the five-seed run averaged 0.95–1.06 times the three-seed fresh-fit count. No other cell is within ±25%. In these 25 matched cells:
- the estimate equals the three-seed measurement almost exactly;
- the **five-seed measured runtime is 0.78× the estimate** (median; range 0.65–0.98).

Across all 125 cells, five-seed measured ÷ estimate has a median of 0.75, a 10–90% range of 0.62–0.93, and per-dataset medians of 0.65–0.88. The non-iterative selectors, whose runtime does not depend on budget, show the same direction: SHAP 1.15 s vs 1.50 s, RFE 4.60 vs 5.97, LASSO 20.5 vs 25.3 and LIME 39.7 vs 51.9. Boruta is the exception (47.3 vs 41.3).

**What this means.** On identical work, the five-seed run was about 20–25% **faster** per fit than the three-seed run, fairly uniformly across datasets. That pattern fits a systematic machine or configuration difference better than sporadic load spikes. The files do not show the five-seed runtimes being inflated relative to the three-seed run. The claim that "part of the five-seed run executed under heavy machine load" cannot be confirmed or refuted from the CSVs, because no load or host log is stored alongside them. If load slowed some five-seed runs, the three-seed run was slower still per fit.

## 5. Assumptions and how well they hold

1. **Cost per fresh fit within a (dataset, classifier) cell does not depend on the budget.** Well supported by (i): CV is about 0.05 across methods and seeds. Subset sizes barely change between runs (mean |S| moves from 87 to 81 for GA and is roughly flat for the others), so fit cost per call should not shift much.
2. **Search overhead apart from fits is negligible or also proportional.** Supported by the within-pair r ≈ 0.98. The scheme does not model the cost of a cache lookup or GA bookkeeping.
3. **The machine speed of the three-seed run applies.** This is a choice, not an assumption the data confirm. The estimate is on the three-seed run's clock, which (ii) shows is about 1.3× slower than the five-seed run's clock on the same work. On the five-seed clock, the estimate would be about 0.75–0.78× the values above, which is close to the five-seed measured column.

**Bottom line.**
- The scaled estimate is internally sound.
- It mainly shows that the current runtime column understates the 2,000-budget cost about 3–4× (GA 114 s → about 400 s).
- At the 2,000 budget, GA costs about 10× Boruta (400.6 / 41.3) and about 270× SHAP (400.6 / 1.50). The paper currently says about 3× and 75×.
- The files give no evidence that the five-seed measured runtimes are unusable. They are self-consistent with the estimate up to a uniform speed factor.

## 6. Not done

- No rerun or timing experiment.
- No look at machine logs; none are stored with the CSVs.
- The estimate is **not** applied to the manuscript. A ready-to-swap version is in `report/panafricon2026/CHANGES.md` under "OPTIONAL SWAP".
