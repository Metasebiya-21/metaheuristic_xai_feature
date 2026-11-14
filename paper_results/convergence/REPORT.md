# Convergence analysis (GA-XAI, reviewer request)

Generated 2026-11-11.

- `results/`, `results_5seed/`, the code and the manuscripts were only read, not modified.
- Everything was written to `results_convergence/`. No experiment was rerun.

## 1. What is stored

**Source.** The histories come from the per-cell parquet files `results_5seed/<dataset>_<classifier>_<method>_seed<s>.parquet`, column `convergence_history`. The same column in `results_5seed/all_runs.csv` holds **truncated NumPy reprs** (SA's contains "…"), so it is unusable for this analysis.

| Method | Stored? | Length (all 125 runs) | One entry is… | Code |
|---|---|---|---|---|
| GA | yes | 40 | best fitness in the population after generation g = 1…40. The population is elitist, so this is best-so-far. The initial population is **not** logged. | `algorithms/ga.py:183–208` |
| BPSO | yes | 40 | pyswarms `cost_history`: the global-best cost after each of 40 iterations (best-so-far) | `algorithms/pso.py:93, 104–106` |
| SA | yes | 2,001 | entry 0 is the cost of the initial random mask; entry k is the best-so-far after move k (2,000 single-flip moves) | `algorithms/sa.py:93–107` |
| GWO | yes | 41 | entry 0 is the alpha (best) of the initial pack of 50; entry t is the alpha after iteration t (40 iterations) | `algorithms/gwo.py:65, 79, 133` |
| NSGA-II | **NO** | 0 (empty in all 125 runs) | — | `algorithms/nsga2.py:241–313` |

**About NSGA-II.** `run_nsga2` returns no scalar `convergence`. It keeps only Pareto-front snapshots at about 5 generations (`pareto_history`), held in memory by the selector (`selectors/metaheuristic.py:260`), and the runner never writes them out. So NSGA-II convergence **cannot be analysed from disk**, under any scalarization.

**Fitness logged.** Every entry is the inner-validation (search-oracle) fitness f = α(1−Acc) + (1−α)|S|/d, with α = 0.8 (`oracle.py:219`). Lower is better.

**How entries map to oracle evaluations.** These are nominal lookups. Lookups include cache hits, because the oracle counts every `evaluate()` call (`oracle.py:121–123, 145`).

| Method | Lookups at entry i (0-based) | Nominal total |
|---|---|---|
| GA | 50 + 50·(i+1) | 2,050 |
| BPSO | 50·(i+1) | 2,000 |
| SA | i + 1 | 2,001 |
| GWO | 50·(i+1) | 2,050 |

- **GA is an upper bound.** DEAP does not re-evaluate offspring left unchanged by crossover and mutation, so GA makes only about 1,728 lookups on average (`actual_oracle_evaluations`) rather than 2,050. The per-generation split is not stored, so GA's x-values are about 16% too far right on average.
- **SA's fit count is lower than its lookup count.** About 44% of SA lookups are cache hits, so the number of classifier fits is lower still.

**Verification.**

| Check | Result |
|---|---|
| Final history value equals the run's stored `val_fitness` (\|diff\| ≤ 1e-9) | **500 / 500 runs (0 mismatches)** |
| Every history is non-increasing | 500 / 500 |

Full results are in `verification.csv`.

## 2. Definition used

- For each run, progress(i) = (h₀ − hᵢ) / (h₀ − h_final), where h₀ is the first stored entry.
- The milestone for X% is the first entry with progress ≥ X, expressed in nominal lookups.
- "Still improving in the last 10%" means the best-so-far fitness decreased at least once within the final 10% of the nominal budget.
- **Runs with zero total improvement were excluded:** 3 BPSO and 1 GWO, out of 500.

**Caveat on comparing methods.** The first entry is not the same kind of starting point for every method:
- **SA:** h₀ is a single random mask.
- **GWO:** h₀ is the best of 50 random masks.
- **GA and BPSO:** h₀ is the state after one generation or iteration, about 100 and 50 lookups in.

SA therefore has more easy improvement to make at the start, and its early milestones look faster for that reason.

## 3. Tables

Values are oracle lookups needed to reach X% of total improvement, as **median [IQR]**.

**Pooled over datasets and classifiers**

| Method | n | 50% | 80% | 90% | 95% | Still improving in last 10% |
|---|---|---|---|---|---|---|
| GA | 125 | 450 [300–650] | 800 [550–1300] | 1050 [750–1600] | 1300 [750–1800] | 41% |
| BPSO | 122 | 975 [550–1400] | 1425 [912–1750] | 1575 [1150–1800] | 1625 [1150–1800] | 35% |
| SA | 125 | 144 [31–340] | 318 [200–727] | 494 [235–1040] | 633 [259–1162] | 12% |
| GWO | 124 | 200 [150–300] | 600 [300–950] | 800 [450–1300] | 850 [600–1400] | 15% |
| NSGA-II | — | not stored | | | | |

**By dataset** (25 runs per cell, except where flat runs were excluded):

| Method | Dataset | 50% | 80% | 90% | 95% | Still improving in last 10% |
|---|---|---|---|---|---|---|
| GA | Colon | 700 [600–800] | 1350 [1100–1550] | 1650 [1400–1750] | 1800 [1650–1900] | **80%** |
| GA | Ionosphere | 350 [250–400] | 650 [500–800] | 850 [750–1350] | 1150 [750–1550] | 12% |
| GA | Leukemia | 650 [600–700] | 1250 [1150–1350] | 1650 [1400–1750] | 1800 [1700–1900] | **88%** |
| GA | Madelon | 350 [350–550] | 700 [550–950] | 900 [700–1100] | 950 [700–1100] | 20% |
| GA | WDBC | 300 [250–400] | 550 [300–700] | 800 [600–950] | 800 [600–950] | 4% |
| BPSO | Colon | 1000 [600–1450] | 1550 [1150–1700] | 1550 [1150–1750] | 1550 [1150–1750] | 32% |
| BPSO | Ionosphere (n=24) | 650 [475–1175] | 1325 [650–1638] | 1550 [650–1812] | 1550 [650–1812] | 42% |
| BPSO | Leukemia | 1350 [1050–1600] | 1700 [1150–1800] | 1700 [1350–1800] | 1700 [1400–1800] | 40% |
| BPSO | Madelon (n=24) | 800 [575–1050] | 1125 [838–1575] | 1400 [1038–1700] | 1500 [1188–1712] | 25% |
| BPSO | WDBC (n=24) | 575 [400–1350] | 1550 [588–1800] | 1650 [1300–1800] | 1650 [1300–1800] | 38% |
| SA | Colon | 328 [9–440] | 461 [233–837] | 637 [434–1046] | 991 [551–1271] | 8% |
| SA | Ionosphere | 88 [26–217] | 241 [84–318] | 260 [219–464] | 278 [227–871] | 0% |
| SA | Leukemia | 388 [65–498] | 588 [403–833] | 788 [616–1128] | 1076 [633–1318] | 20% |
| SA | Madelon | 61 [26–229] | 688 [176–928] | 829 [356–1165] | 865 [464–1376] | 32% |
| SA | WDBC | 134 [53–198] | 252 [214–283] | 309 [228–404] | 324 [250–562] | 0% |
| GWO | Colon | 200 [100–250] | 500 [250–800] | 700 [350–950] | 850 [600–1250] | 8% |
| GWO | Ionosphere | 250 [150–350] | 700 [250–1100] | 850 [400–1150] | 850 [600–1350] | 16% |
| GWO | Leukemia | 200 [200–250] | 550 [350–1050] | 950 [400–1400] | 1200 [700–1700] | 12% |
| GWO | Madelon | 300 [150–500] | 700 [450–1100] | 850 [700–1450] | 950 [750–1500] | 12% |
| GWO | WDBC (n=24) | 200 [150–262] | 475 [200–912] | 675 [450–1262] | 675 [450–1262] | 25% |

**GA progress after generation 20 of 40** (entry index 19, about 1,050 nominal lookups):

| Dataset | Runs with ≥50% | ≥80% | ≥90% | Median progress |
|---|---|---|---|---|
| Colon | 92% | **24%** | **16%** | 0.68 |
| Ionosphere | 100% | 88% | 72% | 0.94 |
| Leukemia | 92% | **20%** | **12%** | 0.72 |
| Madelon | 100% | 84% | 72% | 1.00 |
| WDBC | 100% | 96% | 92% | 1.00 |
| **Pooled** | **97%** | **62%** | **53%** | 0.92 |

Files: `milestones_pooled.csv`, `milestones_by_dataset.csv`, `per_run_milestones.csv`, `analysis_output.txt`.

## 4. Figure

`convergence.png`: one panel, 5.0 × 3.4 in at 300 dpi; the saved file is 4.75 × 3.17 in after tight cropping.

- **Axes.** The x-axis is the **fraction of the nominal evaluation budget** (lookups ÷ nominal total, about 2,000). The y-axis is the fraction of total improvement.
- **Curves.** Each method has a median line and an IQR band. Each run is interpolated onto a common grid, and there is no value before a method's first logged entry.
- **Style.** Okabe–Ito colorblind-safe colours, direct labels instead of a legend, and 9 pt text.
- **Missing method.** NSGA-II is absent because its history is not stored.

Script: `plot.py`.

## 5. Relation to the paper's claim

The current manuscript (`main_v3`/`main_v4`) says:
- 8 GA traces from a **separate** small run (Random Forest, WDBC and Ionosphere, seeds 0–1, α ∈ {0.5, 0.8}, **100 generations**) reached 80% of their improvement within 20 generations in all 8 cases, and 90% in 6 of 8.
- The original wording was "most of the improvement within the first 20 generations".

Against the main benchmark (40 generations, 125 GA runs):

- **"Most" (≥50%) within 20 generations:** holds in **97%** of GA runs. Even Colon and Leukemia reach it in 92%.
- **80%:** holds in only **62%** pooled.
  - WDBC 96%, Ionosphere 88% and Madelon 84% match the separate check: for Random Forest on WDBC and Ionosphere, 9 of 10 runs reach both 80% and 90%.
  - **Colon (24%) and Leukemia (20%) do not.**
- **The gene-expression datasets have not converged at the main budget.** In 80% (Colon) and 88% (Leukemia) of GA runs, validation fitness was **still improving in the final 10% of the 40-generation budget**. The separate 8-trace check covered only the two low-dimensional datasets, so it does not represent these cases.
- **This does not contradict the budget sweep.** That sweep found a larger budget does not raise test accuracy. Validation fitness can keep improving while test accuracy does not, which is the overfitting the paper describes. But the convergence sentence should not imply that GA had converged on all datasets.

## 6. Other findings

1. **BPSO converges latest by a wide margin.**
   - Its median reaches 90% only at about 79% of the budget (1,575 lookups), against 25% for SA, 39% for GWO and 51% for GA.
   - The pooled median curve shows **no improvement for about the first quarter of the budget**: over half of BPSO runs do not improve on their first-iteration best for about 10 iterations.
   - 35% of runs are still improving at the end.
2. **SA and GWO plateau early.** Only 12–15% of their runs are still improving in the last 10% of the budget. SA's early speed is partly an artefact of its single-mask starting point (§2).
3. **The axis is lookups, not classifier fits.**
   - Cache hits count as lookups: about 48% for GA and 44% for SA.
   - GA's x-positions are nominal upper bounds.
4. **NSGA-II, the method with the second-largest metaheuristic generalization gap in the paper (Table 3), has no recorded convergence trajectory.**

## 7. Cost of a minimal rerun to add NSGA-II

This was not run. The estimate uses the stored five-seed runtimes for the requested slice: WDBC, Ionosphere and Colon × Random Forest × seeds 42–46.

| Scope | Runs | Serial time |
|---|---|---|
| NSGA-II only | 15 | about 62 min, or about 10–15 min on 6 workers |
| All 5 methods | 75 | about 240 min, or about 40 min on 6 workers |

**Prerequisite:** a code change. `run_nsga2` must log a per-generation scalar, such as the minimum over the current front of f = α·error + (1−α)·|S|/d, the same scalarization used to pick NSGA-II's final mask (`nsga2.py:299–303`). The runner must also persist that history. Without the change, a rerun would produce no NSGA-II history.

The other four methods do not need a rerun.
