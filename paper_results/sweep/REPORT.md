# Parameter-robustness sweep (GA-XAI, reviewer request)

The sweep finished on 2026-11-14 at 00:29 EAT: 1,025 runs in 7.8 h wall time on 6 workers, about 246 CPU-hours.

Write scope:
- `results/`, `results_5seed/`, the code and the manuscripts were only read, not modified.
- Everything was written to `results_sweep/`. Nothing was committed.

## 1. Design

**Shared settings**
- **Classifier and data.** Random Forest on all 5 datasets (WDBC, Ionosphere, Colon, Leukemia, Madelon), seeds 42–46.
- **Call path.** The same as the main benchmark: `runner.run_feature_selection`, `DEFAULT_METHOD_KWARGS`, validation split 0.25, `load_dataset(name, random_state=seed)`.
- **Determinism.** A rerun of the baseline reproduced `results_5seed` exactly (15/15 identical, `step0_determinism.txt`).
- **Baseline.** The baseline cells are not rerun; they are taken from `results_5seed`.

**Defaults (baseline)**

| Method | Defaults | α |
|---|---|---|
| GA | population 50, 40 generations, mutation_rate 0.2 | 0.8 |
| BPSO | swarm 50, 40 iterations, PySwarms options c1 0.5, c2 0.3, w 0.9, k = swarm−1, p 2 | 0.8 |
| SA | initial_temp 100, cooling 0.95, max_steps 2,000 | 0.8 |
| GWO | population 50, 40 iterations | 0.8 |
| NSGA-II | population 50, 40 generations, mutation_rate 0.05 | 0.8 |

The nominal budget is about 2,000 oracle lookups.

**Factors, varied one at a time** (all other settings stay at the defaults):

| Factor | Levels | Notes |
|---|---|---|
| F1 budget | 1,000 / 5,000 / 10,000 lookups | generations = budget / 50; SA max_steps = budget |
| F2 population | 25 / 100 at a 2,000 budget | generations = 2000 / population. N/A for SA. |
| F3 α | 0.7 / 0.9 | |
| F4 method parameter, 0.5× and 2× | GA mutation_rate 0.1 / 0.4; BPSO w 0.45 / 1.8; SA initial_temp 50 / 200; NSGA-II mutation_rate 0.025 / 0.1 | N/A for GWO, which has no tunable parameter |

**Size.** There are 41 non-baseline configurations, each run on 25 cells (5 datasets × 5 seeds), for 1,025 runs.

**BPSO's w.** `BPSOSelector` does not expose the PySwarms `options`, so w was set by patching `run_binary_pso` inside the worker process (`sweep.py:81–87`). No code file was changed.

## 2. Analysis (pre-specified in `analyze.py` before the sweep finished)

- **(a)** Mean test accuracy, |S| and generalization gap over 25 cells. Paired Wilcoxon test against the same method's baseline on the same (dataset, seed), Holm-corrected within each factor.
- **(b)** Paired comparison against SHAP and Boruta (Random Forest, same 25 cells).
  - SHAP mean accuracy is **0.8946** and Boruta's is **0.9173**.
  - Flags: a configuration's mean is above SHAP's, or within 0.005 of Boruta's.
- **(c)** Nogueira stability of the selected subsets across the 5 seeds, computed per dataset and averaged.

Full table: `analysis_table.csv`. Raw output: `analysis_output.txt`. Per-run JSON files: `runs/`.

### (a) and (c): effect of each factor

Columns:
- **Δ vs base:** mean paired difference in accuracy from the method's own baseline.
- **p (Holm):** Holm-corrected p-value.
- **Nog.:** Nogueira stability.

| Method | Factor | Level | Acc | \|S\| | Gap | Δ vs base | p raw | p (Holm) | Nog. |
|---|---|---|---|---|---|---|---|---|---|
| GA | baseline | — | 0.8856 | 83.8 | 0.037 | — | — | — | 0.057 |
| GA | budget | 1,000 | 0.8845 | 89.0 | 0.036 | −0.0011 | 0.90 | 1.00 | 0.055 |
| GA | budget | 5,000 | 0.8794 | 77.2 | 0.050 | −0.0062 | 0.48 | 1.00 | 0.048 |
| GA | budget | 10,000 | 0.8800 | 72.7 | 0.053 | −0.0056 | 0.60 | 1.00 | 0.055 |
| GA | population | 25 | 0.8807 | 83.0 | 0.043 | −0.0049 | 0.65 | 1.00 | 0.008 |
| GA | population | 100 | 0.8771 | 83.0 | 0.051 | −0.0085 | 0.79 | 1.00 | 0.016 |
| GA | α | 0.7 | 0.8906 | 79.9 | 0.031 | +0.0050 | 0.31 | 1.00 | 0.015 |
| GA | α | 0.9 | 0.8901 | 84.9 | 0.035 | +0.0045 | 0.13 | 1.00 | 0.043 |
| GA | mutation | 0.1 | 0.8877 | 87.8 | 0.037 | +0.0021 | 0.24 | 1.00 | −0.002 |
| GA | mutation | 0.4 | 0.8742 | 79.2 | 0.055 | −0.0114 | 0.17 | 1.00 | 0.013 |
| BPSO | baseline | — | 0.8761 | 105.6 | 0.037 | — | — | — | 0.026 |
| BPSO | budget | 1,000 | 0.8760 | 105.9 | 0.028 | −0.0001 | 0.79 | 1.00 | 0.013 |
| BPSO | budget | 5,000 | 0.8766 | 99.5 | 0.048 | +0.0004 | 0.90 | 1.00 | 0.050 |
| BPSO | budget | 10,000 | 0.8902 | 97.3 | 0.037 | +0.0141 | **0.037** | 0.56 | 0.054 |
| BPSO | population | 25 | 0.8824 | 100.6 | 0.038 | +0.0063 | 0.36 | 1.00 | 0.007 |
| BPSO | population | 100 | 0.8824 | 108.2 | 0.030 | +0.0063 | 0.40 | 1.00 | 0.026 |
| BPSO | α | 0.7 | 0.8728 | 100.6 | 0.033 | −0.0034 | 0.41 | 1.00 | 0.007 |
| BPSO | α | 0.9 | 0.8727 | 105.4 | 0.045 | −0.0034 | 0.20 | 1.00 | 0.025 |
| BPSO | w | 0.45 | 0.8754 | 106.1 | 0.035 | −0.0008 | 0.79 | 1.00 | −0.009 |
| BPSO | w | 1.8 | 0.8747 | 110.9 | 0.016 | −0.0014 | 0.96 | 1.00 | −0.005 |
| SA | baseline | — | 0.8686 | 93.9 | 0.037 | — | — | — | 0.001 |
| SA | budget | 1,000 | 0.8721 | 97.6 | 0.029 | +0.0035 | 0.34 | 1.00 | 0.020 |
| SA | budget | 5,000 | 0.8745 | 91.8 | 0.044 | +0.0059 | 0.58 | 1.00 | −0.004 |
| SA | budget | 10,000 | 0.8688 | 90.8 | 0.050 | +0.0002 | 0.97 | 1.00 | −0.005 |
| SA | α | 0.7 | 0.8703 | 88.8 | 0.034 | +0.0017 | 0.86 | 1.00 | 0.040 |
| SA | α | 0.9 | 0.8658 | 98.1 | 0.043 | −0.0028 | 0.97 | 1.00 | 0.011 |
| SA | initial temp | 50 | 0.8686 | 95.2 | 0.043 | −0.0000 | 0.88 | 1.00 | 0.053 |
| SA | initial temp | 200 | 0.8786 | 93.1 | 0.028 | +0.0100 | 0.15 | 1.00 | −0.009 |
| GWO | baseline | — | 0.8866 | 95.1 | 0.033 | — | — | — | −0.007 |
| GWO | budget | 1,000 | 0.8719 | 98.2 | 0.043 | −0.0147 | 0.062 | 0.87 | 0.022 |
| GWO | budget | 5,000 | 0.8805 | 94.2 | 0.044 | −0.0062 | 0.49 | 1.00 | 0.021 |
| GWO | budget | 10,000 | 0.8880 | 93.7 | 0.039 | +0.0014 | 0.82 | 1.00 | 0.035 |
| GWO | population | 25 | 0.8805 | 99.2 | 0.033 | −0.0062 | 0.66 | 1.00 | 0.019 |
| GWO | population | 100 | 0.8797 | 96.6 | 0.039 | −0.0069 | 0.38 | 1.00 | 0.005 |
| GWO | α | 0.7 | 0.8854 | 94.1 | 0.030 | −0.0012 | 0.78 | 1.00 | −0.010 |
| GWO | α | 0.9 | 0.8900 | 97.6 | 0.032 | +0.0034 | 0.62 | 1.00 | −0.018 |
| NSGA-II | baseline | — | 0.8875 | 70.9 | 0.042 | — | — | — | 0.027 |
| NSGA-II | budget | 1,000 | 0.8786 | 80.2 | 0.041 | −0.0089 | 0.34 | 1.00 | 0.020 |
| NSGA-II | budget | 5,000 | 0.8866 | 59.3 | 0.046 | −0.0009 | 0.81 | 1.00 | 0.059 |
| NSGA-II | budget | 10,000 | 0.8832 | 51.4 | 0.056 | −0.0043 | 0.86 | 1.00 | 0.066 |
| NSGA-II | population | 25 | 0.8680 | 67.8 | 0.064 | −0.0195 | **0.049** | 0.39 | 0.021 |
| NSGA-II | population | 100 | 0.8848 | 82.5 | 0.048 | −0.0027 | 0.78 | 1.00 | 0.039 |
| NSGA-II | α | 0.7 | 0.8848 | 68.9 | 0.042 | −0.0027 | 0.31 | 1.00 | 0.025 |
| NSGA-II | α | 0.9 | 0.8873 | 71.6 | 0.044 | −0.0002 | 0.57 | 1.00 | 0.035 |
| NSGA-II | mutation | 0.025 | 0.8681 | 62.3 | 0.063 | −0.0194 | 0.19 | 1.00 | 0.025 |
| NSGA-II | mutation | 0.1 | 0.8718 | 81.3 | 0.050 | −0.0157 | 0.17 | 1.00 | 0.032 |

Nogueira stability of the reference selectors on the same 25 cells:

| Selector | Nogueira |
|---|---|
| SHAP | 0.488 |
| RFE | 0.494 |
| LIME | 0.464 |
| Boruta | 0.348 |
| Lasso | 0.181 |

### (b) Against SHAP and Boruta

- **No configuration has a mean above SHAP's (0.8946).** The best is BPSO at a 10,000 budget (0.8902), which is 0.0044 lower (p = 0.52).
- **No configuration is within 0.005 of Boruta's mean (0.9173).**
  - The closest are GA at α = 0.7 (Δ = −0.027) and BPSO at a 10,000 budget (Δ = −0.027).
  - Every configuration is 0.027–0.052 below Boruta. Raw p is below 0.05 in 38 of 46 rows.

Per-row paired differences and p-values are in the `d_vs_shap`, `p_vs_shap`, `d_vs_boruta` and `p_vs_boruta` columns of `analysis_table.csv`.

## 3. Plain-language summary

- **No setting changed any method's accuracy reliably.** Of 41 tests, two have a raw p below 0.05 (BPSO at a 10,000 budget, +1.4 points; NSGA-II at population 25, −2.0 points). Neither survives Holm correction (adjusted p = 0.56 and 0.39). Every other mean is within about 1.5 points of its baseline.
- **More budget does not help.**
  - Raising the budget 5× (to 10,000 lookups) leaves GA, SA, GWO and NSGA-II at the same accuracy or slightly lower. Their generalization gap grows: for GA it goes from 0.037 to 0.053, and for NSGA-II from 0.042 to 0.056.
  - This matches the paper's overfitting account: the search keeps improving validation fitness, but test accuracy does not follow.
- **The paper's ranking survives tuning.** Across all 46 configurations, no metaheuristic reaches SHAP's mean accuracy or comes close to Boruta's. The conclusion "the XAI and filter baselines match or beat the metaheuristics" does not depend on the default settings.
- **Metaheuristic subsets are unstable across seeds under every setting.** Nogueira stability is between −0.02 and 0.07 in every configuration, against 0.35–0.49 for SHAP, RFE, LIME and Boruta. No tuning setting brings the metaheuristics close.

## 4. Flags and limitations

1. **Small n, one factor at a time.**
   - Each test has n = 25 pairs (5 datasets × 5 seeds), and the pairs are not independent within a dataset. Power to detect differences of about 1 point is low.
   - "Not significant" therefore means no evidence of an effect, not evidence of no effect.
   - The design varies one factor at a time, so it cannot detect interactions, such as budget × population.
2. **This uses Random Forest only.**
   - SHAP's mean here is 0.8946, over 25 cells. The paper pools 125 cells across classifiers.
   - In the 15-cell subset used during the probe (WDBC, Ionosphere and Colon only), baseline BPSO was above SHAP. On all 5 datasets it is not.
   - Random Forest–only subsets can rank methods differently from the pooled result, so these numbers should not be quoted as the paper's main comparison.
3. **BPSO at w = 1.8 is degenerate.**
   - 74% of its oracle lookups are cache hits (1,488 of 2,002), against 30% at the default. Its median wall time is 67 s, against about 230 s.
   - The cause: with w > 1, velocities grow, the sigmoid transfer saturates, and positions stop changing.
   - Its accuracy (0.8747) is close to the baseline only because the final mask is still the best one seen. Treat this level as "outside the working range", not as evidence that BPSO is robust to w.
4. **Cache hits make wall time a poor proxy for budget.**
   - At a 10,000 budget, GA makes 8,446 lookups, of which 6,123 (72%) are cache hits. SA makes 10,003, of which 6,891 (69%) are cache hits.
   - Both methods spend most of the extra budget re-visiting masks they have already seen. That is why their runtime grows far less than 5×.
   - GWO (4% hits) and BPSO (14%) make mostly fresh fits.
   - The "budget" axis is nominal lookups, as in the paper's Table 4.
5. **SA's initial temperature barely matters at cooling 0.95.**
   - With T ← 0.95·T after every move, the temperature falls below 1 after about 77 steps from 50, 90 from 100 and 104 from 200, out of 2,000. The 0.5× and 2× levels therefore change only the first ~5% of the run.
   - The +1.0 point at 200 (p = 0.15) should not be read as a tuning effect.
6. **BPSO's w was patched in-process** (§1). The patch passes the same options `pso.py:71–72` uses by default, with only w changed. It is not part of the released code.
7. **GWO has no F4 level, and SA has no F2 level.** Both are N/A by design.
8. **Minor bug in the output text.** `analyze.py` labels the SHAP/Boruta means "15 cells" in its printed output. They are over the 25 Random Forest cells. The table values are correct.
