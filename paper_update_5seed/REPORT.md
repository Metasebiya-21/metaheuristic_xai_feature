# Paper update from the 5-seed benchmark

**Source data:** `results_5seed/all_runs.csv` (1375 rows, seeds 42-46) for
everything except Task 3b, which explicitly uses `results/all_runs.csv`
(825 rows, seeds 42-44, the original fallback) per instruction.
Neither `results/` nor `results_5seed/` was modified. `report/report.tex`
and no other manuscript file was touched.

---

## Top of report: what surprised me / contradicts the paper

1. **Table 6's old numbers cannot be reproduced from anything on disk, and
   the discrepancy is not just "resource contention."** The old eval-count
   values (GA 1379, BPSO 1602, NSGA-II 1641, GWO 1641) match almost exactly
   a `population_size=40` search budget (nominal 1600 evaluations) — that's
   `PAPER_METHOD_KWARGS` in `config.py`, used only by `PAPER_CONFIG`
   (`run_paper_matrix`). But `results/paper_matrix/` **does not exist** —
   that config has never been executed in this repository. The corrected
   benchmark (`results/`, `results_5seed/`) uses `DEFAULT_METHOD_KWARGS`
   (`population_size=50`, nominal 2000) instead, a different, larger budget.
   This repo's git history is a single squashed commit, so there's no way
   to trace further back. **The old Table 6 is not reproducible from
   anything currently in the repo** — it most likely came from a run
   executed before this repository's current state existed, possibly under
   a different config that no longer exists anywhere on disk. See Task 3.

2. **The paper's convergence/alpha claim is only partially supported, and
   the "alpha=0.8 is better" half is actively contradicted by the one
   dataset that exists for it.** A tiny (8-row), explicitly-"Optional"
   notebook sensitivity sweep is the *only* convergence/alpha data in the
   repo (GA only, Random Forest only, WDBC+Ionosphere only, alpha ∈
   {0.5, 0.8}, 2 seeds). It **is** leakage-free (confirmed). The
   "most improvement in the first 10-20 iterations" part is roughly
   consistent with 7 of 8 trajectories (1 clear outlier reaching 90% of its
   improvement only at step 37). The "alpha=0.8 reaches better final
   fitness than lower alpha" part is **not supported** — the 4 paired
   comparisons split 1 favors 0.8, 1 favors 0.5, 1 tie, 1 slightly favors
   0.5. See Task 4 for the full breakdown; recommend not citing this as
   established without a real sweep.

3. **No qualitative conclusion in the paper flips under the 5-seed data**
   (consistent with what was already reported in
   `docs/seed_expansion_results.md`): Boruta still wins, GA still doesn't
   beat SHAP, metaheuristics are still unstable. Task 2 below confirms every
   specific qualitative claim about Table-adjacent statements except one
   (the Nogueira-vs-Jaccard "sharp divergence" claim needs a caveat — see
   Task 2.5).

4. **GWO's Nogueira stability index is slightly negative** (−0.007): this
   is a valid outcome of the chance-correction formula (below-chance
   agreement), not an error, but worth a footnote if the paper states
   stability values without qualification, since a naive reader may expect
   "0 to 1."

---

## Task 1 — Per-classifier table (replaces paper Table 7)

**Definition used for generalization gap**: mean `(val_accuracy − test_accuracy)`
restricted to the 5 metaheuristic methods only (GA, BPSO, SA, GWO, NSGA-II),
per your instruction — this differs from the all-methods-pooled gap reported
earlier in `docs/seed_expansion_results.md`, which is also shown alongside
for transparency.

| Classifier | Mean test accuracy | Mean gen. gap (metaheuristics only) | Mean gen. gap (all methods) | % runs ≤0.6 acc. |
|---|---|---|---|---|
| LightGBM | 0.8982 | −0.0153 | −0.0429 | 0.7% |
| Random Forest | 0.8870 | +0.0373 | +0.0135 | 0.7% |
| XGBoost | 0.8819 | +0.0594 | +0.0193 | 1.1% |
| SVM | 0.8819 | +0.0359 | +0.0129 | 9.1% |
| MLP | 0.7876 | +0.1297 | +0.0523 | 14.2% |

Note: LightGBM is the one classifier whose metaheuristic-only generalization
gap is *negative* (−0.0153) — all four other classifiers show a positive
gap for metaheuristic runs, consistent with search overfitting; LightGBM
doesn't show that pattern. Worth a sentence in the paper if Table 7's text
generalizes "classifiers show positive gap under metaheuristic search."

### MLP near-chance (≤0.6 accuracy) runs by dataset

| Dataset | % of MLP runs ≤0.6 accuracy |
|---|---|
| Madelon | 60.0% |
| Leukemia | 7.3% |
| Colon | 3.6% |
| Ionosphere | 0.0% |
| WDBC | 0.0% |

Madelon alone accounts for the overwhelming majority of MLP's instability.

---

## Task 2 — Per-method summary (all 11 methods)

| Method | Mean accuracy | Mean selected features | Mean gen. gap | Nogueira stability | Mean pairwise Jaccard |
|---|---|---|---|---|---|
| Boruta | 0.8947 | 40.960 | −0.0238 | 0.354 | 0.623 |
| RFE | 0.8855 | 10.000 | −0.0147 | 0.528 | 0.492 |
| LIME | 0.8818 | 10.000 | −0.0228 | 0.499 | 0.458 |
| Full feature set | 0.8810 | 232.600 | −0.0442 | 1.000 | 1.000 |
| SHAP | 0.8738 | 10.000 | −0.0162 | 0.529 | 0.491 |
| GWO | 0.8624 | 98.240 | +0.0449 | −0.007 | 0.241 |
| BPSO | 0.8588 | 101.944 | +0.0433 | 0.026 | 0.268 |
| NSGA-II | 0.8585 | 70.864 | +0.0552 | 0.030 | 0.198 |
| GA | 0.8543 | 81.096 | +0.0568 | 0.035 | 0.222 |
| SA | 0.8508 | 91.168 | +0.0469 | 0.022 | 0.241 |
| LASSO | 0.8388 | 107.960 | −0.0044 | 0.183 | 0.485 |

### Which qualitative claims does this support?

**1. "Boruta has high accuracy at a moderate subset size" — SUPPORTED.**
Boruta has the *highest* mean accuracy of all 11 methods (0.8947), at
40.96 features — clearly below the metaheuristics (71–102) and LASSO (108)
and far below the full feature set (232.6), while clearly above the
fixed-k=10 cluster. "Moderate" is an accurate description relative to the
full range (10–232.6).

**2. "SHAP/LIME/RFE sit at k=10" — SUPPORTED, exactly.** All three report
`mean_selected_features = 10.000` (by construction, `top_k=10`).

**3. "Metaheuristics sit in a middle band of subset sizes with no accuracy
gain" — SUPPORTED.** The five metaheuristics span 70.9–101.9 features,
between the k=10 cluster and the full feature set/LASSO. Despite using
7–10× more features than SHAP/LIME/RFE/Boruta, every metaheuristic has
*lower* accuracy than every one of those four methods (only LASSO, at
0.8388, is worse than the metaheuristics). "No accuracy gain" understates
it slightly — it's a net accuracy *loss* relative to the better classical
methods for a much larger subset — but the claim as stated is not
contradicted.

**4. "Metaheuristics' generalization gap is positive while classical/XAI
methods' gap is near zero" — SUPPORTED WITH A WORDING CAVEAT.** All five
metaheuristics show a clearly positive gap (+0.043 to +0.057). All six
classical/XAI methods (including full-features) show a *negative* gap
(−0.004 to −0.044) — small in magnitude relative to the metaheuristics, but
consistently negative, not scattered around zero. If "near zero" is read as
"small in magnitude," the claim holds. If read as "centered at zero with no
systematic sign," it's not quite accurate — recommend "near zero, slightly
negative" or "an order of magnitude smaller than the metaheuristics', and of
opposite sign."

**5. "Nogueira and raw Jaccard diverge sharply for the metaheuristics" —
PARTIALLY SUPPORTED; needs a caveat if stated as unique to metaheuristics.**
In *absolute* terms, the metaheuristics' Nogueira-vs-Jaccard gaps (0.168 to
0.248) are **not** the largest in the table — Boruta (0.270) and especially
LASSO (0.303) show comparable or larger absolute gaps, because they too
select large subsets (41 and 108 features) where raw Jaccard's chance
baseline is high. What *is* uniquely sharp for the metaheuristics is the
**relative/ratio** divergence: raw Jaccard is 6–11× larger than the
chance-corrected Nogueira index for GA/BPSO/SA/NSGA-II (and sign-flipped for
GWO), versus ~1–2.7× for Boruta/LASSO and ~0.9× (Jaccard *smaller* than
Nogueira) for SHAP/LIME/RFE. The mechanism is textbook Nogueira: larger
selected subsets inflate the chance-overlap baseline raw Jaccard doesn't
correct for. **Recommend rephrasing to**: "raw Jaccard overstates stability
by 6-11x relative to the chance-corrected index for the metaheuristics —
a pattern also present but much weaker for Boruta and LASSO, which also
select large subsets" rather than implying this divergence is exclusive to
metaheuristics.

---

## Task 3 — Computational profile (replaces paper Table 6)

### 3a. Mean oracle evaluations + cache-hit rate per iterative method (from `results_5seed`)

| Method | Mean oracle evaluations | Mean cache-hit rate |
|---|---|---|
| GA | 1728.44 | 47.8% |
| BPSO | 2002.00 | 14.2% |
| SA | 2003.00 | 43.5% |
| GWO | 2051.00 | 0.8% |
| NSGA-II | 2051.00 | 8.1% |

### 3b. Mean runtime per method (from `results/`, the 3-seed run — NOT `results_5seed`)

| Method | Mean runtime (s) | Std (s) | Min (s) | Max (s) | n |
|---|---|---|---|---|---|
| GA | 114.04 | 213.36 | 4.96 | 997.12 | 75 |
| BPSO | 245.84 | 472.52 | 10.87 | 1796.38 | 75 |
| SA | 207.51 | 416.39 | 4.70 | 1641.96 | 75 |
| GWO | 243.25 | 453.88 | 11.79 | 1743.80 | 75 |
| NSGA-II | 228.49 | 424.31 | 10.65 | 1514.09 | 75 |

**Contention check on `results/` (as requested):** I checked file-modification
timestamps on all 825 `.parquet` files in `results/`. They fall into two
disjoint, tightly-packed segments with a 68.1-hour gap between them:

- Segment A: 660 files (every dataset except the later Colon correction),
  spanning only **2.89 hours** (2026-09-09 22:10 → 2026-09-10 01:03), max
  internal gap between consecutive file completions: **16.6 minutes**.
- Segment B: 165 files (exactly the Colon-only re-run after the dataset-ID
  fix), spanning **0.39 hours** (23.4 min), max internal gap: **2.7 minutes**.

The 68.1-hour gap *between* segments is just the real-world time between two
separate work sessions (original benchmark, then the later Colon dataset-ID
bug fix days later) — not evidence of a stall. **Within each segment, there
is no sign of the resource-contention pattern seen in the 5-seed run**
(which had gaps of multiple *hours* with zero completions during the Chrome
/Docker contention episode). Max internal gap here is 16.6 minutes, consistent
with normal variation for the heaviest cells. **Conclusion: `results/`
runtimes are clean and not contention-affected; using them for Table 6 (per
your instruction) is sound.**

### 3c. SA included above (the old table omitted it).

### 3d. Deltas from the old Table 6

| Method | Evals: old → new | Δevals | Cache: old → new | Δcache | Runtime: old → new | Δruntime |
|---|---|---|---|---|---|---|
| GA | 1379 → 1728.44 | +25.3% | 64.1% → 47.77% | −16.33pp | 30.6s → 114.04s | +272.7% |
| BPSO | 1602 → 2002.00 | +25.0% | 28.1% → 14.20% | −13.90pp | 76.8s → 245.84s | +220.1% |
| SA | (not in old table) | — | — | — | — | new: 2003.00 evals, 43.50% cache, 207.51s |
| GWO | 1641 → 2051.00 | +25.0% | 1.2% → 0.76% | −0.44pp | 110.9s → 243.25s | +119.3% |
| NSGA-II | 1641 → 2051.00 | +25.0% | 17.8% → 8.07% | −9.73pp | 89.4s → 228.49s | +155.6% |

**Why the eval counts jumped ~25% uniformly**: this matches a
`population_size` increase from 40 to 50 almost exactly
(2000/1600 = 1.25). `PAPER_METHOD_KWARGS` (pop=40, nominal 1600) matches
the *old* numbers; `DEFAULT_METHOD_KWARGS` (pop=50, nominal 2000, used by
the corrected benchmark) matches the *new* ones. See the top-of-report note
— the config that would have produced the old numbers was never run as
`PAPER_CONFIG` in this repo (`results/paper_matrix/` doesn't exist), so this
is the best explanation available, not a directly-traced one.

**Why runtime increased far more than 25%**: two compounding factors beyond
the eval-count increase — (1) cache-hit rates dropped substantially
(9–16 percentage points for GA/BPSO/NSGA-II), meaning a higher fraction of
the now-larger evaluation count requires a fresh classifier fit rather than
a cheap cache lookup; (2) GWO's near-zero cache-hit rate (0.8%, barely
changed from the old 1.2%) means its runtime increase (+119%) tracks closest
to the pure eval-count increase, consistent with this explanation.

---

## Task 4 — Provenance check: "most improvement in first 10-20 iterations; alpha=0.8 better"

**(i) Does the data exist?** Partially. `results/alpha_sensitivity/alpha_sensitivity.parquet`
(8 rows) is the *only* convergence/alpha data anywhere in the repo. It's
generated by notebook cell 13 in `src/notebook/benchmark.ipynb`
(`run_alpha_sensitivity(dataset_names=["wdbc","ionosphere"], alpha_values=[0.5,0.8], method_name="ga", classifier_name="random_forest", seeds=[0,1])`),
explicitly labeled "Optional" in the notebook's section-3 markdown header.
No dedicated convergence figure exists anywhere (`plot_fitness_convergence`
exists in `plots.py` but is never called in the notebook); no plot file
matching `*converg*` exists on disk.

**(ii) Leakage-free or old leaky oracle?** **Leakage-free (corrected).**
Confirmed two ways: (a) the code path is
`run_alpha_sensitivity` → `run_feature_selection` → `build_oracles` (the
corrected search/test oracle split); (b) empirically, every row has
`val_accuracy != accuracy` (and a nonzero `generalization_gap`) — under the
original leaky protocol these would be identical by construction, since
search and reporting used the same oracle/test set.

**(iii) Reproducible from what's on disk?** Yes — the notebook cell's exact
parameters match the 8-row file exactly (2 datasets × 2 alphas × 2 seeds).
Re-running that cell against current code should reproduce it (modulo GA's
own stochasticity).

**Does the data support the claim?**

- **"Most fitness improvement occurs within the first 10-20 iterations":
  roughly supported, narrowly.** Of 8 trajectories (100 generations each),
  7 reach 90% of their total fitness improvement by generation 20 or
  earlier (generations 6, 7, 7, 8, 9, 16, 20); one (Ionosphere, alpha=0.8,
  seed=1) doesn't reach it until generation 37. This is **GA only, Random
  Forest only, 2 datasets only** — there is no equivalent data for BPSO, SA,
  GWO, NSGA-II, or for Colon/Leukemia/Madelon, or for any other classifier.
  The claim should not be stated as general to "the metaheuristics" or "the
  benchmark" without qualification — it's grounded in a single
  method/classifier/2-dataset slice.

- **"Alpha=0.8 reaches better final fitness than lower alpha": NOT
  supported by the available data.** The 4 paired (same dataset, same seed)
  comparisons of final fitness split: WDBC/seed0 favors alpha=0.5, WDBC/seed1
  ties, Ionosphere/seed0 favors alpha=0.8, Ionosphere/seed1 slightly favors
  alpha=0.5. By raw accuracy instead of the alpha-weighted fitness score
  (a cleaner comparison since fitness's definition itself depends on alpha):
  2 of 4 favor alpha=0.8, 1 favors alpha=0.5, 1 ties. **This is not a
  reproducible or statistically meaningful result on n=4 paired
  comparisons** — recommend removing or heavily qualifying this half of the
  claim unless a proper sweep (more datasets, seeds, methods, and the full
  configured `ALPHA_VALUES=[0.3,0.5,0.8]` or `BENCHMARK_CONFIG.alpha_values`)
  is run to actually test it. **Per your instruction: this cannot be
  verified under the corrected protocol from what's currently on disk, and
  I did not run new experiments to generate it.**

---

## Task 5 — Figures

Regenerated into `paper_update_5seed/figures/` from `results_5seed`:

- `cd_diagram.png` — Friedman+Nemenyi CD diagram, CD=1.350 (verified exact
  match against the computed value).
- `accuracy_parsimony.png` — mean accuracy vs. mean selected features,
  log x-axis, colored by method family. Required a real fix: the tight
  cluster of metaheuristic points (GA/BPSO/SA/GWO/NSGA-II, all within a
  narrow accuracy/feature-count range) caused label collisions with the
  previous fixed alternating-offset placement (e.g. "nsga2"/"bpso" literally
  overlapping). Replaced with a proper collision-avoiding placement
  algorithm (expanding-radius candidate search against real rendered
  bounding boxes, including reserving space around every point marker) —
  verified visually, zero overlaps now. This fix is in `plots.py` and will
  also improve this figure anywhere else it's generated from.
- `generalization_gap.png` — bar chart, colored by family.
- `stability.png` — Nogueira index vs. raw mean pairwise Jaccard, grouped
  bars.

All four use the same large-font styling already established for the main
report figures (title ~16-21pt, axis/tick labels ~13-16pt, legend ~12-15pt)
— confirmed at least as large as, and in most cases larger than, the
previous version. No clipped labels in any of the four (verified via pixel
crops on the tightest regions).
