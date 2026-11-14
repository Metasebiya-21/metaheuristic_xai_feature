# Reproducing the GA-XAI paper

**Paper:** "GA-XAI: A Benchmark of Metaheuristic and Classical Feature Selection for Clinical Machine Learning".

**Where the numbers are.**
- `results/`, `results_5seed/`, `results_sweep/`, `results_convergence/` and `results_wdbc_features/` are working directories. They are **not tracked by git**: only a few files in them are committed, and the rest are gitignored or untracked.
- **The numbers behind the paper are in [`paper_results/`](paper_results/).** [`paper_results/MANIFEST.md`](paper_results/MANIFEST.md) lists every file with its SHA-256, row count and the table or figure it feeds.

**Release v1.0.0 predates the leakage correction.** Do not use it to reproduce the paper; use v2.0.0 or later.

## Environment

- Python 3.12. The paper runs used 3.12.15 on macOS with 6 worker processes.
- Pinned dependencies are in `uv.lock`:

  ```bash
  uv sync
  ```

- `requirements.txt` lists the packages **without versions**. Use `uv.lock` for exact reproduction.
- Run every command below **from the repository root**.

## Expected runtimes and outputs

| Step | Command | Writes to | Wall time (6 workers) |
|---|---|---|---|
| 1. Main benchmark (seeds 42–46) | `main.py` (below) | `results_5seed/` | about 17 h |
| 2. Nested-CV GA variants (Table 5) | `scripts/paper/nested_ga/compare_ga.py` | `results/compare_ga_nested.csv` | not recorded |
| 3. Parameter sweep (Table 4) | `scripts/paper/sweep/sweep.py run` | `results_sweep/runs/`, `results_sweep/run_runs.csv` | about 7.8 h |
| 4. Sweep analysis | `scripts/paper/sweep/analyze.py` | `results_sweep/analysis_table.csv` | under 1 min |
| 5. Convergence (Figure 5) | `scripts/paper/convergence/*.py` | `results_convergence/` | under 1 min |
| 6. WDBC agreement (Table 6, §4.7) | `scripts/paper/wdbc_agreement/*.py` | `results_wdbc_features/` | under 1 min |
| 7. Figures 3, 4 and the CD diagram | `scripts/paper/figures/make_paper_figures.py` | `--out-dir` | under 1 min |

**Step dependencies.** Steps 3–7 read the step-1 outputs in `results_5seed/`.
- Steps 5 and 6 read the **per-cell `.parquet` files**, not just `all_runs.csv`, because `all_runs.csv` stores only truncated convergence histories.
- Those parquet files (32 MB) are not in `paper_results/`, so you need step 1 first.

**Overwrite warning.** The scripts in steps 2, 5 and 6 write to fixed paths and overwrite existing files there. Run them in a fresh clone if you want to keep the shipped outputs for comparison.

## 1. Main benchmark

5 datasets × 5 classifiers × 11 methods × 5 seeds, giving 1,375 runs. α = 0.8, and the inner validation split is 0.25.

```bash
python main.py --datasets all --classifiers all --methods all --seeds 5 --jobs 6 --results-dir results_5seed
```

- **Configuration.** The CLI builds the same configuration as `BENCHMARK_CONFIG` in `src/metaheuristic_xai/config.py`: `DEFAULT_METHOD_KWARGS`, seeds 42–46, `search_val_size` 0.25.
- **Original wrapper.** The paper run used a small wrapper, `run_5seed_benchmark.py`, which called `run_full_benchmark(BENCHMARK_CONFIG, resume=True, n_jobs=6)` and then `save_results`. That wrapper is not in the repository; the CLI command above is the equivalent.
- **Resume.** The run resumes from the per-cell parquet files in `--results-dir`, so it can be restarted after an interruption. Pass `--fresh` to recompute everything.
- **Long runs.** For a run this long on macOS:

  ```bash
  nohup caffeinate -i python main.py ... > run.log 2>&1 &
  ```

**Outputs**
- `results_5seed/all_runs.csv`: one row per run.
- `results_5seed/summary.csv`
- `results_5seed/feature_stability.csv` and `wilcoxon_results.csv` (CLI only).
- One `.parquet` file per run.

**Paper sources.** `paper_results/main_5seed/` feeds Table 2 (accuracy, Friedman ranks, Wilcoxon vs SHAP, Nogueira stability), Table 3 (runtime, evaluations, cache-hit rate) and Figures 3–4.

## 2. Nested-CV GA variants (Table 5)

```bash
python scripts/paper/nested_ga/compare_ga.py
```

The design uses 5 disjoint outer folds, Random Forest, population 35 and 25 generations, on WDBC, Ionosphere, Colon and Leukemia. Output: `results/compare_ga_nested.csv`, copied to `paper_results/nested_ga/`.

## 3. Parameter-robustness sweep (Table 4)

Each configuration changes one factor away from the defaults. The design covers GA, BPSO, SA, GWO and NSGA-II on Random Forest, all 5 datasets, seeds 42–46 (1,025 runs):

| Factor | Levels |
|---|---|
| Budget | 1,000 / 5,000 / 10,000 evaluations |
| Population | 25 / 100 at a budget of 2,000 |
| α | 0.7 / 0.9 |
| Method parameter | 0.5× and 2× of the default (GA mutation rate, BPSO inertia weight w, SA initial temperature, NSGA-II mutation rate) |

```bash
python scripts/paper/sweep/sweep.py verify      # optional: reruns the baseline (seed 42) and compares it with results_5seed
python scripts/paper/sweep/sweep.py run --n-jobs 6
python scripts/paper/sweep/analyze.py | tee results_sweep/analysis_output.txt
```

- **Resume.** `run` writes one JSON file per run to `results_sweep/runs/` and skips files that already exist, so an interrupted sweep resumes.
- **Baseline.** The sweep's baseline cells come from `results_5seed/all_runs.csv`, so step 1 must be complete.
- **BPSO's inertia weight w.** It is passed as a `BPSOSelector` constructor argument (`c1`, `c2`, `w`; defaults 0.5, 0.3 and 0.9).
  - The original sweep set w by patching `run_binary_pso` inside the worker process.
  - Both routes give identical results: verified on BPSO, WDBC, seed 42, w = 0.45 (see `REPORT_RELEASE.md`).

**Paper sources:** `paper_results/sweep/`.

## 4. Convergence analysis (Figure 5)

```bash
python scripts/paper/convergence/analyze.py | tee results_convergence/analysis_output.txt
python scripts/paper/convergence/gen20_and_rerun_cost.py | tee -a results_convergence/analysis_output.txt
python scripts/paper/convergence/plot.py
```

This reads `convergence_history` from `results_5seed/*.parquet`. NSGA-II stores no scalar history, so it is absent. **Paper sources:** `paper_results/convergence/`.

## 5. WDBC feature agreement (Table 6, Section 4.7)

```bash
python scripts/paper/wdbc_agreement/step0_names.py        | grep -v Warning | tee results_wdbc_features/step0_names.txt
python scripts/paper/wdbc_agreement/step1_verify.py       | tee results_wdbc_features/step1_verify.txt
python scripts/paper/wdbc_agreement/step2_metrics.py      | tee results_wdbc_features/step2_metrics.txt
python scripts/paper/wdbc_agreement/step3_seed_cluster.py | tee results_wdbc_features/step2_seed_cluster.txt
```

These four scripts were recovered verbatim from the commands that produced the shipped outputs, and they reproduce them byte for byte.
- Run them in order: step 1 writes `wdbc_masks.pkl`, which steps 2 and 3 read.
- Create `results_wdbc_features/` first.

**Paper sources:** `paper_results/wdbc_agreement/`.

## 6. Paper figures (Figures 3–4, CD diagram)

```bash
python scripts/paper/figures/make_paper_figures.py --runs results_5seed/all_runs.csv --out-dir figures_out
```

The script checks that the Friedman subjects (125) and the critical difference (1.35) match the paper. Its output matches the manuscript figures byte for byte.

## Legacy: GA α × budget ablation

```bash
python src/run_ablation.py --generations 40 100 --n-jobs -1
```

This is the earlier 90-run GA-only ablation, kept in `paper_results/ga_ablation/`. The sweep in §3 supersedes it.
