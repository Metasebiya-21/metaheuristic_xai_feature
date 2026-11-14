# Metaheuristic vs. XAI Feature Selection Benchmark

## Paper

This repository accompanies **"GA-XAI: A Benchmark of Metaheuristic and Classical
Feature Selection for Clinical Machine Learning"**. The numbers behind the paper
are in [`paper_results/`](paper_results/) (see its `MANIFEST.md`), the scripts in
`scripts/paper/`, and the exact commands in [REPRODUCE.md](REPRODUCE.md).

> **Release v1.0.0 predates the leakage correction** (see
> `docs/leakage_correction_note.md`) and must not be used to reproduce the paper.
> Use v2.0.0 or later.

Benchmarks nature-inspired metaheuristics against explainability-based and classical
feature selection across five public datasets and five classifier families, under a
shared evaluation oracle and a single scalar fitness.

## What it compares

| Family | Methods |
|---|---|
| Metaheuristic (scalar) | `ga` (GA/DEAP), `bpso` (binary PSO/PySwarms), `sa` (simulated annealing), `gwo` (grey wolf) |
| Metaheuristic (multi-objective) | `nsga2` |
| XAI | `shap` (TreeExplainer top-k), `lime` (aggregated LIME weights top-k) |
| Classical | `lasso` (L1 logistic regression), `rfe`, `boruta` |
| Baseline | `full_features` |

**Datasets** (`--list-datasets`): `wdbc`, `ionosphere` (low-dim), `madelon` (medium),
`colon`, `leukemia` (ultra-high-dim gene expression; pre-filtered to the top 300
features on the training split only).

**Classifiers**: `random_forest`, `xgboost`, `lightgbm`, `svm`, `mlp`.

## Fitness (minimized)

```
f(S) = alpha * (1 - accuracy) + (1 - alpha) * (|S| / d)
```

`alpha` defaults to `0.8`. The all-zero mask is the worst case (accuracy 0,
fitness 1). NSGA-II instead optimizes `(error, feature_ratio)` as two independent
objectives.

## Install

```bash
uv sync            # or: pip install -e .
```

Python 3.12+. Dependencies are declared in `pyproject.toml` (`uv.lock` pins them).

## Run

```bash
python main.py --quick --datasets wdbc --methods ga shap sa
python main.py --datasets wdbc ionosphere --classifiers random_forest --methods all --seeds 5
```

Flags: `--datasets/--classifiers/--methods` (names or `all`), `--seeds N` (splits
`42 .. 42+N-1`), `--quick` (1 seed + short search schedules), `--estimate` (print
the evaluation budget and exit), `--complexity`, `--list-datasets`,
`--results-dir/--plots-dir`.

Outputs (default `results/`, `plots/`):

| Path | Contents |
|---|---|
| `results/all_runs.csv` | one row per (dataset, classifier, method, seed): accuracy + full metric set, `selected_features`, `fitness`, `runtime_seconds`, oracle evaluation counts |
| `results/summary.csv` | per (dataset, classifier, method) means |
| `results/wilcoxon_results.csv` | paired Wilcoxon signed-rank vs SHAP (when >= 2 seeds and `shap` is included) |
| `results/<dataset>_<clf>_<method>_seed<n>.parquet` | per-run artifact (resume-friendly) |
| `plots/convergence.png` | best-fitness convergence for the search methods |

### Parameter-robustness sweep (used in the paper)

```bash
python scripts/paper/sweep/sweep.py run --n-jobs 6     # ~7.8 h on 6 workers
python scripts/paper/sweep/analyze.py
```

All five metaheuristics (GA, BPSO, SA, GWO, NSGA-II), Random Forest, all five
datasets, seeds 42-46, one factor varied at a time from the defaults: budget
(1,000 / 5,000 / 10,000 evaluations), population (25 / 100 at 2,000), alpha
(0.7 / 0.9) and one method parameter at 0.5x and 2x (GA and NSGA-II mutation
rate, BPSO inertia weight `w`, SA initial temperature) -- 1,025 runs, compared
pairwise with the five-seed baseline (Wilcoxon, Holm within factor). See
[REPRODUCE.md](REPRODUCE.md) and `paper_results/sweep/`.

### Notebook

`src/notebook/benchmark.ipynb` runs the whole flow top to bottom — setup, the
same `run_full_benchmark` the CLI uses, optional sensitivity sweeps, and analysis.
See `src/notebook/README.md`.

## Package layout (`src/metaheuristic_xai/`)

```
datasets.py        dataset registry + train-only preprocessing / split
classifiers.py     classifier adapters + CLASSIFIER_REGISTRY
oracle.py          EvaluationOracle: memoized mask -> metrics
selectors/         base.py, classical.py, metaheuristic.py + SELECTOR_REGISTRY
algorithms/        ga.py pso.py sa.py gwo.py nsga2.py (search operators)
config.py          BenchmarkConfig, presets, per-method evaluation budgets
runner.py          run_feature_selection / run_full_benchmark / sensitivity sweeps
evaluation.py      Wilcoxon/Friedman tests, CSV export, summary tables
pareto.py          Pareto-front metrics and plots (NSGA-II)
plots.py           convergence and bar-chart helpers
complexity.py      asymptotic complexity reference (--complexity)
cli.py             argument parsing -> runner -> summaries/plots
```

## Notes

- Leakage protocol: the search optimises against an inner validation split of the
  training set (`BenchmarkConfig.search_val_size`, default 0.25); the reported
  `accuracy` is a single held-out **test** evaluation of the final mask, identical
  for every method. `val_accuracy` and `generalization_gap` (val - test) are also
  recorded. `results/feature_stability.csv` reports selection stability across seeds
  (mean pairwise Jaccard + Nogueira index).
- For meaningful paired statistics use `--seeds 30` or more.
- Licensed under the Apache License 2.0 (see `LICENSE`). Please cite the paper (`CITATION.cff`) and scikit-learn, SHAP, LIME, DEAP, PySwarms, Boruta.
