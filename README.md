# Metaheuristic vs. XAI Feature Selection Benchmark

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

### Fitness-weight ablation

```bash
python src/run_ablation.py --n-runs 10        # GA with alpha in {0.7, 0.8, 0.9}
```

Writes `results/ablation_alpha.csv` and `results/ablation_alpha_summary.csv`.

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
- Academic / research use; cite scikit-learn, SHAP, LIME, DEAP, PySwarms, Boruta.
