# Metaheuristic XAI Feature Selection Benchmark

This repository benchmarks nature-inspired feature selection methods against a SHAP-based baseline on the Breast Cancer Wisconsin dataset. The project compares a genetic algorithm (GA), binary particle swarm optimization (BPSO), and simulated annealing (SA) with a SHAP reference method using paired stratified train/test splits and a Random Forest classifier.

## What this project does

The workflow is designed for reproducible experiments in feature selection and explainable AI:

- Load and scale the Breast Cancer Wisconsin dataset
- Create paired stratified splits for all methods
- Evaluate binary feature masks with a Random Forest-based fitness function
- Compare metaheuristic search methods to a SHAP-top-k baseline
- Generate CSV summaries and visualization artifacts for analysis and reporting

The optimization objective is:

$$
f(S) = 0.9(1 - \text{Accuracy}) + 0.1\frac{|S|}{d}
$$

where:

- $S$ is the selected feature subset
- $d$ is the total number of features
- lower values are better

The all-zero mask is treated as the worst-case solution with zero accuracy and maximum penalty.

## Project structure

- [main.py](main.py) — top-level entry point that forwards to the benchmark runner
- [metaheuristic_xai/main.py](metaheuristic_xai/main.py) — main experiment runner
- [metaheuristic_xai/src](metaheuristic_xai/src) — implementation modules for:
  - [metaheuristic_xai/src/baseline.py](metaheuristic_xai/src/baseline.py) — SHAP baseline
  - [metaheuristic_xai/src/ga.py](metaheuristic_xai/src/ga.py) — genetic algorithm
  - [metaheuristic_xai/src/pso.py](metaheuristic_xai/src/pso.py) — binary PSO
  - [metaheuristic_xai/src/sa.py](metaheuristic_xai/src/sa.py) — simulated annealing
  - [metaheuristic_xai/src/evaluation.py](metaheuristic_xai/src/evaluation.py) — metrics, CSV export, and Wilcoxon tests
  - [metaheuristic_xai/src/fitness.py](metaheuristic_xai/src/fitness.py) — fitness function and Random Forest evaluation
  - [metaheuristic_xai/src/data_loader.py](metaheuristic_xai/src/data_loader.py) — dataset loading and preprocessing
  - [metaheuristic_xai/src/utils.py](metaheuristic_xai/src/utils.py) — plotting and helper functions
- [metaheuristic_xai/run_ablation.py](metaheuristic_xai/run_ablation.py) — GA fitness-weight ablation study
- [metaheuristic_xai/results](metaheuristic_xai/results) — generated CSV artifacts
- [metaheuristic_xai/plots](metaheuristic_xai/plots) — generated figures
- [metaheuristic_xai/report](metaheuristic_xai/report) — LaTeX term paper source and build tooling

## Requirements

This project uses Python 3.10+ (the package metadata targets 3.12). Install the dependencies with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dependency list includes:

- numpy
- pandas
- matplotlib
- scikit-learn
- scipy
- shap
- deap
- pyswarms

For reproducible and less noisy runs, it is also useful to cap joblib/sklearn parallelism:

```bash
export SKLEARN_N_JOBS=1
```

## Running the benchmark

From the repository root:

```bash
python3 main.py
```

This runs the full benchmark with the default configuration of 30 paired splits. Each split uses a deterministic random seed of $42 + i$, and SHAP, GA, PSO, and SA all share the same split for a paired comparison design.

### Quick smoke test

```bash
python3 main.py --quick
```

This uses:

- one data split
- shorter GA and PSO schedules
- a smaller SA step cap

It is useful for verifying that the pipeline runs correctly without waiting for the full experiment.

### Fitness-weight ablation study

```bash
python3 metaheuristic_xai/run_ablation.py --n-runs 10
```

This evaluates GA behavior for different accuracy weights ($\alpha \in \{0.7, 0.8, 0.9\}$) and writes ablation CSV files to the results directory.

## Outputs

Running the benchmark generates the following artifacts:

- [metaheuristic_xai/results/all_runs.csv](metaheuristic_xai/results/all_runs.csv) — one row per run and method with accuracy, selected feature count, fitness, and runtime
- [metaheuristic_xai/results/summary.csv](metaheuristic_xai/results/summary.csv) — per-method summary statistics
- [metaheuristic_xai/results/wilcoxon_vs_shap.csv](metaheuristic_xai/results/wilcoxon_vs_shap.csv) — paired Wilcoxon signed-rank test results versus SHAP
- [metaheuristic_xai/plots/convergence_fitness.png](metaheuristic_xai/plots/convergence_fitness.png) — convergence plots for GA, PSO, and SA
- [metaheuristic_xai/plots/bar_accuracy.png](metaheuristic_xai/plots/bar_accuracy.png), [metaheuristic_xai/plots/bar_n_features.png](metaheuristic_xai/plots/bar_n_features.png), and [metaheuristic_xai/plots/bar_runtime.png](metaheuristic_xai/plots/bar_runtime.png) — summary bar charts
- [metaheuristic_xai/plots/shap_baseline_top_features.png](metaheuristic_xai/plots/shap_baseline_top_features.png) — top-k SHAP feature importance plot

## Rebuilding the report

The LaTeX report source is located in [metaheuristic_xai/report](metaheuristic_xai/report). To build the paper:

```bash
cd metaheuristic_xai/report
make
```

Or use the provided shell script.

## Notes

- The benchmark is intended for academic and research use.
- For statistically meaningful Wilcoxon comparisons, use a larger number of paired splits (for example, 30 or more).
- The implementation uses standardized features and a Random Forest classifier for consistent evaluation across methods.

## License

This repository is intended for academic and research use. When publishing results, cite the underlying libraries and datasets used by the project.
