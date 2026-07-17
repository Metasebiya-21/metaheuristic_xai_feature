# Benchmarking nature-inspired metaheuristics against SHAP for feature selection

Research-grade Python 3.10+ code comparing **genetic algorithm (DEAP)**, **binary PSO (PySwarms)**, and **simulated annealing** to a **SHAP (Tree Explainer) baseline** that keeps the *k* features (default 10) with the largest mean absolute SHAP on the [Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) dataset. Data are scaled, splits are **stratified**, and the base learner is a `RandomForestClassifier` with `random_state=42` as in the spec.

## Objective and fitness (minimize)

$$
f(S) = 0.9(1 - \text{Acc}) + 0.1\frac{|S|}{d}
$$

* $S$: selected feature set (binary mask)  
* $d$: number of features  
* Lower is better. All-zero masks are handled as the worst case (no training, accuracy 0, fitness 1).

## Setup

```bash
cd metaheuristic_xai
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional: limit parallelism inside `RandomForestClassifier` (e.g. HPC, CI, or to avoid over-subscription on laptops):

```bash
export SKLEARN_N_JOBS=1
```

## Run (default 30 paired splits; may take a long time)

```bash
python3 main.py
```

* Each run $i$ uses a **new stratified** `train_test_split(..., random_state=42 + i)`.
* **SHAP, GA, PSO, and SA share that split** (paired design); Wilcoxon signed-rank compares per-split accuracy differences.
* GA/PSO/SA and the tree models use the same `base_seed` pattern for stochastic engines.

## Quick smoke test (short schedul, one split)

```bash
python3 main.py --quick
```

Uses: one split, GA/PSO with **5** steps each, SA **150** steps, and writes plots and CSVs under `results/` and `plots/`.

## Fitness-weight ablation (GA, α ∈ {0.7, 0.8, 0.9})

```bash
python3 run_ablation.py --n-runs 10
```

Writes `results/ablation_alpha.csv` and `results/ablation_alpha_summary.csv` for the term paper.

## Rebuild term paper (after experiments)

```bash
cd report && make
```

## Outputs

| Path | Description |
|------|-------------|
| `results/all_runs.csv` | One row per (run, method): accuracy, \|S\|, fitness, wall time. |
| `results/summary.csv` | Per-method means and standard deviations. |
| `results/wilcoxon_vs_shap.csv` | **Wilcoxon signed-rank** (paired by `run_id`) on test **accuracy** vs SHAP, per metaheuristic. |
| `plots/shap_baseline_top_features.png` | Mean \|SHAP\| for the top-$k$ baseline (first run with `--quick` or run 0). |
| `plots/convergence_fitness.png` | Mean best fitness (± std) for GA, PSO, and SA. |
| `plots/bar_*.png` | Accuracy, feature count, and runtime (means). |

**Note:** for meaningful Wilcoxon inferences, use the default $n_\text{runs} \ge 5$; paper-grade studies typically use 30+ paired splits.

## Example console output (abbreviated, `--quick`)

The following is representative of a successful end-of-run log:

```
--quick: 1 data split, GA/PSO=5 steps, SA cap=150.
Finished 1 / 1 paired data splits.
Saved run-level CSV: results/all_runs.csv (4 rows).
...
SUMMARY (mean ± std) — n_runs=1 paired stratified splits (random_state=42+i; methods share each split)
 method  n_runs  accuracy_mean  ...
  SHAP       1         0.9474   ...
    GA       1         0.9737   ...
   PSO       1         0.9561   ...
    SA       1         0.9561   ...
```

## Algorithm details (as implemented)

* **SHAP:** Train RF → Tree Explainer on a **stratified** subsample of the training set → top-$k$ (default $k=10$) by mean $\|\text{SHAP}\|$.
* **GA (DEAP):** Binary, pop 50, 100 gen, tournament, two-point, bit flip, elitist replacement (`selBest(offspring + [elite], pop_size)`).  
* **BPSO (PySwarms):** Ring topology with $k = n-1$ neighbors, swarm 50, 100 iterations, cost history from the optimizer.  
* **SA:** Minimize fitness, single-random-bit neighbor,Metropolis, $T_0=100$, multiply by 0.95, stop when $T<10^{-3}$ (with a high step cap when `max_steps` is `None`).

## License

This repository code is for academic and research use; **cite scikit-learn, SHAP, DEAP, and PySwarms** when publishing.
