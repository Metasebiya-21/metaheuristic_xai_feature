# benchmark.ipynb

One notebook, run top to bottom. Sections:

1. **Setup** — resolves `REPO_ROOT` from the installed package, so `RESULTS_DIR`
   (`<repo>/results`) and `PLOTS_DIR` (`<repo>/plots`) are the same no matter where
   Jupyter was launched. Prints the dataset / classifier / selector registries and
   the complexity summary.
2. **Run the benchmark** — a single `run_feature_selection`, then a seconds-long
   smoke matrix, then the full `run_full_benchmark` (edit `bench_cfg`). Writes
   per-run `.parquet` plus `all_runs.csv` / `summary.csv`. `resume=True` lets you
   stop and re-run the full cell.
3. **Sensitivity** (optional) — alpha and hyper-parameter sweeps.
4. **Analysis (leakage-controlled)** — loads `all_runs.csv` and builds the paper
   tables/figures: per-method summary, per-dataset breakdown, Friedman + Nemenyi
   critical-difference diagram, Holm-corrected Wilcoxon vs SHAP, search-overfitting
   (`val_accuracy - accuracy`) bars, selection-stability (Jaccard + Nogueira) bars,
   and the accuracy-vs-parsimony scatter. Figures land in `PLOTS_DIR`.

## Running

```bash
uv sync                        # installs jupyterlab (dev group)
uv run jupyter lab src/notebook
```

The same runner backs the CLI: `python src/main.py --quick --datasets wdbc` is
the smoke matrix from section 2.
