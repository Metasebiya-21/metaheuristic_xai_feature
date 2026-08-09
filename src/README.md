# src/

- `metaheuristic_xai/` — the installable package (see the layout table in the
  [top-level README](../README.md)).
- `notebook/benchmark.ipynb` — the full run/analysis workflow ([README](notebook/README.md)).
- `run_ablation.py` — GA fitness-weight ablation script.
- `__main__.py` — `python -m` entry point that calls `metaheuristic_xai.cli:main`.

Run the benchmark from the repository root with `python main.py` (see the
[top-level README](../README.md) for flags and outputs).
