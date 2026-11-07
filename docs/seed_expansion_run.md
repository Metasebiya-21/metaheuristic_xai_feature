# 5-seed benchmark expansion — run log

**Status:** run in progress, started 2026-10-03.
**Deadline:** 2026-10-05.

## 1. What changed

`BENCHMARK_CONFIG.seeds` in `src/metaheuristic_xai/config.py` changed from
`[42, 43, 44]` (3 seeds) to `[42, 43, 44, 45, 46]` (5 seeds). All other fields
unchanged (5 datasets, 5 classifiers, 11 methods, `alpha=0.8` nominal).

The unused 15-seed `PAPER_CONFIG` (`seeds=list(range(42, 57))`) was **not**
touched and was never run (`results/paper_matrix/` does not exist on disk) —
it is a separate, much larger, narrower-scope config (3 datasets, 3
classifiers, 8 methods) and out of scope for this run.

**Correction to the original ask:** the existing 3-seed/825-run output lives
directly in `results/` (`all_runs.csv`, `summary.csv`, ~825 per-cell
`.parquet` files), not in `results/paper_matrix/` (which doesn't exist).
`results/` is the directory being protected as the fallback.

## 2. Fallback protection

- `results/` (3-seed, 825 rows, seeds 42/43/44 only) is **untouched** —
  verified before starting the new run (825 rows confirmed via pandas,
  `seed.value_counts()` shows only 42/43/44).
- The new run writes to **`results_5seed/`**, a fresh, separate directory.
  `run_full_benchmark` was called with `resume=True` against this empty
  directory, so it computes all 1375 cells fresh rather than resuming from
  (or overwriting) anything in `results/`.

## 3. Timing estimate (before starting the full run)

Two independent estimates were combined rather than relying on either alone:

**a) Historical, data-driven estimate** — the existing 3-seed run already
recorded `runtime_seconds` per cell (825 rows). Summing and scaling by 5/3
(no new compute needed):

| | value |
|---|---|
| 3-seed total compute (sum of `runtime_seconds`) | 24.27 compute-hours |
| 5-seed total compute (scaled ×5/3, 1375 cells) | 40.46 compute-hours |
| Ideal wall-clock @ `n_jobs=6` | 6.74 h |

**b) Live probe (as requested)** — ran the single fastest dataset/classifier
combination (`ionosphere` / `lightgbm`, identified from (a) as the lowest
total-runtime combo) across all 5 new seeds × 11 methods = 55 cells, fresh,
timed end-to-end:

| | value |
|---|---|
| Cells | 55 |
| Observed wall-clock | 171.3 s |
| Naive-parallel ideal (historical compute / 6) | 74.4 s |
| Implied per-task overhead (dataset load, dispatch, I/O) | 10.57 s/task (serial-equivalent) |

The gap between observed and ideal is per-task fixed overhead (dataset
loading, joblib dispatch, parquet write), not compute — it does not scale
with how expensive an individual cell's search is, so it was modeled as an
**additive** term, not a multiplier on the whole estimate:

overhead wall-clock contribution (full run, 1375 cells) = 10.57 s × 1375 / 6
≈ 0.67 h

**Combined estimate: 6.74 h + 0.67 h ≈ 7.4 h wall-clock**, comfortably under
the 18-hour ceiling, with the heaviest cells (e.g. Madelon GA/BPSO, up to
~1796s single-threaded) dispatched heaviest-first by the runner to minimize
tail-latency risk near the end of the run.

Decision: **proceed** (estimate well under 18h threshold).

## 4. Run launch

- Command: `python3 run_5seed_benchmark.py` (wraps
  `run_full_benchmark(config, resume=True, n_jobs=6)` +
  `save_results(df, results_5seed/)`), run via `nohup ... &` + `disown`,
  with `caffeinate -i -w <pid>` pinning the machine awake for the run's
  duration (this repo's established convention for surviving terminal
  disconnection on macOS).
- Log: `run_5seed_benchmark.log` (scratchpad directory this session; see
  "How to check progress" below for a durable path if needed).
- Output: `results_5seed/` (per-cell `.parquet` files during the run;
  `all_runs.csv` + `summary.csv` written once at the very end by
  `save_results`).

### How to check progress

```bash
# count completed cells so far (out of 1375)
ls results_5seed/*.parquet 2>/dev/null | wc -l

# tail the live log
tail -f <scratchpad>/run_5seed_benchmark.log
```

### How to confirm it's finished

```bash
grep RUN_5SEED_DONE <scratchpad>/run_5seed_benchmark.log
ls results_5seed/all_runs.csv results_5seed/summary.csv
```

The log ends with `elapsed <N> h` followed by `RUN_5SEED_DONE` on success.

## 5. Before/after statistical comparison

**Pending** — to be filled in once the run completes:
Friedman χ²/p, mean-rank table (replaces Table 3), Holm-Wilcoxon-vs-SHAP
table (replaces Table 4, n=125 instead of n=75), Nogueira stability
(Table 5), and a flag on whether any qualitative conclusion changes.
