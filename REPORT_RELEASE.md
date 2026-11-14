# Release preparation report (v2.0.0)

Prepared 2026-11-14. **Nothing was committed, pushed or tagged.**
- `results/`, `results_5seed/`, `results_sweep/`, `results_convergence/` and `results_wdbc_features/` were only read and copied, never modified. The scripts that write into them were test-run only in a scratch sandbox.
- No manuscript (`main*.tex`) or `.bib` file was touched.

## 1. Inventory (step 1)

**(a) Git state**
- Branch `main`. HEAD is `18f2046d6f8f8269e0da79ce6d5ef59bbd6c4293` ("update .gitignore").
- The tree was **not clean** before this work. It had three untracked folders: `results_convergence/`, `results_sweep/` and `results_wdbc_features/`.
- No local tags exist. `gh` is not authenticated here, so I could not check whether a v1.0.0 release exists on GitHub.

**(b) `.gitignore`**
- Its patterns, in order: `*.pdf`, `*.aux`, `*.log`, `*.out`, `*.toc`, `*.bbl`, `*.blg`, `*.fls`, `*.fdb_latexmk`, `*.synctex.gz`, `*.synctex.gz(busy|idle|old)` (the last two groups appear twice), `.DS_Store`, `*.zip`, `.vscode`, `results`, `plots`, `report`, `.mpl`, `__pycache__`, `*.pyc`, `*.pyo`, `.venv`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`, `outputs`, `*.egg-info`, `build`, `dist`, `.claude`, `results_5seed`, `verify_report`. There is no trailing newline.
- The plain `git check-ignore -v` printed nothing for every path, because a directory containing tracked files is not reported. With `--no-index`:

| Path | Matched rule | Tracked files inside |
|---|---|---|
| `paper_results/` | not ignored | 0 |
| `results/` | `.gitignore:22:results` | 2 (`results/ablation/*.csv`) |
| `results_5seed/` | `.gitignore:38:results_5seed` | 3 (`all_runs.csv`, `summary.csv`, `aggregated/summary.csv`) |
| `results_sweep/` | **not ignored** (untracked) | 0 |
| `results_convergence/` | **not ignored** (untracked) | 0 |
| `results_wdbc_features/` | **not ignored** (untracked) | 0 |

- **No `.gitignore` change was needed**, because `paper_results/` is not ignored.
- The three untracked `results_*` folders are not ignored either, so a `git add -A` or `git add .` would sweep them in, including `results_sweep/runs/` (16 MB). The commands in §9 list paths explicitly for that reason.

**(c) Script locations**

| Script | Location | Tracked by git? |
|---|---|---|
| sweep.py, analyze.py | `results_sweep/` | no |
| convergence analyze.py, plot.py | `results_convergence/` | no |
| WDBC agreement | **no file**: it was run as inline commands, now recovered (§3) | — |
| make_paper_figures.py | `src/` | yes |
| compare_ga.py | repo root | yes |
| run_ablation.py | `src/` | yes |

**(d) Folder sizes**

| Folder | Size |
|---|---|
| `results/` | 23 MB |
| `results_5seed/` | 37 MB (32 MB of that is 1,375 parquet files) |
| `results_sweep/` | 31 MB (16 MB of that is `runs/`) |
| `results_convergence/` | 1.1 MB |
| `results_wdbc_features/` | 76 KB |
| `paper_results/` | did not exist |

The 20 largest files:

| Bytes | File |
|---|---|
| 14,025,874 | `results_sweep/run_runs.csv` |
| 2,485,597 | `results_5seed/raw/all_runs.csv` |
| 2,485,597 | `results_5seed/all_runs.csv` |
| 1,712,197 | `results/raw/all_runs.csv` |
| 1,712,197 | `results/all_runs.csv` |
| 800,388 | `results_convergence/curves.npz` |
| 730,043 | `results_sweep/run.log` |
| 597,208 | `results_sweep/probe_runs.csv` |
| 219,691 | `results_sweep/runs/sa_ionosphere_F1_budget_10000_seed44.json` |
| 219,379 | `results_sweep/runs/sa_leukemia_F1_budget_10000_seed44.json` |
| 219,336 | `results_sweep/runs/sa_ionosphere_F1_budget_10000_seed46.json` |
| 219,128 | `results_sweep/runs/sa_wdbc_F1_budget_10000_seed46.json` |
| 214,104 | `results_sweep/runs/sa_colon_F1_budget_10000_seed46.json` |
| 212,875 | `results_sweep/runs/sa_colon_F1_budget_10000_seed42.json` |
| 211,748 | `results_sweep/runs/sa_madelon_F1_budget_10000_seed43.json` |
| 211,334 | `results_sweep/runs/sa_wdbc_F1_budget_10000_seed42.json` |
| 211,242 | `results_sweep/runs/sa_leukemia_F1_budget_10000_seed45.json` |
| 211,105 | `results_sweep/runs/sa_leukemia_F1_budget_10000_seed42.json` |
| 210,979 | `results_sweep/runs/sa_leukemia_F1_budget_10000_seed46.json` |
| 210,779 | `results_sweep/runs/sa_wdbc_F1_budget_10000_seed44.json` |

## 2. What was copied (step 2)

`paper_results/` holds 30 data files plus `MANIFEST.md`, **18.8 MB in total**. The largest file is `sweep/sweep_runs.csv` at 14.0 MB. **No file exceeds 25 MB.**

| Folder | Contents | Source |
|---|---|---|
| `main_5seed/` | `all_runs.csv` (1,375 rows), `summary.csv` | `results_5seed/`. The `raw/` and `aggregated/` copies are byte-identical and were not duplicated. |
| `main_3seed/` | `all_runs.csv` (825 rows), `summary.csv` | `results/`, seeds 42–44. **They do exist.** The current manuscript's cost table uses the 5-seed runtimes, so these are for provenance only. |
| `nested_ga/` | `compare_ga_nested.csv`, `compare_ga.csv`, `ga_consensus_loso.csv` | `results/` |
| `sweep/` | `analysis_table.csv`, `analysis_output.txt`, `REPORT.md`, `step0_determinism.txt`, `step1_projection.txt`, `sweep_runs.csv` | `results_sweep/`. `sweep_runs.csv` is `run_runs.csv` renamed: one row per run, 1,025 rows. |
| `convergence/` | `REPORT.md`, `milestones_pooled.csv`, `milestones_by_dataset.csv`, `per_run_milestones.csv`, `convergence.png`, `verification.csv`, `analysis_output.txt` | `results_convergence/` |
| `wdbc_agreement/` | `REPORT.md`, `metrics_by_method.csv`, `metrics_seed_cluster_ci.csv`, `top10_features.csv`, `step*.txt` | `results_wdbc_features/` |
| `ga_ablation/` (**extra**, not in your layout) | `ablation_runs.csv`, `ablation_summary.csv` | `results/ablation/`. Table 4 in `main_v4.tex` (`tab:ablation`) is this GA α×budget ablation; see Missing Sources item 6. |

**Skipped**
- `results_5seed/*.parquet`: 32 MB, over the limit.
- `results_sweep/runs/`: 16 MB, the same data as `sweep_runs.csv`.
- `results/*.parquet`, `curves.npz`, `wdbc_masks.pkl` and `*.log`: not in the allowed file types.

## 3. Scripts (step 4)

`scripts/paper/` contains the following. No script was rewritten; the only edits are listed in §4.

| Path | Origin |
|---|---|
| `sweep/sweep.py`, `sweep/analyze.py` | `results_sweep/` |
| `convergence/analyze.py`, `convergence/plot.py` | `results_convergence/` (byte-identical copies) |
| `convergence/gen20_and_rerun_cost.py` | **recovered verbatim** from two inline commands in the session transcript. They produced the last 19 lines of `analysis_output.txt`: the GA generation-20 table and the rerun-cost estimate. |
| `wdbc_agreement/step0_names.py`, `step1_verify.py`, `step2_metrics.py`, `step3_seed_cluster.py` | **recovered verbatim** from the four inline commands, with timestamps, that produced `results_wdbc_features/`. |
| `figures/make_paper_figures.py` | `src/make_paper_figures.py` (the original stays in `src/`) |
| `nested_ga/compare_ga.py` | repo root (byte-identical copy; the original is still tracked at the root) |

**Paths changed to be relative to the repo root.** No script had an absolute path. Three had paths relative to their **own** location, which broke after moving:
1. `scripts/paper/sweep/sweep.py`: `ROOT = Path(__file__).resolve().parent.parent` became `.parents[3]`.
2. `scripts/paper/sweep/analyze.py`: the same change.
3. `scripts/paper/figures/make_paper_figures.py`: `_SRC = Path(__file__).resolve().parent` and `ROOT = _SRC.parent` became `ROOT = .parents[3]` and `_SRC = ROOT / "src"`.

The convergence, WDBC and `compare_ga.py` scripts already used cwd-relative paths (`results_5seed/...`, `src`), so they run from the repo root unchanged.

**Verification.** Every script was run from the repo root, or from a scratch sandbox that symlinked `src/` and `results_5seed/` so that no original was overwritten. Each was compared with the shipped outputs:

| Script | Result |
|---|---|
| sweep `analyze.py` | `analysis_table.csv` **byte-identical**. The output text differs only in the fixed label. |
| convergence `analyze.py` + `gen20_and_rerun_cost.py` + `plot.py` | all 7 outputs **byte-identical**, including `convergence.png` |
| WDBC steps 0–3 | all 8 outputs **byte-identical**, including `wdbc_masks.pkl` |
| `make_paper_figures.py` | `cd_diagram.png`, `generalization_gap.png` and `stability.png` **byte-identical** to `report/panafricon2026/figures/` |
| `sweep.py` | `--help` works. Only the step-5 cells were rerun (§5). |
| `compare_ga.py` | **not run**: it is a long job, and it would overwrite `results/compare_ga_nested.csv` |

**Scripts the paper's numbers depend on that I could NOT find:**
- `run_5seed_benchmark.py`: the wrapper used for the 5-seed run. `python main.py ... --results-dir results_5seed` is the equivalent command, documented in REPRODUCE.md.
- The script that built **Table 2** (ranks, Wilcoxon/Holm, rank-biserial, Nogueira) and **Table 3** (cost). The building blocks are `src/metaheuristic_xai/evaluation.py` functions, but no table-assembly script exists on disk.
- The script that produced `results/compare_ga.csv` and `results/ga_consensus_loso.csv`. `compare_ga.py` writes only `compare_ga_nested.csv`, and git history shows no other version.
- The script that produced `results/alpha_sensitivity/alpha_sensitivity.parquet` (the "eight GA traces" check).

## 4. Code changes (full diffs)

### 4.1 `analyze.py` label fix, and paths in the script copies

```diff
--- results_sweep/sweep.py	2026-11-11 14:53:49
+++ scripts/paper/sweep/sweep.py	2026-11-14 10:34:58
@@ -22,7 +22,7 @@
 
 os.environ.setdefault("OMP_NUM_THREADS", "1")
 os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
-ROOT = Path(__file__).resolve().parent.parent
+ROOT = Path(__file__).resolve().parents[3]  # repo root (was .parent.parent when this lived in results_sweep/)
 sys.path.insert(0, str(ROOT / "src"))
 
 import numpy as np
@@ -75,21 +75,15 @@
     from metaheuristic_xai.config import DEFAULT_METHOD_KWARGS
     from metaheuristic_xai.datasets import load_dataset
     from metaheuristic_xai import runner
-    import metaheuristic_xai.selectors.metaheuristic as sm
-    from metaheuristic_xai.algorithms.pso import run_binary_pso
 
-    # BPSOSelector does not expose PySwarms `options`; patch it in THIS process only (no code file changed).
-    if bpso_options is not None:
-        opts = dict(bpso_options)
-        sm.run_binary_pso = lambda *a, **k: run_binary_pso(
-            *a, **{**k, "options": {**opts, "k": int(max(1, k.get("swarm_size", POP) - 1))}})
-    else:
-        sm.run_binary_pso = run_binary_pso
     bundle = load_dataset(dataset, random_state=seed)
     kw = {**DEFAULT_METHOD_KWARGS[method], **override}
+    # BPSO c1/c2/w go through BPSOSelector's constructor arguments (k and p are fixed there);
+    # they are kept out of `kw` so the recorded method_kwargs match the original sweep rows.
+    pso = {k: bpso_options[k] for k in ("c1", "c2", "w")} if bpso_options is not None else {}
     t0 = time.perf_counter()
     rec = runner.run_feature_selection(bundle, "random_forest", method, seed=seed, alpha=alpha,
-                                       method_kwargs=kw, defaults=DEFAULT_METHOD_KWARGS, val_size=0.25)
+                                       method_kwargs={**kw, **pso}, defaults=DEFAULT_METHOD_KWARGS, val_size=0.25)
     wall = time.perf_counter() - t0
     conv = rec.get("convergence_history")
     return {"method": method, "dataset": dataset, "factor": factor, "level": level, "seed": seed,
--- results_sweep/analyze.py	2026-11-11 14:53:49
+++ scripts/paper/sweep/analyze.py	2026-11-14 10:33:32
@@ -19,7 +19,7 @@
 import pandas as pd
 from scipy.stats import wilcoxon
 
-ROOT = Path(__file__).resolve().parent.parent
+ROOT = Path(__file__).resolve().parents[3]  # repo root (was .parent.parent when this lived in results_sweep/)
 sys.path.insert(0, str(ROOT / "src"))
 from metaheuristic_xai.evaluation import _nogueira_stability  # noqa: E402
 
@@ -98,7 +98,7 @@
     T.to_csv(os.environ.get("SWEEP_TABLE_OUT", OUT / "analysis_table.csv"), index=False)
     pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
     print(T.round(4).to_string(index=False))
-    print("\nSHAP mean acc on these 15 cells:", round(ref["shap"].accuracy.mean(), 4),
+    print("\nSHAP mean acc on these 25 cells (Random Forest, 5 datasets x 5 seeds):", round(ref["shap"].accuracy.mean(), 4),
           "| Boruta:", round(ref["boruta"].accuracy.mean(), 4))
     print("\nconfigs above SHAP:", T[T.above_SHAP][["method", "factor", "level", "acc", "d_vs_shap", "p_vs_shap"]].to_string(index=False))
     print("\nconfigs within 0.005 of Boruta:", T[T["within_0.005_of_Boruta"]][["method", "factor", "level", "acc", "d_vs_boruta", "p_vs_boruta"]].to_string(index=False))
--- src/make_paper_figures.py	2026-11-10 16:25:55
+++ scripts/paper/figures/make_paper_figures.py	2026-11-14 10:33:32
@@ -8,7 +8,8 @@
 import sys
 from pathlib import Path
 
-_SRC = Path(__file__).resolve().parent
+ROOT = Path(__file__).resolve().parents[3]  # repo root (was src/ when this lived in src/)
+_SRC = ROOT / "src"
 sys.path.insert(0, str(_SRC))
 
 import pandas as pd
@@ -16,7 +17,6 @@
 from metaheuristic_xai.evaluation import friedman_nemenyi
 from metaheuristic_xai.plots import plot_cd_diagram, plot_generalization_gap_bars, plot_stability_bars
 
-ROOT = _SRC.parent
 
 
 def main() -> None:
```

The three other copies (`convergence/analyze.py`, `convergence/plot.py`, `nested_ga/compare_ga.py`) are byte-identical to their sources.

### 4.2 `BPSOSelector`: c1, c2 and w exposed (`src/metaheuristic_xai/selectors/metaheuristic.py`)

```diff
diff --git a/src/metaheuristic_xai/selectors/metaheuristic.py b/src/metaheuristic_xai/selectors/metaheuristic.py
index 5877ff5..309b5c3 100644
--- a/src/metaheuristic_xai/selectors/metaheuristic.py
+++ b/src/metaheuristic_xai/selectors/metaheuristic.py
@@ -122,11 +122,15 @@ class BPSOSelector(FeatureSelector):
         iters: int = 100,
         oracle: Optional[EvaluationOracle] = None,
         random_state: int = 42,
+        c1: float = 0.5,
+        c2: float = 0.3,
+        w: float = 0.9,
         **kwargs: Any,
     ) -> None:
         super().__init__("bpso", oracle=oracle, random_state=random_state, **kwargs)
         self.swarm_size = swarm_size
         self.iters = iters
+        self.c1, self.c2, self.w = c1, c2, w
 
     def fit(
         self, X_train: NDArray[np.floating], y_train: NDArray[np.signedinteger]
@@ -141,6 +145,8 @@ class BPSOSelector(FeatureSelector):
             swarm_size=self.swarm_size,
             iters=self.iters,
             base_seed=self.random_state,
+            # same as run_binary_pso's built-in defaults when c1/c2/w are left at their defaults
+            options={"c1": self.c1, "c2": self.c2, "w": self.w, "k": int(max(1, self.swarm_size - 1)), "p": 2},
             oracle=orc,
         )
 
```

- **Defaults are equal to today's values**: c1 0.5, c2 0.3, w 0.9, k = swarm − 1, p 2. That is the same dict `run_binary_pso` builds when `options=None` (`algorithms/pso.py:70–72`).
- `k` and `p` stay fixed, since they are not exposed.
- No call site needed changing. The selector registry passes `**method_kwargs` through, so `{"w": 0.45}` in `method_kwargs` now reaches the constructor.

### 4.3 README.md

- Added a **Paper** section with the title, pointers to `paper_results/`, `scripts/paper/` and REPRODUCE.md, and the v1.0.0 warning.
- Replaced the "Fitness-weight x budget ablation" section (`src/run_ablation.py`, GA-only α×budget on 3 datasets) with a description of the sweep actually used.
- **Note:** the line you described ("--n-runs 10, WDBC-only GA alpha sweep") **is not in the current README**, nor in its git history. The closest outdated line was the `run_ablation.py` section, which is what I replaced.

## 5. BPSO inertia-weight patch (step 5)

**(a) The original patch**, at `results_sweep/sweep.py:81–87`:

```python
    # BPSOSelector does not expose PySwarms `options`; patch it in THIS process only (no code file changed).
    if bpso_options is not None:
        opts = dict(bpso_options)
        sm.run_binary_pso = lambda *a, **k: run_binary_pso(
            *a, **{**k, "options": {**opts, "k": int(max(1, k.get("swarm_size", POP) - 1))}})
    else:
        sm.run_binary_pso = run_binary_pso
```

**(b)** The change was applied as shown in §4.2. `scripts/paper/sweep/sweep.py` now passes c1, c2 and w through `method_kwargs` instead of patching. They are kept out of the *recorded* `method_kwargs` so that output rows stay identical to the originals.

**(c) Verification.** It was run from the repo root, using `run_cell` imported from the rewritten `scripts/paper/sweep/sweep.py`:

```
bpso  default: acc 0.929824561404 vs 0.929824561404 |d|=0.0e+00  |S| 6 vs 6  same indices: True
ga    default: acc 0.956140350877 vs 0.956140350877 |d|=0.0e+00  |S| 6 vs 6  same indices: True
sweep row bpso/wdbc/w=0.45/seed42: fields compared 18 | differing: none | acc 0.9473684210526315 vs 0.9473684210526315, |S| 8 vs 8
ALL OK
```

- Tolerance was 1e-9; every difference was exactly 0.
- The sweep-row check compared all 18 stored fields except the two timing fields: accuracy, |S|, gap, selected indices, validation and test fitness, lookups, cache hits, the full convergence history, and the recorded kwargs and options.
- **All comparisons passed, so the code change was kept.** No `bpso_w_patch.py` fallback was needed.

## 6. Tests (step 7)

```bash
python -m pytest -o addopts="" tests/test_leakage.py tests/test_selectors.py -q
```

Result: **17 passed** in 53 s.
- `-o addopts=""` was needed because `pyproject.toml` adds `--cov` options and `pytest-cov` is not installed in `.venv`.
- No new tests were written.

## 7. MANIFEST summary and MISSING SOURCES

`paper_results/MANIFEST.md` lists 30 files (18,770,340 bytes) with sizes, SHA-256 hashes, CSV row counts and the table or figure each one feeds.

**MISSING SOURCES** (copied from the manifest):
1. **Table 2 statistics** (Friedman χ², mean ranks, CD, Holm-corrected Wilcoxon vs SHAP, rank-biserial, Nogueira) have no output file on disk. They are computed on the fly from `main_5seed/all_runs.csv`, and no script that assembles the table exists. The 5-seed run produced no `feature_stability.csv` or `wilcoxon_results.csv`.
2. **Table 3** inputs are in `main_5seed/all_runs.csv`. The script that assembled the table was not found.
3. **Figures 3 and 4 and the CD diagram** are rebuilt by `scripts/paper/figures/make_paper_figures.py` (byte-identical output). The PNGs themselves live only in the gitignored `report/` and are not in `paper_results/`.
4. **`compare_ga.csv` and `ga_consensus_loso.csv`** have no producing script. Table 5 itself is backed by `compare_ga_nested.csv` and `compare_ga.py`.
5. **The "eight GA convergence traces" sentence** (main_v4 §4.3) comes from `results/alpha_sensitivity/alpha_sensitivity.parquet`, which was not copied because it is a parquet file.
   - **It is dated 2026-09-10, before the leakage-corrected outputs of 2026-09-12. Check that it used the corrected oracle, or drop the sentence; the Figure 5 convergence analysis supersedes it.**
6. **Table 4 mismatch.** In `main_v4.tex`, Table 4 (`tab:ablation`) is the GA α×budget ablation (`paper_results/ga_ablation/`), not the sweep. Your map assigns Table 4 to the sweep, perhaps for a newer manuscript. Both are included.
7. **Table 6 and §4.7** do not exist in `main_v4.tex`. Your map was followed as given.
8. **Rerunning the convergence and WDBC analyses** needs `results_5seed/*.parquet`, which is not in git. Without it, the main benchmark must be rerun first.

## 8. Size, secrets, licence

**Size added to git**

| Item | Count | Size |
|---|---|---|
| New files | 46: 31 in `paper_results/`, 11 in `scripts/paper/`, plus `REPRODUCE.md`, `REPORT_RELEASE.md`, `CITATION.cff`, `.zenodo.json` | 18.8 MB (18,848,300 bytes) |
| Modified tracked files | 3: `README.md`, `pyproject.toml`, `src/metaheuristic_xai/selectors/metaheuristic.py` | — |
| New licence file | `LICENSE` (Apache-2.0) | 11 KB |

The largest file is `paper_results/sweep/sweep_runs.csv` at 14.0 MB.

**Secrets and absolute paths**
- **None in any file to be added.** I grepped for `/Users/`, `/home/`, `/private/tmp`, the username, and key, token, password and private-key patterns.
- Absolute home paths exist only in `results_sweep/run.log` and `probe.log`, which were not copied.
- `pyproject.toml` (already tracked; only its licence line changed) contains a work email address: `metasebiya@eaglelionsystems.com`. That is not a secret, but note it is public.

**Licence: you chose Apache-2.0 (2026-11-14), and it has been applied.** The sources before the change disagreed:

| Source | Says |
|---|---|
| `LICENSE` file | **does not exist** |
| `pyproject.toml` | `license = { text = "Proprietary" }` |
| `README.md` | "Academic / research use; cite scikit-learn, SHAP, LIME, DEAP, PySwarms, Boruta." No licence named. |
| `CITATION.cff`, `.zenodo.json` | did not exist (created now, with **no licence field**) |

- "Proprietary" conflicts with an open Zenodo release and with the README's research-use wording.
- **Applied:**
  - `LICENSE` added, with the canonical Apache License 2.0 text (201 lines, copied unchanged from an installed package).
  - `pyproject.toml`: `license = { text = "Apache-2.0" }`.
  - `CITATION.cff`: `license: Apache-2.0`.
  - `.zenodo.json`: `"license": "Apache-2.0"`.
  - README notes line now reads: "Licensed under the Apache License 2.0 (see `LICENSE`)".

## 9. Commands for you to run (none were run)

```bash
git add paper_results scripts/paper REPRODUCE.md REPORT_RELEASE.md CITATION.cff .zenodo.json LICENSE README.md pyproject.toml src/metaheuristic_xai/selectors/metaheuristic.py
```

```bash
git status --short   # check: nothing from results_sweep/, results_convergence/ or results_wdbc_features/ should be staged
```

```bash
git commit -m "Release prep v2.0.0: paper_results, paper scripts, REPRODUCE, citation metadata; expose BPSO c1/c2/w; Apache-2.0 licence"
```

```bash
git push origin main
```

```bash
git tag -a v2.0.0 -m "v2.0.0: leakage-corrected GA-XAI benchmark and paper analyses"
```

```bash
git push origin v2.0.0
```

**Zenodo archives only when a GitHub *release* is published from the tag. Pushing the tag alone does nothing on Zenodo.**
- After pushing the tag, create the release on GitHub (Releases → Draft a new release → choose `v2.0.0` → Publish), or with `gh release create v2.0.0 --title "v2.0.0" --notes-file REPORT_RELEASE.md` once `gh` is authenticated.
- The Zenodo–GitHub integration must be switched on for this repository beforehand.
- Zenodo's metadata, including the Apache-2.0 licence, is taken from `.zenodo.json` at release time.
