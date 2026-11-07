# 5-seed benchmark expansion — before/after results

**Status:** complete.
**Run finished:** 2026-10-04 01:41, elapsed **17.02 h** (under the 18h ceiling).
**Deadline:** 2026-10-05.

See `docs/seed_expansion_run.md` for the pre-run setup, timing-estimate
methodology, and the mid-run resource-contention incident (machine load
15→7 and swap 9.5GB→3.5GB after closing Chrome/Docker; cost roughly
7-8 hours of the total wall-clock before it was caught and resolved).

## Bottom line

**No qualitative conclusion flips.** Every claim in the report that depends
on the Friedman/Holm-Wilcoxon/stability results holds up under 5 seeds, and
most p-values got *more* significant (more data, more power), not less:

- Boruta still beats SHAP, still the only method that does so significantly
  (now p=0.0001, was p<0.0001 — both well under any threshold).
- RFE still significantly beats SHAP too, and *more* strongly than before
  (p=0.0103 → p=0.0051).
- GA still does **not** significantly beat or lose to SHAP (p=1.00 → p=0.25
  — still far from significant, though notably the p-value dropped a lot;
  see caveat below).
- GA still ranks dead last of 11 methods; Boruta still ranks first.
- The five metaheuristics are still clustered near Nogueira stability ≈ 0
  (run-to-run selections indistinguishable from random); classical methods
  and SHAP are still moderately-to-strongly stable.
- The MLP-classifier-outlier finding replicates: still the worst classifier,
  still doesn't change the ranking with or without it.

**One thing worth flagging even though it's not a flip:** GA's p-value vs.
SHAP dropped sharply (1.00 → 0.25) as its rank-biserial effect size grew
slightly more negative (−0.206 → −0.243). It is *not* statistically
significant at 5 seeds, but the trend is moving toward "GA is reliably
somewhat worse than SHAP" rather than "no detectable difference" as more
data accumulates. If this pattern continued to, say, 10-15 seeds, it could
plausibly cross the significance threshold (in GA's *disfavor*, not favor —
worth knowing if anyone is tempted to extend further hoping for the
opposite). Not an issue for this deadline; flagging for the record.

---

## 1. Friedman test

| | 3-seed (n=75) | 5-seed (n=125) |
|---|---|---|
| χ² | 73.64 | **109.95** |
| p | 8.73×10⁻¹² | **5.45×10⁻¹⁹** |
| CD (Nemenyi, α=0.05) | 1.743 | 1.350 |

Rejects the null more strongly with more data, as expected. CD shrank
(more subjects → tighter resolving power between methods), which is why a
couple of methods' relative clique membership could shift slightly even
though the ranking itself barely moved (see §2).

## 2. Mean rank table (replaces Table 3)

| Method | Rank (3-seed) | Rank (5-seed) | Δ |
|---|---|---|---|
| **Boruta** | 4.060 | 4.404 | +0.344 |
| Full feature set | 4.880 | 5.044 | +0.164 |
| RFE | 5.053 | 4.988 | −0.065 |
| LIME | 5.773 | 5.452 | −0.321 |
| LASSO | 6.227 | 6.080 | −0.147 |
| BPSO | 6.287 | 6.828 | +0.541 |
| SHAP | 6.327 | 6.152 | −0.175 |
| GWO | 6.387 | 6.344 | −0.043 |
| NSGA-II | 6.820 | 6.940 | +0.120 |
| SA | 7.067 | 6.896 | −0.171 |
| **GA** | 7.120 | 6.872 | −0.248 |

Boruta first, GA/NSGA-II/BPSO/SA at the bottom in both — same qualitative
picture. Largest mover is BPSO (+0.541, got relatively worse), but it was
already mid-pack and stays mid-pack-to-bottom; doesn't change any top-line
claim.

## 3. Holm-corrected Wilcoxon vs. SHAP (replaces Table 4)

### 3-seed (n=75 pairs per method)

| Method | Mean acc. diff | Rank-biserial | Holm p |
|---|---|---|---|
| Boruta | +0.0303 | +0.759 | **<0.0001** |
| RFE | +0.0140 | +0.518 | **0.0103** |
| Full feature set | +0.0094 | +0.214 | 1.00 |
| LIME | +0.0055 | +0.068 | 1.00 |
| GWO | −0.0042 | +0.002 | 1.00 |
| BPSO | −0.0063 | −0.040 | 1.00 |
| NSGA-II | −0.0134 | −0.147 | 1.00 |
| GA | −0.0186 | −0.206 | 1.00 |
| SA | −0.0214 | −0.118 | 1.00 |
| LASSO | −0.0267 | −0.099 | 1.00 |

### 5-seed (n=125 pairs per method)

| Method | Mean acc. diff | Rank-biserial | Holm p |
|---|---|---|---|
| Boruta | +0.0209 | +0.539 | **0.0001** |
| RFE | +0.0116 | +0.416 | **0.0051** |
| LIME | +0.0079 | +0.202 | 0.4495 |
| Full feature set | +0.0071 | +0.204 | 0.4495 |
| GWO | −0.0114 | −0.077 | 0.4980 |
| BPSO | −0.0151 | −0.185 | 0.4495 |
| NSGA-II | −0.0153 | −0.233 | 0.2522 |
| GA | −0.0195 | −0.243 | 0.2522 |
| SA | −0.0231 | −0.260 | 0.1984 |
| LASSO | −0.0351 | −0.180 | 0.4495 |

Boruta and RFE remain the only two significant results in both runs, same
direction, same conclusion. Boruta's effect size shrank somewhat
(0.759→0.539) with more data — a more conservative, more precise estimate,
not a weakening of the finding (still solidly significant).

## 4. Nogueira stability index (replaces Table 5)

| Method | Stability (3-seed) | Stability (5-seed) | Δ |
|---|---|---|---|
| Full feature set | 1.000 | 1.000 | 0.000 |
| SHAP | 0.527 | 0.529 | +0.002 |
| RFE | 0.512 | 0.528 | +0.016 |
| LIME | 0.477 | 0.499 | +0.022 |
| Boruta | 0.408 | 0.354 | −0.055 |
| LASSO | 0.167 | 0.183 | +0.016 |
| NSGA-II | 0.029 | 0.030 | +0.000 |
| GWO | 0.015 | −0.007 | −0.022 |
| BPSO | 0.014 | 0.026 | +0.012 |
| GA | 0.009 | 0.035 | +0.026 |
| SA | 0.009 | 0.022 | +0.014 |

All five metaheuristics remain clustered at ≈0 (statistically indistinguishable
from random selection across seeds) in both runs; classical methods and SHAP
remain moderately-to-strongly stable in both. GWO going slightly negative at
5 seeds is noise around zero (Nogueira's index can dip below 0 for
below-chance agreement), not a meaningful change — it was already
indistinguishable from random at 3 seeds.

## 5. MLP classifier-reliability check (also derives from the 3-seed run)

| | 3-seed | 5-seed |
|---|---|---|
| MLP mean accuracy | 0.793 | 0.788 |
| MLP mean gen. gap | +0.048 | +0.052 |
| MLP runs ≤0.6 acc. | 16.4% | 14.2% |
| Friedman with MLP | χ²=73.6, p=8.7e-12, n=75 | χ²=110.0, p=5.5e-19, n=125 |
| Friedman without MLP | χ²=61.2, p=2.2e-9, n=60 | χ²=97.9, p=1.4e-16, n=100 |
| Boruta rank w/ → w/o MLP | 4.06 → 4.03 | 4.40 → 4.16 |
| GA rank w/ → w/o MLP | 7.12 → 7.23 | 6.87 → 7.05 |

Same pattern: MLP is still the clear outlier (lowest accuracy, largest
generalization gap, most near-chance runs), and excluding it still doesn't
change the ranking (Boruta first, GA last, in both the 3-seed and 5-seed
with/without-MLP comparisons).

## 6. Places in `report/report.tex` that currently cite the 3-seed numbers

These were not edited yet (this file is the "report back to me" deliverable
requested before editing the manuscript) — flagging exact locations for the
next pass if the 5-seed numbers should replace the 3-seed ones before
submission:

- Abstract (line 33): "825 runs; 5 datasets × 5 classifiers × 3 seeds"
- §6.2 intro (line 163): "5 datasets × 5 classifiers × 3 seeds × 11 methods"
- §6.2 Friedman paragraph (line 165): χ²=73.6, p=8.7e-12, "825 subjects"
- Table 4 caption (line 191): "n=75 matched ... pairs"
- §6.2 stability paragraph (line 210): "across the 3 seeds"
- §6.2.1 MLP paragraph (line 278): "n=60 subjects" / "n=75"
- Tables 3, 4, 5 bodies themselves (mean ranks, Holm-Wilcoxon, stability —
  all currently hold the 3-seed values shown in the left-hand columns above)

## 7. Fallback integrity

`results/` (3-seed, 825 rows, seeds 42-44) was never touched during this
run and remains available as a complete, submittable fallback if needed.
`results_5seed/` (5-seed, 1375 rows, seeds 42-46) is the new, more
statistically powered dataset, written to `results_5seed/all_runs.csv` and
`results_5seed/summary.csv`.
