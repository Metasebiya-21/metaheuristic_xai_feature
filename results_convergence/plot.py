"""Convergence figure: median fraction of total improvement vs fraction of nominal budget, IQR band."""
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOTAL = {"ga": 2050, "bpso": 2000, "sa": 2001, "gwo": 2050}
# Okabe-Ito colorblind-safe palette
STYLE = {"ga": ("GA", "#0072B2"), "bpso": ("BPSO", "#E69F00"), "sa": ("SA", "#009E73"), "gwo": ("GWO", "#CC79A7")}


def evals_at(m, n):
    i = np.arange(n)
    return {"ga": 50 + 50 * (i + 1), "bpso": 50 * (i + 1), "sa": i + 1, "gwo": 50 * (i + 1)}[m]


grid = np.linspace(0, 1, 401)
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})
fig, ax = plt.subplots(figsize=(5.0, 3.4))
for m, (lab, col) in STYLE.items():
    cs = []
    for f in sorted(glob.glob(f"results_5seed/*_{m}_seed*.parquet")):
        h = np.asarray(pd.read_parquet(f).convergence_history.iloc[0], float)
        tot = h[0] - h[-1]
        if tot <= 1e-12:
            continue  # flat runs excluded, as in the tables
        x = evals_at(m, len(h)) / TOTAL[m]
        cs.append(np.interp(grid, x, (h[0] - h) / tot, left=np.nan))  # no value before the first entry
    C = np.array(cs)
    q25, q50, q75 = np.nanpercentile(C, [25, 50, 75], axis=0)
    ax.fill_between(grid, q25, q75, color=col, alpha=0.18, lw=0)
    ax.plot(grid, q50, color=col, lw=1.6)
    # direct labels, placed by hand in clear space next to each median curve
    x, y, ha = {"sa": (0.055, 0.62, "right"), "gwo": (0.15, 0.645, "left"),
                "ga": (0.33, 0.56, "left"), "bpso": (0.52, 0.40, "left")}[m]
    ax.text(x, y, lab, color=col, fontsize=9, fontweight="bold", ha=ha, va="center")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Fraction of nominal evaluation budget (about 2,000 lookups)")
ax.set_ylabel("Fraction of total improvement")
ax.grid(alpha=0.25, lw=0.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("results_convergence/convergence.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
print("saved")
