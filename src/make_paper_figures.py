#!/usr/bin/env python3
"""Regenerate the PanAfriCon 2026 paper figures (CD diagram, generalization gap,
selection stability) from the 5-seed results, without titles (captions carry them)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

import pandas as pd

from metaheuristic_xai.evaluation import friedman_nemenyi
from metaheuristic_xai.plots import plot_cd_diagram, plot_generalization_gap_bars, plot_stability_bars

ROOT = _SRC.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=ROOT / "results_5seed" / "all_runs.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT / "report" / "panafricon2026" / "figures")
    args = p.parse_args()

    df = pd.read_csv(args.runs)
    fn = friedman_nemenyi(df)
    assert fn.n_subjects == 125 and abs(fn.cd - 1.35) < 0.005, (fn.n_subjects, fn.cd)  # matches the paper

    plot_cd_diagram(fn.mean_ranks, fn.cd, args.out_dir / "cd_diagram.png", title=None, dpi=300)
    plot_generalization_gap_bars(df, args.out_dir / "generalization_gap.png", title=None, dpi=300)
    plot_stability_bars(df, args.out_dir / "stability.png", title=None, dpi=300)


if __name__ == "__main__":
    main()
