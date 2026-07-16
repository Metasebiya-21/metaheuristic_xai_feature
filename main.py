#!/usr/bin/env python3
"""Redirect to the canonical benchmark in metaheuristic_xai/."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_CANONICAL = Path(__file__).resolve().parent / "metaheuristic_xai" / "main.py"

if __name__ == "__main__":
    if not _CANONICAL.is_file():
        print(f"Expected {_CANONICAL}", file=sys.stderr)
        raise SystemExit(1)
    runpy.run_path(str(_CANONICAL), run_name="__main__")
