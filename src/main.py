#!/usr/bin/env python3
"""Thin wrapper: run the benchmark CLI from the ``src/`` directory.

The implementation lives in :mod:`metaheuristic_xai.cli`; this file only makes
``python src/main.py ...`` work without installing the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metaheuristic_xai.cli import main

if __name__ == "__main__":
    main()
