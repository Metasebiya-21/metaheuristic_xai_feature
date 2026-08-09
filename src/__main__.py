"""Module entry point: ``python -m metaheuristic_xai`` / ``python src``."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metaheuristic_xai.cli import main

if __name__ == "__main__":
    main()
