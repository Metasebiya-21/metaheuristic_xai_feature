"""
Simulated annealing from scratch for binary feature masks (minimize fitness).
"""

from __future__ import annotations

import logging
import math
import time
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from src.fitness import evaluate_mask

logger = logging.getLogger(__name__)


def _metropolis(
    d_cur: float,
    d_cand: float,
    temperature: float,
    rng: np.random.Generator,
) -> bool:
    """
    Accept if the candidate is better, else accept with probability
    :math:`\\exp(-(E_{\\text{new}} - E_{\\text{cur}}) / T)` for a minimization.
    """
    if d_cand < d_cur:
        return True
    if temperature <= 0.0 or not math.isfinite(temperature):
        return False
    p = math.exp((d_cur - d_cand) / temperature)
    return float(rng.random()) < p


def run_simulated_annealing(
    X_train: NDArray,
    X_test: NDArray,
    y_train: NDArray,
    y_test: NDArray,
    initial_temp: float = 100.0,
    cooling: float = 0.95,
    min_temp: float = 1e-3,
    base_seed: int = 0,
    max_steps: Optional[int] = None,
    acc_weight: float | None = None,
) -> dict[str, Union[NDArray[np.signedinteger], List[float], float]]:
    """
    **Simulated annealing** on a binary string of length ``d`` (one flips a single
    random bit as neighbor, Metropolis rule). After each move (accepted or not),
    temperature is multiplied by ``cooling`` until it drops below ``min_temp`` or a
    safety cap on the number of steps (``10 * d * ceil(-log_10 min_temp)``) is hit.

    * State: :math:`d` bits; neighbor: flip exactly one index uniformly
    * Energy: the scalar fitness to minimize
    * ``convergence`` stores the **best so far** after each move

    Parameters
    ----------
    initial_temp, cooling, min_temp
        Annealing schedule as specified; ``T <- T * cooling`` per step.
    base_seed
        :class:`numpy.random.Generator` seed as ``42 + base_seed`` for a run.
    max_steps
        If set, stop after this many Markov steps (in addition to ``T > min_temp``).
        If ``None`` (default), a generous cap of ``max(10_000, 100 * d)`` is used
        to avoid runaway on slow cooling.
    """
    d = int(X_train.shape[1])
    s = 42 + int(base_seed)
    rng = np.random.default_rng(s)

    current = rng.integers(0, 2, size=d, dtype=np.int8)
    t0 = time.perf_counter()
    f_cur, _acc, _n, _i = evaluate_mask(
        current, X_train, X_test, y_train, y_test, acc_weight=acc_weight
    )
    best = current.copy()
    f_best: float = float(f_cur)
    t = float(initial_temp)
    step_cap: int
    if max_steps is not None:
        step_cap = int(max(1, max_steps))
    else:
        step_cap = max(10_000, 100 * d)

    conv: list[float] = [f_best]
    try:
        steps = 0
        while t > min_temp and steps < step_cap:
            j = int(rng.integers(0, d))
            cand = current.copy()
            cand[j] = 1 - cand[j]
            f_cand, _a2, _n2, _i2 = evaluate_mask(
                cand, X_train, X_test, y_train, y_test, acc_weight=acc_weight
            )
            if _metropolis(f_cur, f_cand, t, rng):
                current, f_cur = cand, float(f_cand)
            if f_cur < f_best or (
                f_cur == f_best and int(np.sum(current)) < int(np.sum(best))
            ):
                f_best, best = f_cur, current.copy()
            conv.append(f_best)
            t *= float(cooling)
            steps += 1
    except (ValueError, TypeError) as e:
        logger.error("SA failed during search: %s", e)
        raise

    t1 = time.perf_counter()
    return {
        "best_mask": best,
        "best_fitness": float(
            evaluate_mask(
                best, X_train, X_test, y_train, y_test, acc_weight=acc_weight
            )[0]
        ),
        "convergence": conv,
        "runtime": float(t1 - t0),
    }
