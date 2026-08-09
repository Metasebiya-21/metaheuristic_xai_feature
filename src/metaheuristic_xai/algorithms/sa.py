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

from metaheuristic_xai.oracle import EvaluationOracle

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
    *,
    oracle: EvaluationOracle,
) -> dict[str, Union[NDArray[np.signedinteger], List[float], float]]:
    """
    **Simulated annealing** on a binary string of length ``d`` (flip one random bit
    as neighbour, Metropolis rule). After each move ``T <- max(min_temp, T * cooling)``.
    The search runs for ``max_steps`` candidate evaluations.

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
        Markov-step budget -- SA runs for exactly this many candidate evaluations
        (plus the initial one). ``cooling`` sets the *shape* of the temperature
        decay; once ``T`` reaches ``min_temp`` it is held there (the tail is greedy
        local search) instead of terminating early, so the search actually spends
        its budget. If ``None``, a generous ``max(10_000, 100 * d)`` cap is used.

    Note
    ----
    Previously the loop also stopped as soon as ``T <= min_temp``, which -- with the
    default schedule -- ended the search after ~100-220 steps regardless of
    ``max_steps``, giving SA a far smaller evaluation budget than the other
    metaheuristics. It now honours ``max_steps``.
    """
    d = int(X_train.shape[1])
    s = 42 + int(base_seed)
    rng = np.random.default_rng(s)

    current = rng.integers(0, 2, size=d, dtype=np.int8)
    t0 = time.perf_counter()
    f_cur = float(oracle.evaluate(current).fitness)
    n_evaluations = oracle.total_evaluations
    best = current.copy()
    f_best: float = float(f_cur)
    t = float(initial_temp)
    step_cap = int(max(1, max_steps)) if max_steps is not None else max(10_000, 100 * d)

    conv: list[float] = [f_best]
    try:
        for _step in range(step_cap):
            j = int(rng.integers(0, d))
            cand = current.copy()
            cand[j] = 1 - cand[j]
            f_cand = float(oracle.evaluate(cand).fitness)
            n_evaluations = oracle.total_evaluations
            if _metropolis(f_cur, f_cand, t, rng):
                current, f_cur = cand, float(f_cand)
            if f_cur < f_best or (
                f_cur == f_best and int(np.sum(current)) < int(np.sum(best))
            ):
                f_best, best = f_cur, current.copy()
            conv.append(f_best)
            t = max(min_temp, t * float(cooling))
    except (ValueError, TypeError) as e:
        logger.error("SA failed during search: %s", e)
        raise

    t1 = time.perf_counter()
    best_fitness = float(oracle.evaluate(best).fitness)
    n_evaluations = oracle.total_evaluations
    return {
        "best_mask": best,
        "best_fitness": best_fitness,
        "convergence": conv,
        "runtime": float(t1 - t0),
        "n_evaluations": n_evaluations,
    }
