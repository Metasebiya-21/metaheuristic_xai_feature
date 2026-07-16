"""
Binary particle swarm optimization (PySwarms) for feature selection.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pyswarms.discrete import BinaryPSO

from src.fitness import evaluate_mask

logger = logging.getLogger(__name__)


def _scalarize_cost(c: np.ndarray) -> float:
    """One scalar global-best cost for logging/history from a possibly array cost."""
    a = np.asarray(c, dtype=np.float64)
    if a.size == 0:
        return float("inf")
    if a.shape == ():
        return float(a)
    return float(np.min(a))


def run_binary_pso(
    X_train: NDArray,
    X_test: NDArray,
    y_train: NDArray,
    y_test: NDArray,
    swarm_size: int = 50,
    iters: int = 100,
    base_seed: int = 0,
    options: Optional[Dict[str, float]] = None,
    acc_weight: float | None = None,
) -> dict[str, Union[NDArray[np.signedinteger], List[float], float, None]]:
    """
    Minimize the mask fitness with PySwarms :class:`BinaryPSO` (sigmoid thresholding
    in the library). The objective is evaluated in parallel for all particles; each
    position row is a binary 0/1 vector.

    The ring topology with ``k = n_particles - 1`` (maximum allowed) approximates
    a dense neighborhood so the swarm shares a single global best cost.

    Parameters
    ----------
    X_train, X_test, y_train, y_test
        Pre-scaled data.
    swarm_size, iters
        Defaults: 50 particles, 100 iterations.
    base_seed
        Seeded as ``42 + base_seed`` for :func:`numpy.random.seed` to diversify runs.
    options
        PSO hyper-parameters for ``c1, c2, w, k, p``; if ``None`` sensible defaults
        are used.
    """
    d = int(X_train.shape[1])
    s = 42 + int(base_seed)
    try:
        np.random.seed(s)
    except (ValueError, TypeError) as e:
        logger.warning("Could not seed numpy: %s", e)

    # PySwarms requires k to be a positive int strictly less than n_particles
    k = int(max(1, swarm_size - 1))
    if options is None:
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9, "k": k, "p": 2}

    def obj(pos: NDArray) -> NDArray[np.floating]:
        m = int(pos.shape[0])
        out = np.empty(m, dtype=np.float64)
        for j in range(m):
            p = pos[j]
            c, _a, _n, _i = evaluate_mask(
                p, X_train, X_test, y_train, y_test, acc_weight=acc_weight
            )
            out[j] = c
        return out

    t0 = time.perf_counter()
    try:
        opt = BinaryPSO(
            n_particles=swarm_size, dimensions=d, options=options, init_pos=None
        )
    except (KeyError, AssertionError) as e:
        logger.error("Failed to build BinaryPSO: %s", e)
        raise

    try:
        res = opt.optimize(obj, iters, verbose=False)
        _fc, best_pos = res[0], res[1]
    except (ValueError, RuntimeError) as e:
        logger.exception("Binary PSO optimization failed: %s", e)
        raise
    t1 = time.perf_counter()

    mask = np.round(best_pos).astype(np.int8)
    f, _acc, _n, _ = evaluate_mask(
        mask, X_train, X_test, y_train, y_test, acc_weight=acc_weight
    )

    conv: list[float] = []
    for step in opt.cost_history:
        conv.append(_scalarize_cost(step))

    if len(conv) < iters:
        logger.debug(
            "PSO cost_history length %d (may stop early on ftol).", len(conv)
        )

    return {
        "best_mask": mask,
        "best_fitness": float(f),
        "convergence": conv,
        "runtime": float(t1 - t0),
    }
