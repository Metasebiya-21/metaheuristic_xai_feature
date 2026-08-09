"""
Binary Grey Wolf Optimizer (GWO) for feature selection.

Tracks Alpha, Beta, Delta wolves and updates positions using sigmoidal transfer function
to optimize feature selection masks via EvaluationOracle.
"""

from __future__ import annotations

import logging
import time
from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from metaheuristic_xai.oracle import EvaluationOracle

logger = logging.getLogger(__name__)


def sigmoidal_transfer(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Sigmoidal transfer function mapping continuous vector to [0, 1]."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))


def run_binary_gwo(
    oracle: EvaluationOracle,
    population_size: int = 50,
    iterations: int = 100,
    base_seed: int = 0,
) -> dict[str, Union[NDArray[np.signedinteger], List[float], float, int]]:
    """
    Run Binary Grey Wolf Optimizer.

    Parameters
    ----------
    oracle
        EvaluationOracle to evaluate masks.
    population_size
        Number of wolves in population (P).
    iterations
        Maximum iterations (T).
    base_seed
        Random seed for reproducibility.

    Returns
    -------
    dict
        "best_mask", "best_fitness", "convergence", "runtime", "n_evaluations"
    """
    d = oracle.total_d
    rng = np.random.default_rng(42 + int(base_seed))

    # Initialize continuous positions for wolves in range [-3, 3]
    positions = rng.uniform(-3.0, 3.0, size=(population_size, d))

    # Convert initial positions to binary masks
    sig_pos = sigmoidal_transfer(positions)
    masks = (sig_pos > rng.random(size=(population_size, d))).astype(np.int8)

    t0 = time.perf_counter()

    # Initial evaluation
    fitnesses = np.array([oracle.evaluate(m).fitness for m in masks], dtype=np.float64)

    # Sort to find Alpha, Beta, Delta wolves
    sorted_idx = np.argsort(fitnesses)
    alpha_pos = positions[sorted_idx[0]].copy()
    alpha_fit = fitnesses[sorted_idx[0]]
    alpha_mask = masks[sorted_idx[0]].copy()

    beta_pos = positions[sorted_idx[1]].copy()
    beta_fit = fitnesses[sorted_idx[1]]

    delta_pos = positions[sorted_idx[2]].copy()
    delta_fit = fitnesses[sorted_idx[2]]

    convergence: list[float] = [float(alpha_fit)]

    for t in range(iterations):
        a = 2.0 - t * (2.0 / float(iterations))  # Decreases linearly from 2 to 0

        for i in range(population_size):
            for j in range(d):
                # Alpha wolf vector calculation
                r1, r2 = rng.random(), rng.random()
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                # Beta wolf vector calculation
                r1, r2 = rng.random(), rng.random()
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                # Delta wolf vector calculation
                r1, r2 = rng.random(), rng.random()
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                # Update position
                positions[i, j] = (X1 + X2 + X3) / 3.0

            # Convert new continuous position to binary mask using sigmoidal transfer
            sig_val = sigmoidal_transfer(positions[i])
            masks[i] = (sig_val > rng.random(size=d)).astype(np.int8)

            # Evaluate mask
            res = oracle.evaluate(masks[i])
            fitnesses[i] = res.fitness

            # Update Alpha, Beta, Delta
            if fitnesses[i] < alpha_fit:
                delta_fit, delta_pos = beta_fit, beta_pos.copy()
                beta_fit, beta_pos = alpha_fit, alpha_pos.copy()
                alpha_fit = fitnesses[i]
                alpha_pos = positions[i].copy()
                alpha_mask = masks[i].copy()
            elif fitnesses[i] < beta_fit:
                delta_fit, delta_pos = beta_fit, beta_pos.copy()
                beta_fit = fitnesses[i]
                beta_pos = positions[i].copy()
            elif fitnesses[i] < delta_fit:
                delta_fit = fitnesses[i]
                delta_pos = positions[i].copy()

        convergence.append(float(alpha_fit))

    t1 = time.perf_counter()

    return {
        "best_mask": alpha_mask,
        "best_fitness": float(alpha_fit),
        "convergence": convergence,
        "runtime": float(t1 - t0),
        "n_evaluations": oracle.total_evaluations,
    }
