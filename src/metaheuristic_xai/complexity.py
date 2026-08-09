"""
Computational Complexity Analysis module.

Formally derives asymptotic time and space complexity for all benchmarked algorithms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmComplexity:
    """Asymptotic complexity specification for a feature selection algorithm."""

    name: str
    time_complexity: str
    space_complexity: str
    evaluations_formula: str
    description: str


COMPLEXITY_REGISTRY: Dict[str, AlgorithmComplexity] = {
    "shap": AlgorithmComplexity(
        name="SHAP Top-k",
        time_complexity="O(C_model(D, N) + N_sub * D * M * depth)",
        space_complexity="O(N_sub * D + D)",
        evaluations_formula="E = 1",
        description="Trains base model once; computes TreeExplainer SHAP values on N_sub samples.",
    ),
    "lime": AlgorithmComplexity(
        name="LIME Top-k",
        time_complexity="O(C_model(D, N) + N_sub * K_samples * C_model(D, 1))",
        space_complexity="O(N_sub * D)",
        evaluations_formula="E = 1",
        description="Trains base model once; generates surrogate perturbations around N_sub samples.",
    ),
    "lasso": AlgorithmComplexity(
        name="LASSO (L1 Logistic Regression)",
        time_complexity="O(N * D^2 + D^3)",
        space_complexity="O(N * D + D)",
        evaluations_formula="E = 1",
        description="Solves L1-regularized convex optimization problem using SAGA/coordinate descent.",
    ),
    "rfe": AlgorithmComplexity(
        name="Recursive Feature Elimination (RFE)",
        time_complexity="O(D * C_model(D, N))",
        space_complexity="O(N * D + D)",
        evaluations_formula="E = k_steps",
        description="Iteratively fits base model and prunes least important features in steps.",
    ),
    "boruta": AlgorithmComplexity(
        name="Boruta",
        time_complexity="O(T_boruta * C_model(2D, N))",
        space_complexity="O(N * D)",
        evaluations_formula="E = T_boruta",
        description="Doubles feature space with shadow features and fits base model T_boruta times.",
    ),
    "ga": AlgorithmComplexity(
        name="Genetic Algorithm (GA)",
        time_complexity="O(P * T * C_model(|S|, N) + P * T * D)",
        space_complexity="O(P * D)",
        evaluations_formula="E = P * T",
        description="Evaluates P individuals across T generations with selection, crossover, and mutation.",
    ),
    "bpso": AlgorithmComplexity(
        name="Binary Particle Swarm Optimization (BPSO)",
        time_complexity="O(P * T * C_model(|S|, N) + P * T * D)",
        space_complexity="O(P * D)",
        evaluations_formula="E = P * T",
        description="Evaluates P particles across T iterations updating velocity and sigmoidal positions.",
    ),
    "sa": AlgorithmComplexity(
        name="Simulated Annealing (SA)",
        time_complexity="O(T_sa * C_model(|S|, N) + T_sa * D)",
        space_complexity="O(D)",
        evaluations_formula="E = T_sa",
        description="Single trajectory local search evaluating 1 candidate per temperature step.",
    ),
    "gwo": AlgorithmComplexity(
        name="Grey Wolf Optimizer (GWO)",
        time_complexity="O(P * T * C_model(|S|, N) + P * T * D)",
        space_complexity="O(P * D)",
        evaluations_formula="E = P * T",
        description="Evaluates P wolves across T iterations updating towards Alpha, Beta, Delta wolves.",
    ),
    "nsga2": AlgorithmComplexity(
        name="NSGA-II (Multi-Objective)",
        time_complexity="O(P * T * C_model(|S|, N) + T * (M_obj * P^2 + P * D))",
        space_complexity="O(P * D + P * M_obj)",
        evaluations_formula="E = P * T",
        description="Evaluates P solutions across T generations with non-dominated sorting and crowding distance.",
    ),
}


def get_complexity_summary() -> str:
    """Return plain-text breakdown of asymptotic complexity formulas."""
    lines = ["=" * 78, " ASYMPTOTIC COMPUTATIONAL COMPLEXITY OF ALGORITHMS", "=" * 78]
    for key, c in COMPLEXITY_REGISTRY.items():
        lines.append(f" {c.name} ({key}):")
        lines.append(f"   Time Complexity : {c.time_complexity}")
        lines.append(f"   Space Complexity: {c.space_complexity}")
        lines.append(f"   Evaluations     : {c.evaluations_formula}")
        lines.append(f"   Details         : {c.description}")
        lines.append("-" * 78)
    return "\n".join(lines)
