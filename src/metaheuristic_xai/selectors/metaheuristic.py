"""
Metaheuristic Feature Selectors module wrapping GA, BPSO, SA, GWO, and NSGA-II.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from metaheuristic_xai.algorithms.ga import run_genetic_algorithm
from metaheuristic_xai.algorithms.gwo import run_binary_gwo
from metaheuristic_xai.algorithms.nsga2 import run_nsga2
from metaheuristic_xai.algorithms.pso import run_binary_pso
from metaheuristic_xai.algorithms.sa import run_simulated_annealing
from metaheuristic_xai.classifiers import RandomForestAdapter
from metaheuristic_xai.oracle import EvaluationOracle
from metaheuristic_xai.selectors.base import FeatureSelector
from metaheuristic_xai.selectors.classical import (
    BorutaSelector,
    LassoSelector,
    LIMESelector,
    RFESelector,
    SHAPSelector,
)

logger = logging.getLogger(__name__)


def _ensure_oracle(
    oracle: Optional[EvaluationOracle],
    X_train: NDArray[np.floating],
    y_train: NDArray[np.signedinteger],
    random_state: int,
) -> EvaluationOracle:
    """Fallback oracle created if no external oracle is passed."""
    if oracle is not None:
        return oracle

    # Use stratified split to prevent class-imbalance issues on sorted datasets
    X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train if len(np.unique(y_train)) > 1 else None,
    )

    clf = RandomForestAdapter(n_estimators=100, random_state=random_state)
    return EvaluationOracle(
        classifier=clf,
        X_train=X_tr_sub,
        y_train=y_tr_sub,
        X_test=X_val_sub,
        y_test=y_val_sub,
        enable_cache=True,
    )


def _apply_run_result(
    selector: FeatureSelector,
    orc: EvaluationOracle,
    res: dict[str, Any],
) -> None:
    """Populate common selector attributes from a metaheuristic run result."""
    selector.support_ = np.asarray(res["best_mask"], dtype=np.int8)
    selector.runtime_ = float(res["runtime"])
    selector.convergence_ = [float(x) for x in res.get("convergence", [])]
    selector.n_evaluations_ = int(res.get("n_evaluations", orc.total_evaluations))


class GASelector(FeatureSelector):
    """Genetic Algorithm Feature Selector."""

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        oracle: Optional[EvaluationOracle] = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__("ga", oracle=oracle, random_state=random_state, **kwargs)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def fit(
        self, X_train: NDArray[np.floating], y_train: NDArray[np.signedinteger]
    ) -> GASelector:
        orc = _ensure_oracle(self.oracle, X_train, y_train, self.random_state)

        res = run_genetic_algorithm(
            orc.X_train,
            orc.X_test,
            orc.y_train,
            orc.y_test,
            population_size=self.population_size,
            generations=self.generations,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            base_seed=self.random_state,
            oracle=orc,
        )

        _apply_run_result(self, orc, res)
        return self


class BPSOSelector(FeatureSelector):
    """Binary PSO Feature Selector."""

    def __init__(
        self,
        swarm_size: int = 50,
        iters: int = 100,
        oracle: Optional[EvaluationOracle] = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__("bpso", oracle=oracle, random_state=random_state, **kwargs)
        self.swarm_size = swarm_size
        self.iters = iters

    def fit(
        self, X_train: NDArray[np.floating], y_train: NDArray[np.signedinteger]
    ) -> BPSOSelector:
        orc = _ensure_oracle(self.oracle, X_train, y_train, self.random_state)

        res = run_binary_pso(
            orc.X_train,
            orc.X_test,
            orc.y_train,
            orc.y_test,
            swarm_size=self.swarm_size,
            iters=self.iters,
            base_seed=self.random_state,
            oracle=orc,
        )

        _apply_run_result(self, orc, res)
        return self


class SASelector(FeatureSelector):
    """Simulated Annealing Feature Selector."""

    def __init__(
        self,
        initial_temp: float = 100.0,
        cooling: float = 0.95,
        max_steps: Optional[int] = None,
        oracle: Optional[EvaluationOracle] = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__("sa", oracle=oracle, random_state=random_state, **kwargs)
        self.initial_temp = initial_temp
        self.cooling = cooling
        self.max_steps = max_steps

    def fit(
        self, X_train: NDArray[np.floating], y_train: NDArray[np.signedinteger]
    ) -> SASelector:
        orc = _ensure_oracle(self.oracle, X_train, y_train, self.random_state)

        res = run_simulated_annealing(
            orc.X_train,
            orc.X_test,
            orc.y_train,
            orc.y_test,
            initial_temp=self.initial_temp,
            cooling=self.cooling,
            base_seed=self.random_state,
            max_steps=self.max_steps,
            oracle=orc,
        )

        _apply_run_result(self, orc, res)
        return self


class GWOSelector(FeatureSelector):
    """Grey Wolf Optimizer Feature Selector."""

    def __init__(
        self,
        population_size: int = 50,
        iterations: int = 100,
        oracle: Optional[EvaluationOracle] = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__("gwo", oracle=oracle, random_state=random_state, **kwargs)
        self.population_size = population_size
        self.iterations = iterations

    def fit(
        self, X_train: NDArray[np.floating], y_train: NDArray[np.signedinteger]
    ) -> GWOSelector:
        orc = _ensure_oracle(self.oracle, X_train, y_train, self.random_state)

        res = run_binary_gwo(
            oracle=orc,
            population_size=self.population_size,
            iterations=self.iterations,
            base_seed=self.random_state,
        )

        _apply_run_result(self, orc, res)
        return self


class NSGA2Selector(FeatureSelector):
    """True Multi-Objective NSGA-II Feature Selector."""

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
        oracle: Optional[EvaluationOracle] = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__("nsga2", oracle=oracle, random_state=random_state, **kwargs)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.pareto_front_: list[Any] = []
        self.pareto_history_: list[list[Any]] = []
        self.pareto_history_gens_: list[int] = []
        self.hypervolume_: float = 0.0

    def fit(
        self, X_train: NDArray[np.floating], y_train: NDArray[np.signedinteger]
    ) -> NSGA2Selector:
        orc = _ensure_oracle(self.oracle, X_train, y_train, self.random_state)

        res = run_nsga2(
            oracle=orc,
            population_size=self.population_size,
            generations=self.generations,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            base_seed=self.random_state,
        )

        _apply_run_result(self, orc, res)
        self.pareto_front_ = res.get("pareto_front", [])
        self.pareto_history_ = res.get("pareto_history", [])
        self.pareto_history_gens_ = res.get("pareto_history_gens", [])
        self.hypervolume_ = float(res.get("hypervolume", 0.0))
        return self


SELECTOR_REGISTRY: dict[str, type[FeatureSelector]] = {
    "shap": SHAPSelector,
    "lime": LIMESelector,
    "lasso": LassoSelector,
    "rfe": RFESelector,
    "boruta": BorutaSelector,
    "ga": GASelector,
    "bpso": BPSOSelector,
    "sa": SASelector,
    "gwo": GWOSelector,
    "nsga2": NSGA2Selector,
}


def get_feature_selector(name: str, **kwargs: Any) -> FeatureSelector:
    """Instantiate feature selector by name."""
    if name not in SELECTOR_REGISTRY:
        raise ValueError(
            f"Unknown selector '{name}'. Available: {list(SELECTOR_REGISTRY.keys())}"
        )
    return SELECTOR_REGISTRY[name](**kwargs)
