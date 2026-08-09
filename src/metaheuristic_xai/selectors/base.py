"""
Base FeatureSelector interface for feature selection methods.

CRITICAL LEAKAGE PREVENTION:
Feature selectors MUST ONLY receive X_train and y_train in fit().
Test data is never passed to fit().
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from metaheuristic_xai.oracle import EvaluationOracle, EvaluationResult

logger = logging.getLogger(__name__)


class FeatureSelector(ABC):
    """Abstract base class for all feature selection algorithms."""

    def __init__(
        self,
        name: str,
        oracle: Optional[EvaluationOracle] = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.oracle = oracle
        self.random_state = random_state
        self.kwargs = kwargs

        self.support_: Optional[NDArray[np.signedinteger]] = None
        self.runtime_: float = 0.0
        self.n_evaluations_: int = 0
        self.convergence_: list[float] = []

    @abstractmethod
    def fit(
        self, X_train: NDArray[np.floating], y_train: NDArray[np.signedinteger]
    ) -> FeatureSelector:
        """
        Fit feature selector on training data ONLY.

        Must populate `self.support_` as a 1D binary numpy array of 0s and 1s.
        """

    def get_support(self) -> NDArray[np.signedinteger]:
        """Return binary mask of selected features (1 = selected, 0 = excluded)."""
        if self.support_ is None:
            raise RuntimeError(f"Selector '{self.name}' has not been fitted yet.")
        return np.asarray(self.support_, dtype=np.int8)

    def transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Transform feature matrix X by selecting only active features."""
        supp = self.get_support()
        idx = np.where(supp == 1)[0]
        if idx.size == 0:
            logger.warning("Transforming X with empty mask; returning empty array.")
            return np.empty((X.shape[0], 0), dtype=X.dtype)
        return X[:, idx]

    def evaluate(self) -> Optional[EvaluationResult]:
        """Evaluate the selected feature mask using the oracle if available."""
        if self.oracle is None:
            return None
        supp = self.get_support()
        return self.oracle.evaluate(supp)
