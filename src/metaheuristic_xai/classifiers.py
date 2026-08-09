"""
Classifier Abstraction module for feature-selection benchmark.

Supports 5 classifier architectures:
- RandomForestAdapter (Random Forest)
- XGBoostAdapter (XGBoost)
- LightGBMAdapter (LightGBM)
- SVMAdapter (Support Vector Machines)
- MLPAdapter (Multi-Layer Perceptron)
"""

from __future__ import annotations

import inspect
import logging
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Silence C-level and Python-level LightGBM logging globally
os.environ["LIGHTGBM_VERBOSITY"] = "-1"
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

logger = logging.getLogger(__name__)


def _filter_valid_kwargs(cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs to match valid constructor parameters of target estimator class."""
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid_params}


class ClassifierAdapter(ABC):
    """Abstract base class for model adapters in feature selection."""

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.params = params or {}
        self.model: Any = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.signedinteger]) -> ClassifierAdapter:
        """Fit underlying estimator on feature matrix X and target array y."""
        self.model.fit(X, y)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.signedinteger]:
        """Predict class labels for X."""
        return self.model.predict(X)

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict class probabilities for X."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise NotImplementedError(
            f"Probability estimation not supported by {self.__class__.__name__}"
        )

    def score(self, X: NDArray[np.floating], y: NDArray[np.signedinteger]) -> float:
        """Return mean classification accuracy."""
        return float(self.model.score(X, y))

    @abstractmethod
    def clone(self) -> ClassifierAdapter:
        """Return an un-fitted copy of this adapter with identical configuration."""


class RandomForestAdapter(ClassifierAdapter):
    """Random Forest Classifier Adapter."""

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        n_jobs: int = 1,
        **kwargs: Any,
    ) -> None:
        all_params = {
            "n_estimators": n_estimators,
            "random_state": random_state,
            "n_jobs": n_jobs,
            **kwargs,
        }
        super().__init__("random_forest", all_params)
        model_kwargs = _filter_valid_kwargs(RandomForestClassifier, all_params)
        self.model = RandomForestClassifier(**model_kwargs)

    def clone(self) -> RandomForestAdapter:
        return RandomForestAdapter(**self.params)


class XGBoostAdapter(ClassifierAdapter):
    """XGBoost Classifier Adapter."""

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        n_jobs: int = 1,
        eval_metric: str = "logloss",
        verbosity: int = 0,
        **kwargs: Any,
    ) -> None:
        all_params = {
            "n_estimators": n_estimators,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "eval_metric": eval_metric,
            "verbosity": verbosity,
            **kwargs,
        }
        super().__init__("xgboost", all_params)
        model_kwargs = _filter_valid_kwargs(xgb.XGBClassifier, all_params)
        self.model = xgb.XGBClassifier(**model_kwargs)

    def clone(self) -> XGBoostAdapter:
        return XGBoostAdapter(**self.params)

class LightGBMAdapter(ClassifierAdapter):
    """LightGBM Classifier Adapter tuned for ultra-high dimensional and small-sample datasets."""

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        n_jobs: int = 1,
        verbosity: int = -1,
        **kwargs: Any,
    ) -> None:
        all_params = {
            "n_estimators": n_estimators,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbosity": verbosity,
            "verbose": verbosity,
            "min_child_samples": kwargs.get("min_child_samples", 20),
            **kwargs,
        }
        super().__init__("lightgbm", all_params)

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.signedinteger]
    ) -> LightGBMAdapter:
        n_samples = X.shape[0]

        # Dynamically scale min_child_samples to prevent 'best gain: -inf' warnings on tiny datasets (colon, leukemia)
        adapted_min_child = max(2, min(20, n_samples // 5))

        model_kwargs = _filter_valid_kwargs(lgb.LGBMClassifier, self.params)
        model_kwargs["min_child_samples"] = self.params.get(
            "min_child_samples", adapted_min_child
        )
        model_kwargs["verbosity"] = -1
        model_kwargs["verbose"] = -1

        self.model = lgb.LGBMClassifier(**model_kwargs)
        self.model.fit(X, y)
        return self

    def clone(self) -> LightGBMAdapter:
        return LightGBMAdapter(**self.params)

class SVMAdapter(ClassifierAdapter):
    """Support Vector Machine (SVC) Adapter without deprecated probability flag."""

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        probability: bool = True,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        all_params = {
            "C": C,
            "kernel": kernel,
            "random_state": random_state,
            **kwargs,
        }
        super().__init__("svm", all_params)
        model_kwargs = _filter_valid_kwargs(SVC, all_params)
        base_svc = SVC(**model_kwargs)

        if probability:
            # Replaces deprecated SVC(probability=True) for sklearn 1.9+
            self.model = CalibratedClassifierCV(base_svc, ensemble=False)
        else:
            self.model = base_svc

    def clone(self) -> SVMAdapter:
        return SVMAdapter(**self.params)


class MLPAdapter(ClassifierAdapter):
    """Multi-Layer Perceptron (Neural Network) Adapter."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100,),
        max_iter: int = 500,           # Increased from 200 to allow convergence
        early_stopping: bool = True,   # Halts early if validation loss flattens
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        all_params = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "max_iter": max_iter,
            "early_stopping": early_stopping,
            "random_state": random_state,
            "n_iter_no_change": 10,
            **kwargs,
        }
        super().__init__("mlp", all_params)
        model_kwargs = _filter_valid_kwargs(MLPClassifier, all_params)
        self.model = MLPClassifier(**model_kwargs)

    def clone(self) -> MLPAdapter:
        return MLPAdapter(**self.params)

CLASSIFIER_REGISTRY: dict[str, Type[ClassifierAdapter]] = {
    "random_forest": RandomForestAdapter,
    "xgboost": XGBoostAdapter,
    "lightgbm": LightGBMAdapter,
    "svm": SVMAdapter,
    "mlp": MLPAdapter,
}


def get_classifier_adapter(name: str, **kwargs: Any) -> ClassifierAdapter:
    """Instantiate a classifier adapter by name."""
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier '{name}'. Available: {list(CLASSIFIER_REGISTRY.keys())}"
        )
    return CLASSIFIER_REGISTRY[name](**kwargs)
