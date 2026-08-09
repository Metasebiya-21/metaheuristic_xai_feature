"""
Classical and attribution-based Feature Selectors module:
- SHAPSelector (SHAP TreeExplainer / KernelExplainer top-k)
- LIMESelector (LIME Tabular Explainer aggregated weight top-k)
- LassoSelector (L1 Logistic Regression / Lasso sparse selection)
- RFESelector (Recursive Feature Elimination)
- BorutaSelector (Boruta shadow feature selection)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import lime
import lime.lime_tabular
import numpy as np
import shap
from boruta import BorutaPy
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

from metaheuristic_xai.oracle import EvaluationOracle
from metaheuristic_xai.selectors.base import FeatureSelector

logger = logging.getLogger(__name__)


class SHAPSelector(FeatureSelector):
    """SHAP top-k feature selector using TreeExplainer on a trained classifier."""

    def __init__(
        self,
        top_k: int = 10,
        shap_max_samples: int = 200,
        oracle: Optional[EvaluationOracle] = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__("shap", oracle=oracle, random_state=random_state, **kwargs)
        self.top_k = top_k
        self.shap_max_samples = shap_max_samples

    def fit(
        self, X_train: NDArray[np.floating], y_train: NDArray[np.signedinteger]
    ) -> SHAPSelector:
        t0 = time.perf_counter()
        n_rows, n_dim = X_train.shape
        k = min(self.top_k, n_dim)

        clf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=1)
        clf.fit(X_train, y_train)

        n_sub = min(n_rows, self.shap_max_samples)
        if n_sub < n_rows:
            X_sub, _, _, _ = train_test_split(
                X_train, y_train, train_size=n_sub, stratify=y_train, random_state=self.random_state
            )
        else:
            X_sub = X_train

        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_sub)

        # Map to 1D importance vector
        if isinstance(shap_vals, list):
            vals = np.asarray(shap_vals[1] if len(shap_vals) > 1 else shap_vals[0], dtype=np.float64)
        else:
            vals = np.asarray(shap_vals, dtype=np.float64)

        if vals.ndim == 3:
            vals = vals[:, :, 1]
        importance = np.mean(np.abs(vals), axis=0)

        # Sort top-k
        order = np.lexsort((np.arange(n_dim), -importance))
        top_idx = order[:k]

        mask = np.zeros(n_dim, dtype=np.int8)
        mask[top_idx] = 1

        t1 = time.perf_counter()
        self.support_ = mask
        self.runtime_ = float(t1 - t0)
        self.n_evaluations_ = 1
        return self


class LIMESelector(FeatureSelector):
    """LIME-based feature selector aggregating feature weights across instances."""

    def __init__(
        self,
        top_k: int = 10,
        num_samples: int = 100,
        oracle: Optional[EvaluationOracle] = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__("lime", oracle=oracle, random_state=random_state, **kwargs)
        self.top_k = top_k
        self.num_samples = num_samples

    def fit(
        self, X_train: NDArray[np.floating], y_train: NDArray[np.signedinteger]
    ) -> LIMESelector:
        t0 = time.perf_counter()
        n_rows, n_dim = X_train.shape
        k = min(self.top_k, n_dim)

        clf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=1)
        clf.fit(X_train, y_train)

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=[f"f_{i}" for i in range(n_dim)],
            class_names=[str(c) for c in np.unique(y_train)],
            mode="classification",
            random_state=self.random_state,
        )

        n_sub = min(n_rows, self.num_samples)
        rng = np.random.default_rng(self.random_state)
        sample_idx = rng.choice(n_rows, size=n_sub, replace=False)

        importance = np.zeros(n_dim, dtype=np.float64)
        for idx in sample_idx:
            try:
                exp = explainer.explain_instance(
                    X_train[idx], clf.predict_proba, num_features=n_dim
                )
                for feat_idx, weight in exp.as_map()[1 if len(np.unique(y_train)) > 1 else 0]:
                    importance[feat_idx] += abs(weight)
            except Exception:
                continue

        order = np.lexsort((np.arange(n_dim), -importance))
        top_idx = order[:k]

        mask = np.zeros(n_dim, dtype=np.int8)
        mask[top_idx] = 1

        t1 = time.perf_counter()
        self.support_ = mask
        self.runtime_ = float(t1 - t0)
        self.n_evaluations_ = 1
        return self


class LassoSelector(FeatureSelector):
    """LASSO (L1-regularized Logistic Regression) feature selector."""

    def __init__(
        self,
        oracle: Optional[EvaluationOracle] = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__("lasso", oracle=oracle, random_state=random_state, **kwargs)

    def fit(
        self, X_train: NDArray[np.floating], y_train: NDArray[np.signedinteger]
    ) -> LassoSelector:
        t0 = time.perf_counter()
        n_dim = X_train.shape[1]

        clf = LogisticRegressionCV(
            penalty="l1",
            solver="saga",
            max_iter=1000,
            cv=3,
            random_state=self.random_state,
            n_jobs=1,
        )
        clf.fit(X_train, y_train)

        coefs = clf.coef_
        if coefs.ndim > 1:
            importance = np.max(np.abs(coefs), axis=0)
        else:
            importance = np.abs(coefs)

        mask = (importance > 1e-5).astype(np.int8)

        # Fallback if zero selected
        if np.sum(mask) == 0:
            mask[np.argmax(importance)] = 1

        t1 = time.perf_counter()
        self.support_ = mask
        self.runtime_ = float(t1 - t0)
        self.n_evaluations_ = 1
        return self


class RFESelector(FeatureSelector):
    """Recursive Feature Elimination (RFE) selector."""

    def __init__(
        self,
        n_features_to_select: Optional[int] = 10,
        oracle: Optional[EvaluationOracle] = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__("rfe", oracle=oracle, random_state=random_state, **kwargs)
        self.n_features_to_select = n_features_to_select

    def fit(
        self, X_train: NDArray[np.floating], y_train: NDArray[np.signedinteger]
    ) -> RFESelector:
        t0 = time.perf_counter()
        n_dim = X_train.shape[1]
        k = min(self.n_features_to_select or max(1, n_dim // 2), n_dim)

        base_clf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=1)
        rfe = RFE(estimator=base_clf, n_features_to_select=k, step=0.1)
        rfe.fit(X_train, y_train)

        mask = rfe.support_.astype(np.int8)

        t1 = time.perf_counter()
        self.support_ = mask
        self.runtime_ = float(t1 - t0)
        self.n_evaluations_ = 1
        return self


class BorutaSelector(FeatureSelector):
    """Boruta shadow feature selector."""

    def __init__(
        self,
        max_iter: int = 100,
        oracle: Optional[EvaluationOracle] = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__("boruta", oracle=oracle, random_state=random_state, **kwargs)
        self.max_iter = max_iter

    def fit(
        self, X_train: NDArray[np.floating], y_train: NDArray[np.signedinteger]
    ) -> BorutaSelector:
        t0 = time.perf_counter()

        base_clf = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=self.random_state, n_jobs=1
        )
        boruta = BorutaPy(
            estimator=base_clf,
            n_estimators="auto",
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=0,
        )
        boruta.fit(X_train, y_train)

        mask = boruta.support_.astype(np.int8)

        # Fallback to tentative if no confirmed features selected
        if np.sum(mask) == 0 and hasattr(boruta, "support_weak_"):
            mask = (boruta.support_ | boruta.support_weak_).astype(np.int8)

        # Second fallback: pick highest ranking feature
        if np.sum(mask) == 0:
            rankings = boruta.ranking_
            mask[np.argmin(rankings)] = 1

        t1 = time.perf_counter()
        self.support_ = mask
        self.runtime_ = float(t1 - t0)
        self.n_evaluations_ = 1
        return self
