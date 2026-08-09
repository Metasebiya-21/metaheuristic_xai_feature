"""
Centralized Evaluation Oracle and Memoization Cache for feature selection.

Tracks model evaluations, memoizes feature mask scores safely by dataset/classifier split,
computes multi-metric evaluation metrics (accuracy, balanced accuracy, precision, recall, f1, roc_auc, pr_auc),
and supports both single-objective scalar fitness and multi-objective Pareto metrics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .classifiers import ClassifierAdapter

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Standardized result of evaluating a feature mask."""

    mask: NDArray[np.signedinteger]
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    error: float  # 1.0 - accuracy
    n_features: int
    feature_ratio: float  # n_features / total_d
    fitness: float  # alpha * (1 - accuracy) + (1 - alpha) * feature_ratio
    runtime: float  # seconds spent in fit + predict for this evaluation
    evaluation_id: int
    cached: bool = False
    used_indices: NDArray[np.signedinteger] = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )


def decode_mask(mask: ArrayLike) -> NDArray[np.signedinteger]:
    """Convert binary array-like mask to 1D integer array of selected column indices."""
    m = np.ravel(np.asarray(mask, dtype=np.int8))
    return np.where(m == 1)[0].astype(np.int64, copy=False)


class EvaluationOracle:
    """
    Centralized Evaluation Oracle with scoped memoization cache and metric tracking.

    CRITICAL LEAKAGE PROTECTION:
    `X_test` and `y_test` are passed ONLY for evaluation. Feature selection search operates on
    train/val datasets or internal splits.
    """

    def __init__(
        self,
        classifier: ClassifierAdapter,
        X_train: NDArray[np.floating],
        y_train: NDArray[np.signedinteger],
        X_test: NDArray[np.floating],
        y_test: NDArray[np.signedinteger],
        alpha: float = 0.8,
        dataset_name: str = "wdbc",
        split_id: int = 0,
        enable_cache: bool = True,
    ) -> None:
        self.classifier = classifier
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=np.int64)
        self.X_test = np.asarray(X_test, dtype=np.float32)
        self.y_test = np.asarray(y_test, dtype=np.int64)
        self.alpha = float(alpha)
        self.dataset_name = dataset_name
        self.split_id = split_id
        self.enable_cache = enable_cache

        self.total_d = int(self.X_train.shape[1])
        self.n_classes = len(np.unique(self.y_train))
        self.is_multiclass = self.n_classes > 2

        # Cache: tuple(mask_bits) -> EvaluationResult
        self._cache: Dict[Tuple[int, ...], EvaluationResult] = {}

        # Counters
        self.total_evaluations: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.cumulative_runtime: float = 0.0

    def evaluate(self, mask: ArrayLike) -> EvaluationResult:
        """
        Evaluate a binary mask on the target dataset split using the specified classifier.

        Returns comprehensive EvaluationResult.
        """
        m_arr = np.ravel(np.asarray(mask, dtype=np.int8))
        if m_arr.size != self.total_d:
            raise ValueError(
                f"Mask size ({m_arr.size}) does not match dataset dimension ({self.total_d})."
            )

        mask_tuple = tuple(int(x) for x in m_arr)

        if self.enable_cache and mask_tuple in self._cache:
            self.cache_hits += 1
            self.total_evaluations += 1
            cached_res = self._cache[mask_tuple]
            return EvaluationResult(
                mask=cached_res.mask,
                accuracy=cached_res.accuracy,
                balanced_accuracy=cached_res.balanced_accuracy,
                precision=cached_res.precision,
                recall=cached_res.recall,
                f1=cached_res.f1,
                roc_auc=cached_res.roc_auc,
                pr_auc=cached_res.pr_auc,
                error=cached_res.error,
                n_features=cached_res.n_features,
                feature_ratio=cached_res.feature_ratio,
                fitness=cached_res.fitness,
                runtime=0.0,
                evaluation_id=self.total_evaluations,
                cached=True,
                used_indices=cached_res.used_indices,
            )

        self.cache_misses += 1
        self.total_evaluations += 1
        eval_id = self.total_evaluations

        idx = decode_mask(m_arr)
        nsel = int(idx.size)
        feat_ratio = nsel / float(self.total_d)

        # Edge case: Empty mask (no feature selected)
        if nsel == 0:
            res = EvaluationResult(
                mask=m_arr,
                accuracy=0.0,
                balanced_accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                roc_auc=0.5,
                pr_auc=0.0,
                error=1.0,
                n_features=0,
                feature_ratio=0.0,
                fitness=1.0,  # maximum penalty
                runtime=0.0,
                evaluation_id=eval_id,
                cached=False,
                used_indices=idx,
            )
            if self.enable_cache:
                self._cache[mask_tuple] = res
            return res

        # Fit model on selected columns only
        X_tr_sub = self.X_train[:, idx]
        X_te_sub = self.X_test[:, idx]

        clf = self.classifier.clone()
        t0 = time.perf_counter()
        try:
            clf.fit(X_tr_sub, self.y_train)
            y_pred = clf.predict(X_te_sub)
        except Exception as e:
            logger.error("Classifier fit/predict failed for evaluation %d: %s", eval_id, e)
            raise

        t1 = time.perf_counter()
        eval_time = float(t1 - t0)
        self.cumulative_runtime += eval_time

        # Calculate metrics
        acc = float(accuracy_score(self.y_test, y_pred))
        b_acc = float(balanced_accuracy_score(self.y_test, y_pred))
        avg_type = "macro" if self.is_multiclass else "binary"

        prec = float(precision_score(self.y_test, y_pred, average=avg_type, zero_division=0))
        rec = float(recall_score(self.y_test, y_pred, average=avg_type, zero_division=0))
        f1 = float(f1_score(self.y_test, y_pred, average=avg_type, zero_division=0))

        # Probability-based metrics (ROC-AUC, PR-AUC) if available
        roc_auc = 0.5
        pr_auc = 0.0
        try:
            y_proba = clf.predict_proba(X_te_sub)
            if not self.is_multiclass:
                y_p1 = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
                roc_auc = float(roc_auc_score(self.y_test, y_p1))
                pr_auc = float(average_precision_score(self.y_test, y_p1))
            else:
                roc_auc = float(
                    roc_auc_score(self.y_test, y_proba, multi_class="ovr", average="macro")
                )
        except Exception:
            pass  # Fallback to defaults if proba is unsupported or fails

        error = 1.0 - acc
        fitness = self.alpha * error + (1.0 - self.alpha) * feat_ratio

        res = EvaluationResult(
            mask=m_arr,
            accuracy=acc,
            balanced_accuracy=b_acc,
            precision=prec,
            recall=rec,
            f1=f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            error=error,
            n_features=nsel,
            feature_ratio=feat_ratio,
            fitness=fitness,
            runtime=eval_time,
            evaluation_id=eval_id,
            cached=False,
            used_indices=idx,
        )

        if self.enable_cache:
            self._cache[mask_tuple] = res

        return res

    def get_stats(self) -> dict[str, Any]:
        """Return oracle usage statistics."""
        return {
            "total_evaluations": self.total_evaluations,
            "unique_masks": len(self._cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / float(self.total_evaluations)
                if self.total_evaluations > 0
                else 0.0
            ),
            "cumulative_runtime": self.cumulative_runtime,
        }


class CrossValOracle:
    """Search oracle that scores a mask by stratified k-fold cross-validation on
    the training partition, instead of a single held-out inner split.

    Same interface as :class:`EvaluationOracle` (``evaluate`` -> ``EvaluationResult``,
    the same counters and ``X_train``/``X_test`` attributes), so the search
    algorithms and selectors use it interchangeably. Costs ``k`` model fits per
    unique mask; masks are memoised by exact bit pattern.

    The reported ``accuracy`` (hence ``fitness``) is pooled across the k held-out
    folds -- a lower-variance estimate than one 25% split, which is the main
    driver of the run-to-run instability on the small-sample datasets.
    """

    def __init__(
        self,
        classifier: ClassifierAdapter,
        X: NDArray[np.floating],
        y: NDArray[np.signedinteger],
        *,
        n_splits: int = 5,
        n_repeats: int = 1,
        alpha: float = 0.8,
        dataset_name: str = "wdbc",
        split_id: int = 0,
        enable_cache: bool = True,
    ) -> None:
        self.classifier = classifier
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        # aliases so callers that read .X_train / .X_test keep working
        self.X_train = self.X
        self.X_test = self.X
        self.y_train = self.y
        self.y_test = self.y
        self.alpha = float(alpha)
        self.dataset_name = dataset_name
        self.split_id = split_id
        self.enable_cache = enable_cache

        self.total_d = int(self.X.shape[1])
        self.n_classes = len(np.unique(self.y))
        self.is_multiclass = self.n_classes > 2
        n_min = int(np.min(np.bincount(self.y))) if self.n_classes > 1 else len(self.y)
        self.n_splits = max(2, min(int(n_splits), n_min))
        self.n_repeats = max(1, int(n_repeats))

        self._cache: Dict[Tuple[int, ...], EvaluationResult] = {}
        self.total_evaluations = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cumulative_runtime = 0.0

    def evaluate(self, mask: ArrayLike) -> EvaluationResult:
        from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

        m_arr = np.ravel(np.asarray(mask, dtype=np.int8))
        if m_arr.size != self.total_d:
            raise ValueError(
                f"Mask size ({m_arr.size}) does not match dataset dimension ({self.total_d})."
            )
        key = tuple(int(x) for x in m_arr)
        if self.enable_cache and key in self._cache:
            self.cache_hits += 1
            self.total_evaluations += 1
            c = self._cache[key]
            return EvaluationResult(
                mask=c.mask, accuracy=c.accuracy, balanced_accuracy=c.balanced_accuracy,
                precision=c.precision, recall=c.recall, f1=c.f1, roc_auc=c.roc_auc,
                pr_auc=c.pr_auc, error=c.error, n_features=c.n_features,
                feature_ratio=c.feature_ratio, fitness=c.fitness, runtime=0.0,
                evaluation_id=self.total_evaluations, cached=True, used_indices=c.used_indices,
            )

        self.cache_misses += 1
        self.total_evaluations += 1
        eval_id = self.total_evaluations
        idx = decode_mask(m_arr)
        nsel = int(idx.size)
        feat_ratio = nsel / float(self.total_d)

        if nsel == 0:
            res = EvaluationResult(
                mask=m_arr, accuracy=0.0, balanced_accuracy=0.0, precision=0.0, recall=0.0,
                f1=0.0, roc_auc=0.5, pr_auc=0.0, error=1.0, n_features=0, feature_ratio=0.0,
                fitness=1.0, runtime=0.0, evaluation_id=eval_id, cached=False, used_indices=idx,
            )
            if self.enable_cache:
                self._cache[key] = res
            return res

        Xs = self.X[:, idx]
        if self.n_repeats > 1:
            skf = RepeatedStratifiedKFold(
                n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.split_id
            )
        else:
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.split_id)
        y_true_all: list[NDArray] = []
        y_pred_all: list[NDArray] = []
        t0 = time.perf_counter()
        for tr, va in skf.split(Xs, self.y):
            clf = self.classifier.clone()
            clf.fit(Xs[tr], self.y[tr])
            y_true_all.append(self.y[va])
            y_pred_all.append(clf.predict(Xs[va]))
        self.cumulative_runtime += time.perf_counter() - t0

        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        avg = "macro" if self.is_multiclass else "binary"
        acc = float(accuracy_score(y_true, y_pred))
        res = EvaluationResult(
            mask=m_arr,
            accuracy=acc,
            balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
            roc_auc=0.5,
            pr_auc=0.0,
            error=1.0 - acc,
            n_features=nsel,
            feature_ratio=feat_ratio,
            fitness=self.alpha * (1.0 - acc) + (1.0 - self.alpha) * feat_ratio,
            runtime=0.0,
            evaluation_id=eval_id,
            cached=False,
            used_indices=idx,
        )
        if self.enable_cache:
            self._cache[key] = res
        return res

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_evaluations": self.total_evaluations,
            "unique_masks": len(self._cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / float(self.total_evaluations)
                if self.total_evaluations else 0.0
            ),
            "cumulative_runtime": self.cumulative_runtime,
        }


def build_oracles(
    classifier: ClassifierAdapter,
    X_train: NDArray[np.floating],
    y_train: NDArray[np.signedinteger],
    X_test: NDArray[np.floating],
    y_test: NDArray[np.signedinteger],
    *,
    alpha: float = 0.8,
    dataset_name: str = "wdbc",
    split_id: int = 0,
    val_size: float = 0.25,
    cv: int | None = None,
    cv_repeats: int = 1,
    enable_cache: bool = True,
) -> Tuple[Any, "EvaluationOracle"]:
    """Build the (search, test) oracle pair used for a leakage-free run.

    * **search oracle** scores masks using only ``(X_train, y_train)`` -- either a
      single inner stratified validation split (default) or, when ``cv`` is set,
      stratified ``cv``-fold cross-validation (:class:`CrossValOracle`). This is
      what the feature-selection search optimises against.
    * **test oracle** fits on the full ``(X_train, y_train)`` and scores on the
      held-out ``(X_test, y_test)`` -- used exactly once, on the final mask, to
      produce the reported metrics. The same test oracle is used for every method,
      so the comparison is fair.

    The held-out test set never reaches the search oracle.
    """
    from sklearn.model_selection import train_test_split

    if cv is not None:
        search_oracle: Any = CrossValOracle(
            classifier=classifier,
            X=X_train,
            y=y_train,
            n_splits=int(cv),
            n_repeats=int(cv_repeats),
            alpha=alpha,
            dataset_name=dataset_name,
            split_id=split_id,
            enable_cache=enable_cache,
        )
    else:
        X_tr = np.asarray(X_train)
        y_tr = np.asarray(y_train)
        stratify = y_tr if len(np.unique(y_tr)) > 1 else None
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_tr, y_tr, test_size=val_size, random_state=split_id, stratify=stratify
        )
        search_oracle = EvaluationOracle(
            classifier=classifier,
            X_train=X_fit,
            y_train=y_fit,
            X_test=X_val,
            y_test=y_val,
            alpha=alpha,
            dataset_name=dataset_name,
            split_id=split_id,
            enable_cache=enable_cache,
        )

    test_oracle = EvaluationOracle(
        classifier=classifier,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        alpha=alpha,
        dataset_name=dataset_name,
        split_id=split_id,
        enable_cache=enable_cache,
    )
    return search_oracle, test_oracle
