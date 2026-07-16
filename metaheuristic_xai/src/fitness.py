"""
Fitness function for feature subset selection: balance accuracy and subset size.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

# Objective weights: minimize f(S) = ACC_WEIGHT * (1-Acc) + SIZE_WEIGHT * (|S|/d)
ACC_WEIGHT: float = 0.9
SIZE_WEIGHT: float = 0.1
RF_N_ESTIMATORS: int = 100
RF_SEED: int = 42


def get_rf_n_jobs() -> int:
    """Respect ``SKLEARN_N_JOBS``; default is all cores (``-1``). Set to ``1`` on CI."""
    try:
        return int(os.environ.get("SKLEARN_N_JOBS", "-1").strip() or -1)
    except (TypeError, ValueError):
        return -1


def decode_mask(mask: ArrayLike) -> NDArray[np.signedinteger]:
    """
    Convert a binary feature mask to the sorted indices of selected features.

    Parameters
    ----------
    mask
        1D binary array-like (0/1 or bool). Length = number of features `d`.

    Returns
    -------
    indices
        Integer array of feature indices with ``mask[j] == 1``, sorted ascending.
    """
    m = np.ravel(np.asarray(mask, dtype=np.int8))
    return np.where(m == 1)[0].astype(np.int64, copy=False)


def evaluate_mask(
    mask: ArrayLike,
    X_train: NDArray[np.floating],
    X_test: NDArray[np.floating],
    y_train: NDArray,
    y_test: NDArray,
    acc_weight: float | None = None,
    size_weight: float | None = None,
) -> tuple[float, float, int, NDArray[np.signedinteger]]:
    """
    Evaluate a binary feature mask using the project fitness and a random forest.

    The fitness to **minimize** is::

        f(S) = 0.9 * (1 - accuracy) + 0.1 * (|S| / d)

    The **all-zero** mask (no feature selected) is a degenerate case: the classifier
    is not fit; fitness is the maximum penalty, accuracy 0, and 0 features reported.

    Parameters
    ----------
    mask
        Binary 1D mask of length ``d`` (number of columns in ``X_train``).
    X_train, X_test, y_train, y_test
        Scikit-learn style arrays. Features must be pre-scaled consistently.

    Returns
    -------
    fitness
        Scalar to minimize; higher is worse.
    accuracy
        Holdout accuracy on ``X_test`` using selected columns only.
    n_features
        Count ``|S|`` of active bits in the mask.
    used_indices
        The indices returned by ``decode_mask`` (empty if no feature selected).
    """
    m = np.ravel(np.asarray(mask, dtype=np.int8))
    d = int(X_train.shape[1])
    if m.size != d:
        msg = f"mask length {m.size} does not match n_features {d}."
        raise ValueError(msg)

    idx = decode_mask(m)
    nsel = int(idx.size)

    if nsel == 0:
        worst = 1.0
        return worst, 0.0, 0, np.array([], dtype=np.int64)

    if np.any(m != 0) and np.any(m != 1) and not np.issubdtype(
        np.asarray(mask).dtype, np.bool_
    ):
        logger.debug("Coercing non-binary mask values; expected 0/1.")

    X_tr = X_train[:, idx]
    X_te = X_test[:, idx]

    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RF_SEED,
        n_jobs=get_rf_n_jobs(),
    )
    try:
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
    except ValueError as e:
        logger.exception("RandomForest training or predict failed for mask of size %d.", nsel)
        raise

    if acc_weight is None:
        aw = ACC_WEIGHT
    else:
        aw = float(acc_weight)
    if size_weight is not None:
        sw = float(size_weight)
    elif acc_weight is not None:
        sw = 1.0 - aw
    else:
        sw = SIZE_WEIGHT
    accuracy = float(accuracy_score(y_test, y_pred))
    ratio = nsel / float(d)
    fitness = aw * (1.0 - accuracy) + sw * ratio
    return fitness, accuracy, nsel, idx
