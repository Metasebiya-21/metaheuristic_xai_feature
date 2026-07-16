"""
Load, split, and scale the Breast Cancer Wisconsin dataset.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_breast_cancer_data(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.signedinteger],
    NDArray[np.signedinteger],
    NDArray[str],
    StandardScaler,
]:
    """
    Load the Breast Cancer Wisconsin dataset, perform a stratified train/test split, and
    z-score the features (fit on train only).

    Parameters
    ----------
    test_size
        Proportion of samples in the test set.
    random_state
        Random seed for reproducible splitting.

    Returns
    -------
    X_train, X_test
        Standardized feature arrays (as float32 for downstream consistency).
    y_train, y_test
        Integer class labels in {0, 1}.
    feature_names
        1D array of feature name strings, in column order.
    scaler
        Fitted ``StandardScaler`` (useful to inspect or persist transforms).
    """
    if not 0.0 < test_size < 1.0:
        msg = f"test_size must be in (0,1), got {test_size!r}."
        raise ValueError(msg)

    try:
        data = load_breast_cancer()
    except (OSError, ValueError) as e:
        logger.error("Failed to load breast cancer dataset from scikit-learn.")
        raise RuntimeError("load_breast_cancer() failed.") from e

    X: NDArray[np.floating] = np.asarray(data.data, dtype=np.float32)
    y: NDArray[np.signedinteger] = np.asarray(data.target, dtype=np.int64)
    names = np.asarray(data.feature_names, dtype=object).astype(str)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )
    except ValueError as e:
        logger.error("Stratified split failed. Check y and test_size.")
        raise

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_test_s = scaler.transform(X_test).astype(np.float32, copy=False)

    logger.info(
        "Data loaded: n_features=%d, n_train=%d, n_test=%d (random_state=%s).",
        X_train_s.shape[1],
        X_train_s.shape[0],
        X_test_s.shape[0],
        random_state,
    )
    return X_train_s, X_test_s, y_train, y_test, names, scaler
