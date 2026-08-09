"""
Dataset Registry and Preprocessing module for feature-selection benchmark.

Supported benchmark tiers:
- Low dimensional: D < 100
- Medium dimensional: 100 <= D <= 1000
- Ultra-high dimensional: D > 1000

Important methodological guarantees:
1. Only real public datasets are used in benchmark experiments.
2. No synthetic fallback is used when OpenML is unavailable.
3. Train/test split occurs before supervised preprocessing.
4. Categorical encoders are fitted on training data only.
5. Imputation is fitted on training data only.
6. Variance filtering is fitted on training data only.
7. StandardScaler is fitted on training data only.
8. Preliminary SelectKBest filtering is fitted on training data only.
9. Raw, preprocessed, and wrapper-search dimensionality are tracked separately.

The distinction between:
    raw_d
    preprocessed_d
    filtered_d

is important for reporting experiments on ultra-high-dimensional datasets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ============================================================
# Types
# ============================================================

ScaleTier = Literal["low", "medium", "ultrahigh"]


# ============================================================
# Dataset configuration
# ============================================================

@dataclass(frozen=True)
class DatasetConfig:
    """
    Metadata and configuration for a benchmark dataset.
    """

    name: str
    source: str

    task: Literal["binary", "multiclass"]

    target_column: Optional[str] = None

    scale_tier: ScaleTier = "low"

    openml_id: Optional[int] = None

    description: str = ""

    # Preliminary supervised filter used ONLY for extremely
    # high-dimensional wrapper experiments.
    #
    # IMPORTANT:
    # raw_d remains the original dimensionality.
    # filtered_d is the actual dimensionality searched by wrappers.
    prefilter_k: Optional[int] = None


# ============================================================
# Dataset registry
# ============================================================

DATASET_REGISTRY: dict[str, DatasetConfig] = {

    # --------------------------------------------------------
    # Low dimensional
    # --------------------------------------------------------

    "wdbc": DatasetConfig(
        name="wdbc",
        source="scikit-learn / UCI Wisconsin Diagnostic Breast Cancer",
        task="binary",
        target_column="target",
        scale_tier="low",
        description=(
            "Wisconsin Diagnostic Breast Cancer dataset "
            "(N=569, D=30)."
        ),
    ),

    "ionosphere": DatasetConfig(
        name="ionosphere",
        source="OpenML / UCI",
        task="binary",
        target_column="class",
        scale_tier="low",
        openml_id=59,
        description=(
            "Ionosphere radar classification dataset "
            "(N=351, D=34)."
        ),
    ),

    # --------------------------------------------------------
    # Medium dimensional
    # --------------------------------------------------------

    "madelon": DatasetConfig(
        name="madelon",
        source="OpenML / NIPS 2003",
        task="binary",
        target_column="Class",
        scale_tier="medium",
        openml_id=1485,
        description=(
            "MADELON feature-selection benchmark "
            "(approximately N=2600, D=500)."
        ),
    ),

    # --------------------------------------------------------
    # Ultra-high dimensional
    # --------------------------------------------------------

    "colon": DatasetConfig(
        name="colon",
        source="OpenML / Alon et al.",
        task="binary",
        target_column="class",
        scale_tier="ultrahigh",
        openml_id=45087,
        prefilter_k=300,
        description=(
            "Colon cancer gene-expression dataset "
            "(N=62, D=2000)."
        ),
    ),

    "leukemia": DatasetConfig(
        name="leukemia",
        source="OpenML / Golub et al.",
        task="binary",
        target_column="Class",
        scale_tier="ultrahigh",
        openml_id=45090,
        prefilter_k=300,
        description=(
            "Leukemia gene-expression dataset "
            "(approximately N=72, D=7129)."
        ),
    ),
}


# ============================================================
# Dataset bundle
# ============================================================

@dataclass
class DatasetBundle:
    """
    Container returned by the dataset loader.

    raw_d:
        Number of features in the original public dataset.

    preprocessed_d:
        Number of features after train-only preprocessing and
        zero-variance removal.

    filtered_d:
        Number of features actually passed to the feature-selection
        algorithms after optional preliminary filtering.

    This distinction is essential for ultra-high-dimensional
    experiments.
    """

    name: str

    X_train: NDArray[np.float32]
    X_test: NDArray[np.float32]

    y_train: NDArray[np.int64]
    y_test: NDArray[np.int64]

    feature_names: NDArray[np.str_]

    raw_d: int
    preprocessed_d: int
    filtered_d: int

    scaler: StandardScaler

    config: DatasetConfig

    class_distribution: dict[str, int] = field(
        default_factory=dict
    )

    # Useful metadata for experiment logging.
    train_size: int = 0
    test_size: int = 0

    class_labels: list[str] = field(
        default_factory=list
    )


# ============================================================
# Dataset validation
# ============================================================

def _validate_dataset(
    X_df: pd.DataFrame,
    y_s: pd.Series,
    config: DatasetConfig,
) -> None:
    """
    Validate a loaded public dataset before preprocessing.
    """

    if X_df is None or X_df.empty:
        raise ValueError(
            f"Dataset '{config.name}' contains no features."
        )

    if y_s is None or len(y_s) == 0:
        raise ValueError(
            f"Dataset '{config.name}' contains no target values."
        )

    if len(X_df) != len(y_s):
        raise ValueError(
            f"Dataset '{config.name}' has inconsistent lengths: "
            f"X={len(X_df)}, y={len(y_s)}."
        )

    if X_df.columns.duplicated().any():
        raise ValueError(
            f"Dataset '{config.name}' contains duplicate feature names."
        )

    n_classes = y_s.nunique(dropna=True)

    if config.task == "binary" and n_classes != 2:
        raise ValueError(
            f"Dataset '{config.name}' is configured as binary "
            f"but contains {n_classes} classes."
        )

    if n_classes < 2:
        raise ValueError(
            f"Dataset '{config.name}' contains fewer than two classes."
        )

    logger.info(
        "Validated dataset '%s': N=%d, D=%d, classes=%d",
        config.name,
        len(X_df),
        X_df.shape[1],
        n_classes,
    )


# ============================================================
# OpenML loader
# ============================================================

def _load_openml_dataset(
    config: DatasetConfig,
    cache_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a real dataset from OpenML.

    IMPORTANT:
    There is deliberately NO synthetic fallback.

    If OpenML cannot be reached, the experiment fails instead of
    silently replacing the public benchmark with synthetic data.
    """

    if config.openml_id is None:
        raise ValueError(
            f"Dataset '{config.name}' does not have an OpenML ID."
        )

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    logger.info(
        "Fetching '%s' from OpenML (data_id=%s)...",
        config.name,
        config.openml_id,
    )

    try:
        data = fetch_openml(
            data_id=config.openml_id,
            as_frame=True,
            parser="auto",
            return_X_y=False,
            data_home=str(cache_dir) if cache_dir else None,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load the real OpenML dataset "
            f"'{config.name}' (data_id={config.openml_id}). "
            f"The benchmark does not use synthetic fallbacks. "
            f"Check network connectivity or the local OpenML cache."
        ) from exc

    X_df = data.data
    y_s = data.target

    if isinstance(X_df, np.ndarray):
        X_df = pd.DataFrame(X_df)

    if isinstance(y_s, pd.DataFrame):
        y_s = y_s.iloc[:, 0]

    if not isinstance(y_s, pd.Series):
        y_s = pd.Series(
            np.asarray(y_s).ravel(),
            name=config.target_column or "target",
        )

    X_df = X_df.copy()
    y_s = y_s.copy()

    logger.info(
        "Loaded '%s' from OpenML: N=%d, D=%d",
        config.name,
        len(X_df),
        X_df.shape[1],
    )

    return X_df, y_s


# ============================================================
# Raw dataset loader
# ============================================================

def load_raw_dataset(
    name: str,
    cache_dir: Optional[Path] = None,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    DatasetConfig,
]:
    """
    Load one of the registered public datasets.

    WDBC is loaded through scikit-learn.

    All other registered datasets are loaded from OpenML.

    No synthetic fallback is permitted.
    """

    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available datasets: "
            f"{list(DATASET_REGISTRY.keys())}"
        )

    config = DATASET_REGISTRY[name]

    logger.info(
        "Loading dataset '%s' [%s]",
        config.name,
        config.scale_tier,
    )

    # --------------------------------------------------------
    # WDBC
    # --------------------------------------------------------

    if config.name == "wdbc":

        breast_cancer = load_breast_cancer()

        X_df = pd.DataFrame(
            breast_cancer.data,
            columns=breast_cancer.feature_names,
        )

        y_s = pd.Series(
            breast_cancer.target,
            name="target",
        )

    # --------------------------------------------------------
    # OpenML
    # --------------------------------------------------------

    else:

        X_df, y_s = _load_openml_dataset(
            config=config,
            cache_dir=cache_dir,
        )

    _validate_dataset(
        X_df=X_df,
        y_s=y_s,
        config=config,
    )

    return X_df, y_s, config


# ============================================================
# Target encoding
# ============================================================

def _encode_target(
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    list[str],
]:
    """
    Encode target labels.

    Target encoding does not use feature information, but fitting
    the encoder on training labels and applying it to test labels
    keeps the train/test transformation explicit.
    """

    encoder = LabelEncoder()

    y_train_encoded = encoder.fit_transform(
        y_train.astype(str)
    ).astype(np.int64)

    unseen_test_labels = set(
        y_test.astype(str).unique()
    ) - set(
        encoder.classes_
    )

    if unseen_test_labels:
        raise ValueError(
            "Test set contains target classes not present "
            f"in training set: {unseen_test_labels}"
        )

    y_test_encoded = encoder.transform(
        y_test.astype(str)
    ).astype(np.int64)

    class_labels = [
        str(label)
        for label in encoder.classes_
    ]

    return (
        y_train_encoded,
        y_test_encoded,
        class_labels,
    )


# ============================================================
# Feature preprocessing
# ============================================================

def _prepare_feature_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.str_],
]:
    """
    Encode categorical features and impute missing values.

    All transformations are fitted using X_train only.

    Numeric:
        median imputation

    Categorical:
        most-frequent imputation
        ordinal encoding

    Unknown test categories are encoded as -1.
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    feature_names = np.asarray(
        X_train.columns,
        dtype=str,
    )

    numeric_columns = [
        col
        for col in X_train.columns
        if pd.api.types.is_numeric_dtype(
            X_train[col]
        )
    ]

    categorical_columns = [
        col
        for col in X_train.columns
        if col not in numeric_columns
    ]

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    if numeric_columns:

        numeric_imputer = SimpleImputer(
            strategy="median"
        )

        X_train_numeric = numeric_imputer.fit_transform(
            X_train[numeric_columns]
        )

        X_test_numeric = numeric_imputer.transform(
            X_test[numeric_columns]
        )

        train_parts.append(
            X_train_numeric
        )

        test_parts.append(
            X_test_numeric
        )

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    if categorical_columns:

        categorical_imputer = SimpleImputer(
            strategy="most_frequent"
        )

        X_train_cat = categorical_imputer.fit_transform(
            X_train[categorical_columns].astype(str)
        )

        X_test_cat = categorical_imputer.transform(
            X_test[categorical_columns].astype(str)
        )

        categorical_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        X_train_cat_encoded = (
            categorical_encoder.fit_transform(
                X_train_cat
            )
        )

        X_test_cat_encoded = (
            categorical_encoder.transform(
                X_test_cat
            )
        )

        train_parts.append(
            X_train_cat_encoded
        )

        test_parts.append(
            X_test_cat_encoded
        )

    if not train_parts:
        raise ValueError(
            "No usable feature columns were found."
        )

    X_train_processed = np.hstack(
        train_parts
    )

    X_test_processed = np.hstack(
        test_parts
    )

    return (
        X_train_processed,
        X_test_processed,
        feature_names,
    )


# ============================================================
# Main preprocessing pipeline
# ============================================================

def preprocess_and_split(
    X_df: pd.DataFrame,
    y_s: pd.Series,
    config: DatasetConfig,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_prefilter: bool = True,
    indices: tuple[NDArray[np.signedinteger], NDArray[np.signedinteger]] | None = None,
) -> DatasetBundle:
    """
    Split, preprocess, scale, and optionally pre-filter a dataset.

    ``indices``, when given, is ``(train_idx, test_idx)`` -- positional row
    indices into ``X_df``/``y_s`` after the missing-target rows below are
    dropped -- and is used verbatim instead of a random ``train_test_split``.
    This is how disjoint outer folds (e.g. proper k-fold CV, as opposed to
    repeated independent holdout splits that can overlap across "seeds") are
    built: pass each fold's own row indices and every downstream fit-on-train
    step (imputation, scaling, prefilter) still only ever touches that fold's
    training rows.

    IMPORTANT ORDER:

        Raw dataset
             |
             v
        Train/test split
             |
             v
        Train-only preprocessing
             |
             v
        Train-only variance filtering
             |
             v
        Train-only scaling
             |
             v
        Train-only preliminary feature filtering
             |
             v
        Feature-selection algorithms

    The test set never participates in supervised feature selection.
    """

    _validate_dataset(
        X_df=X_df,
        y_s=y_s,
        config=config,
    )

    raw_d = int(
        X_df.shape[1]
    )

    # --------------------------------------------------------
    # Remove rows with missing target
    # --------------------------------------------------------

    valid_target_mask = ~y_s.isna()

    X_df = X_df.loc[
        valid_target_mask
    ].reset_index(drop=True)

    y_s = y_s.loc[
        valid_target_mask
    ].reset_index(drop=True)

    # --------------------------------------------------------
    # Train/test split BEFORE feature preprocessing
    # --------------------------------------------------------

    if indices is not None:
        train_idx, test_idx = indices
        X_train_df = X_df.iloc[train_idx]
        X_test_df = X_df.iloc[test_idx]
        y_train_raw = y_s.iloc[train_idx]
        y_test_raw = y_s.iloc[test_idx]
    else:
        (
            X_train_df,
            X_test_df,
            y_train_raw,
            y_test_raw,
        ) = train_test_split(
            X_df,
            y_s,
            test_size=test_size,
            stratify=y_s,
            random_state=random_state,
        )

    X_train_df = X_train_df.reset_index(
        drop=True
    )

    X_test_df = X_test_df.reset_index(
        drop=True
    )

    y_train_raw = y_train_raw.reset_index(
        drop=True
    )

    y_test_raw = y_test_raw.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    (
        y_train,
        y_test,
        class_labels,
    ) = _encode_target(
        y_train_raw,
        y_test_raw,
    )

    class_dist = {
        str(label): int(
            np.sum(
                y_train == idx
            )
        )
        for idx, label in enumerate(
            class_labels
        )
    }

    # --------------------------------------------------------
    # Feature preprocessing
    # FIT ONLY ON TRAIN
    # --------------------------------------------------------

    (
        X_train_processed,
        X_test_processed,
        feature_names,
    ) = _prepare_feature_columns(
        X_train_df,
        X_test_df,
    )

    # Ensure numeric finite representation.
    X_train_processed = np.asarray(
        X_train_processed,
        dtype=np.float64,
    )

    X_test_processed = np.asarray(
        X_test_processed,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Variance filtering
    # FIT ONLY ON TRAIN
    # --------------------------------------------------------

    variance_selector = VarianceThreshold(
        threshold=1e-6
    )

    try:

        X_train_var = (
            variance_selector.fit_transform(
                X_train_processed
            )
        )

        X_test_var = (
            variance_selector.transform(
                X_test_processed
            )
        )

        variance_mask = (
            variance_selector.get_support()
        )

        feature_names = feature_names[
            variance_mask
        ]

    except ValueError:

        # This can occur if every feature has zero variance.
        raise ValueError(
            f"Dataset '{config.name}' has no "
            "non-constant features after preprocessing."
        )

    preprocessed_d = int(
        X_train_var.shape[1]
    )

    # --------------------------------------------------------
    # Standardization
    # FIT ONLY ON TRAIN
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train_var
        ).astype(
            np.float32,
            copy=False,
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test_var
        ).astype(
            np.float32,
            copy=False,
        )
    )

    # --------------------------------------------------------
    # Preliminary supervised filtering
    # FIT ONLY ON TRAIN
    # --------------------------------------------------------

    filtered_d = preprocessed_d

    if (
        apply_prefilter
        and config.prefilter_k is not None
        and preprocessed_d > config.prefilter_k
    ):

        k_filt = min(
            int(config.prefilter_k),
            preprocessed_d,
        )

        if k_filt < 1:
            raise ValueError(
                "prefilter_k must be >= 1."
            )

        filter_selector = SelectKBest(
            score_func=f_classif,
            k=k_filt,
        )

        X_train_filtered = (
            filter_selector.fit_transform(
                X_train_scaled,
                y_train,
            )
        )

        X_test_filtered = (
            filter_selector.transform(
                X_test_scaled
            )
        )

        filter_mask = (
            filter_selector.get_support()
        )

        feature_names = feature_names[
            filter_mask
        ]

        X_train_scaled = (
            X_train_filtered.astype(
                np.float32,
                copy=False,
            )
        )

        X_test_scaled = (
            X_test_filtered.astype(
                np.float32,
                copy=False,
            )
        )

        filtered_d = int(
            X_train_scaled.shape[1]
        )

        logger.info(
            "Training-only preliminary filter: "
            "%s: raw D=%d -> preprocessed D=%d "
            "-> wrapper D=%d",
            config.name,
            raw_d,
            preprocessed_d,
            filtered_d,
        )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    assert X_train_scaled.shape[1] == len(
        feature_names
    )

    assert X_test_scaled.shape[1] == len(
        feature_names
    )

    assert X_train_scaled.shape[0] == len(
        y_train
    )

    assert X_test_scaled.shape[0] == len(
        y_test
    )

    if not np.isfinite(
        X_train_scaled
    ).all():
        raise ValueError(
            f"Non-finite values remain in "
            f"training data for '{config.name}'."
        )

    if not np.isfinite(
        X_test_scaled
    ).all():
        raise ValueError(
            f"Non-finite values remain in "
            f"test data for '{config.name}'."
        )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logger.info(
        "Dataset '%s': "
        "raw D=%d, preprocessed D=%d, "
        "wrapper D=%d, train=%d, test=%d",
        config.name,
        raw_d,
        preprocessed_d,
        filtered_d,
        len(y_train),
        len(y_test),
    )

    return DatasetBundle(
        name=config.name,

        X_train=X_train_scaled,
        X_test=X_test_scaled,

        y_train=y_train,
        y_test=y_test,

        feature_names=feature_names,

        raw_d=raw_d,
        preprocessed_d=preprocessed_d,
        filtered_d=filtered_d,

        scaler=scaler,

        config=config,

        class_distribution=class_dist,

        train_size=len(y_train),
        test_size=len(y_test),

        class_labels=class_labels,
    )


# ============================================================
# Public API
# ============================================================

def load_dataset(
    name: str,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_prefilter: bool = True,
    cache_dir: Optional[Path] = None,
) -> DatasetBundle:
    """
    Load and preprocess a registered dataset.
    """

    X_df, y_s, config = load_raw_dataset(
        name=name,
        cache_dir=cache_dir,
    )

    return preprocess_and_split(
        X_df=X_df,
        y_s=y_s,
        config=config,
        test_size=test_size,
        random_state=random_state,
        apply_prefilter=apply_prefilter,
    )


def load_dataset_folds(
    name: str,
    n_splits: int = 5,
    random_state: int = 0,
    apply_prefilter: bool = True,
    cache_dir: Optional[Path] = None,
) -> list[DatasetBundle]:
    """Load a registered dataset as ``n_splits`` disjoint stratified folds.

    Unlike calling :func:`load_dataset` with different ``random_state`` values
    (independent 80/20 holdouts that can and do overlap -- a given row may be a
    training row under one seed and a test row under another), every row here
    appears in **exactly one** fold's test partition across the returned list.
    This is what a nested / cross-validated procedure (e.g. resampling-based
    stability selection) needs to be leakage-free: whatever is built using
    "the other folds" for fold *i* never touches fold *i*'s rows.

    Preprocessing (imputation, variance filter, scaling, prefilter) is still
    fit independently per fold, on that fold's training rows only.
    """
    from sklearn.model_selection import StratifiedKFold

    X_df, y_s, config = load_raw_dataset(name=name, cache_dir=cache_dir)
    valid = ~y_s.isna()
    X_df, y_s = X_df.loc[valid].reset_index(drop=True), y_s.loc[valid].reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    bundles = []
    for train_idx, test_idx in skf.split(X_df, y_s):
        bundles.append(
            preprocess_and_split(
                X_df=X_df, y_s=y_s, config=config,
                apply_prefilter=apply_prefilter, indices=(train_idx, test_idx),
            )
        )
    return bundles


def list_datasets() -> list[dict[str, Any]]:
    """
    Return dataset registry information.
    """

    return [
        {
            "name": config.name,
            "source": config.source,
            "task": config.task,
            "scale_tier": config.scale_tier,
            "description": config.description,
            "openml_id": config.openml_id,
            "prefilter_k": config.prefilter_k,
        }
        for config in DATASET_REGISTRY.values()
    ]


def get_dataset_config(
    name: str,
) -> DatasetConfig:
    """
    Return configuration for one registered dataset.
    """

    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    return DATASET_REGISTRY[name]
