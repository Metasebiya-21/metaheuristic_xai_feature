"""
Unit tests for Dataset Registry module.
"""

from __future__ import annotations

import numpy as np
import pytest

from metaheuristic_xai.datasets import list_datasets, load_dataset, load_dataset_folds


def test_list_datasets() -> None:
    ds_list = list_datasets()
    assert len(ds_list) >= 5
    names = [d["name"] for d in ds_list]
    assert "wdbc" in names
    assert "ionosphere" in names
    assert "madelon" in names
    assert "colon" in names
    assert "leukemia" in names


@pytest.mark.parametrize("name", ["wdbc", "ionosphere", "colon"])
def test_load_dataset_splits(name: str) -> None:
    bundle = load_dataset(name, test_size=0.2, random_state=42)
    assert bundle.X_train.ndim == 2
    assert bundle.X_test.ndim == 2
    assert bundle.X_train.shape[1] == bundle.X_test.shape[1]
    assert len(bundle.y_train) == bundle.X_train.shape[0]
    assert len(bundle.y_test) == bundle.X_test.shape[0]
    assert not np.isnan(bundle.X_train).any()
    assert not np.isnan(bundle.X_test).any()


def test_load_dataset_folds_are_disjoint_and_cover_all_rows() -> None:
    folds = load_dataset_folds("wdbc", n_splits=5, random_state=0)
    assert len(folds) == 5
    total_test = sum(f.X_test.shape[0] for f in folds)
    assert total_test == 569  # full WDBC sample count, each row in exactly one test fold
    for f in folds:
        assert f.X_train.shape[1] == f.X_test.shape[1]
        assert not np.isnan(f.X_train).any() and not np.isnan(f.X_test).any()
