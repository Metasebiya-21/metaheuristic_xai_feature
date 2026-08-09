"""
Unit tests for classifier adapters.
"""

from __future__ import annotations

import pytest

from metaheuristic_xai.classifiers import CLASSIFIER_REGISTRY, get_classifier_adapter
from metaheuristic_xai.datasets import load_dataset


@pytest.mark.parametrize("c_name", list(CLASSIFIER_REGISTRY.keys()))
def test_classifier_adapters(c_name: str) -> None:
    bundle = load_dataset("wdbc", random_state=42)
    adapter = get_classifier_adapter(c_name)

    adapter.fit(bundle.X_train, bundle.y_train)
    preds = adapter.predict(bundle.X_test)
    probas = adapter.predict_proba(bundle.X_test)

    assert len(preds) == len(bundle.y_test)
    assert probas.shape == (len(bundle.y_test), 2)
