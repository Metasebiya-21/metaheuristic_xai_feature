"""Feature selection methods: shared base class plus classical/XAI and metaheuristic families."""

from metaheuristic_xai.selectors.base import FeatureSelector
from metaheuristic_xai.selectors.classical import (
    BorutaSelector,
    LassoSelector,
    LIMESelector,
    RFESelector,
    SHAPSelector,
)
from metaheuristic_xai.selectors.metaheuristic import (
    SELECTOR_REGISTRY,
    BPSOSelector,
    GASelector,
    GWOSelector,
    NSGA2Selector,
    SASelector,
    get_feature_selector,
)

__all__ = [
    "BPSOSelector",
    "BorutaSelector",
    "FeatureSelector",
    "GASelector",
    "GWOSelector",
    "LIMESelector",
    "LassoSelector",
    "NSGA2Selector",
    "RFESelector",
    "SASelector",
    "SELECTOR_REGISTRY",
    "SHAPSelector",
    "get_feature_selector",
]
