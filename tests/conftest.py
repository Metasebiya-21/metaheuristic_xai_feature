import warnings

from sklearn.exceptions import ConvergenceWarning

try:
    # pytest exposes this warning class when available
    from _pytest.cacheprovider import PytestCacheWarning
except Exception:
    PytestCacheWarning = None


def pytest_configure(config):
    # Ignore SHAP pending deprecation warnings and sklearn future/convergence noise
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning, module="shap")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    if PytestCacheWarning is not None:
        warnings.filterwarnings("ignore", category=PytestCacheWarning)
