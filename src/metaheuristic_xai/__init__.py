"""
metaheuristic_xai: benchmarking nature-inspired metaheuristics (GA, BPSO, SA, GWO, NSGA-II)
against XAI (SHAP, LIME) and classical (LASSO, RFE, Boruta) feature selection across
five public datasets and five classifier families.
"""

import warnings

from sklearn.exceptions import ConvergenceWarning

# Suppress noisy third-party deprecation/future warnings that don't affect results
warnings.filterwarnings("ignore", category=PendingDeprecationWarning, module="shap")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


__version__ = "1.0.0"
