"""Estimators used in the shape-adaptive robust regression study.

The solver implementation is byte-identical to the validated v4 study module.
The paper's application and simulation workflows are held separately.
"""

from .estimators import (
    QuantileLasso,
    ShapeAdaptiveEN,
    ShapePenalizedEN,
    diagnostic_summary,
    fit_en_cv,
    fit_quantile_lasso_cv,
    fit_sken_cv,
)

__version__ = "0.1.0"

__all__ = [
    "QuantileLasso",
    "ShapeAdaptiveEN",
    "ShapePenalizedEN",
    "diagnostic_summary",
    "fit_en_cv",
    "fit_quantile_lasso_cv",
    "fit_sken_cv",
]
