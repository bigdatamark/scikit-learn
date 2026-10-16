# [PACKAGE DOC]
# Private subpackage implementing Generalized Linear Models (GLMs) with a Newton-type
#   solver and lbfgs.
# Re-exports from glm.py: the public estimators PoissonRegressor, GammaRegressor,
#   TweedieRegressor and the private base _GeneralizedLinearRegressor (listed in __all__
#   so linear_model can import it).
# Import-time side effects: none beyond importing glm.py.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.linear_model._glm.glm import (
    GammaRegressor,
    PoissonRegressor,
    TweedieRegressor,
    _GeneralizedLinearRegressor,
)

__all__ = [
    "GammaRegressor",
    "PoissonRegressor",
    "TweedieRegressor",
    "_GeneralizedLinearRegressor",
]
