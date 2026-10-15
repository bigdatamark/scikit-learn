# [PACKAGE DOC]
# Cython components of sklearn.linear_model (all compiled via meson.build):
# - _cd_fast.pyx: coordinate descent solvers with duality-gap stopping and Gap Safe
#   screening for Lasso/ElasticNet (dense, sparse CSC with implicit centering and sample
#   weights, precomputed Gram) and their multi-task L2,1 variants; called from
#   _coordinate_descent.py.
# - _sag_fast.pyx.tp: Tempita template producing sag64/sag32, the SAG and SAGA solvers with
#   just-in-time (lazy) sparse updates and an L1 proximal step, called from _sag.py for Ridge
#   and LogisticRegression.
# - _sgd_fast.pyx.tp: Tempita template producing _plain_sgd64/_plain_sgd32, the SGD loop with
#   L2/L1/elastic-net penalties, averaging, learning-rate schedules, Passive-Aggressive steps
#   and early stopping, plus hinge/Huber/epsilon-insensitive loss types; used by
#   _stochastic_gradient.py and _passive_aggressive.py.
# The SAG and SGD modules rely on sklearn.utils._seq_dataset (sample iteration),
# sklearn.utils._weight_vector (SGD only) and sklearn._loss loss types; _cd_fast relies on
# the BLAS wrappers in sklearn.utils._cython_blas. The three modules are independent of
# each other.
"""A variety of linear models."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# See http://scikit-learn.sourceforge.net/modules/sgd.html and
# http://scikit-learn.sourceforge.net/modules/linear_model.html for
# complete documentation.

from sklearn.linear_model._base import LinearRegression
from sklearn.linear_model._bayes import ARDRegression, BayesianRidge
from sklearn.linear_model._coordinate_descent import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
    MultiTaskLasso,
    MultiTaskLassoCV,
    enet_path,
    lasso_path,
)
from sklearn.linear_model._glm import GammaRegressor, PoissonRegressor, TweedieRegressor
from sklearn.linear_model._huber import HuberRegressor
from sklearn.linear_model._least_angle import (
    Lars,
    LarsCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    lars_path,
    lars_path_gram,
)
from sklearn.linear_model._logistic import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model._omp import (
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    orthogonal_mp,
    orthogonal_mp_gram,
)
from sklearn.linear_model._passive_aggressive import (
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
)
from sklearn.linear_model._perceptron import Perceptron
from sklearn.linear_model._quantile import QuantileRegressor
from sklearn.linear_model._ransac import RANSACRegressor
from sklearn.linear_model._ridge import (
    Ridge,
    RidgeClassifier,
    RidgeClassifierCV,
    RidgeCV,
    ridge_regression,
)
from sklearn.linear_model._stochastic_gradient import (
    SGDClassifier,
    SGDOneClassSVM,
    SGDRegressor,
)
from sklearn.linear_model._theil_sen import TheilSenRegressor

__all__ = [
    "ARDRegression",
    "BayesianRidge",
    "ElasticNet",
    "ElasticNetCV",
    "GammaRegressor",
    "HuberRegressor",
    "Lars",
    "LarsCV",
    "Lasso",
    "LassoCV",
    "LassoLars",
    "LassoLarsCV",
    "LassoLarsIC",
    "LinearRegression",
    "LogisticRegression",
    "LogisticRegressionCV",
    "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
    "MultiTaskLasso",
    "MultiTaskLassoCV",
    "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV",
    "PassiveAggressiveClassifier",
    "PassiveAggressiveRegressor",
    "Perceptron",
    "PoissonRegressor",
    "QuantileRegressor",
    "RANSACRegressor",
    "Ridge",
    "RidgeCV",
    "RidgeClassifier",
    "RidgeClassifierCV",
    "SGDClassifier",
    "SGDOneClassSVM",
    "SGDRegressor",
    "TheilSenRegressor",
    "TweedieRegressor",
    "enet_path",
    "lars_path",
    "lars_path_gram",
    "lasso_path",
    "orthogonal_mp",
    "orthogonal_mp_gram",
    "ridge_regression",
]
