# [PACKAGE DOC]
# Public namespace for scikit-learn's linear models. Re-exports estimators and path
# functions from private submodules: ordinary least squares (LinearRegression), Bayesian
# (ARDRegression, BayesianRidge), coordinate-descent L1/L2 models and their CV/multi-
# task
# variants plus enet_path/lasso_path, GLMs (Gamma/Poisson/Tweedie), HuberRegressor, LARS
# family plus lars_path/lars_path_gram, LogisticRegression(CV), OMP and
# orthogonal_mp(_gram), passive-aggressive models, Perceptron, QuantileRegressor,
# RANSACRegressor, Ridge family plus ridge_regression, SGD models (classifier,
# regressor,
# one-class SVM) and TheilSenRegressor. __all__ lists these names.
# Side effects: importing loads all those submodules (including compiled Cython
# extensions); no other global state is changed.
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
