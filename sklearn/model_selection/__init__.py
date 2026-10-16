# [PACKAGE DOC]
# Public package for model selection: re-exports cross-validation splitters (KFold,
#   StratifiedKFold, GroupKFold, ShuffleSplit, TimeSeriesSplit, etc., check_cv,
#   train_test_split), hyper-parameter search (GridSearchCV, RandomizedSearchCV,
#   ParameterGrid, ParameterSampler), validation utilities (cross_validate,
#   cross_val_score, cross_val_predict, learning_curve, validation_curve,
#   permutation_test_score), decision-threshold tuning (FixedThresholdClassifier,
#   TunedThresholdClassifierCV) and curve displays (LearningCurveDisplay,
#   ValidationCurveDisplay).
# Import-time side effects: imports those submodules. The experimental
#   HalvingGridSearchCV / HalvingRandomSearchCV are only imported for type checkers;
#   they are listed in __all__ but accessing them here raises ImportError (via module
#   __getattr__) unless sklearn.experimental.enable_halving_search_cv has been imported,
#   which injects them into this module.
# KNOWN ISSUE: because the Halving names are in __all__, a star import fails with
#   ImportError unless the experimental enabler was imported first.
"""Tools for model selection, such as cross validation and hyper-parameter tuning."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import typing

from sklearn.model_selection._classification_threshold import (
    FixedThresholdClassifier,
    TunedThresholdClassifierCV,
)
from sklearn.model_selection._plot import LearningCurveDisplay, ValidationCurveDisplay
from sklearn.model_selection._search import (
    GridSearchCV,
    ParameterGrid,
    ParameterSampler,
    RandomizedSearchCV,
)
from sklearn.model_selection._split import (
    BaseCrossValidator,
    BaseShuffleSplit,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePGroupsOut,
    LeavePOut,
    PredefinedSplit,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    check_cv,
    train_test_split,
)
from sklearn.model_selection._validation import (
    cross_val_predict,
    cross_val_score,
    cross_validate,
    learning_curve,
    permutation_test_score,
    validation_curve,
)

if typing.TYPE_CHECKING:
    # Avoid errors in type checkers (e.g. pyrefly) for experimental estimators.
    # TODO: remove this check once the estimator is no longer experimental.
    from sklearn.model_selection._search_successive_halving import (
        HalvingGridSearchCV,
        HalvingRandomSearchCV,
    )


__all__ = [
    "BaseCrossValidator",
    "BaseShuffleSplit",
    "FixedThresholdClassifier",
    "GridSearchCV",
    "GroupKFold",
    "GroupShuffleSplit",
    "HalvingGridSearchCV",
    "HalvingRandomSearchCV",
    "KFold",
    "LearningCurveDisplay",
    "LeaveOneGroupOut",
    "LeaveOneOut",
    "LeavePGroupsOut",
    "LeavePOut",
    "ParameterGrid",
    "ParameterSampler",
    "PredefinedSplit",
    "RandomizedSearchCV",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "ShuffleSplit",
    "StratifiedGroupKFold",
    "StratifiedKFold",
    "StratifiedShuffleSplit",
    "TimeSeriesSplit",
    "TunedThresholdClassifierCV",
    "ValidationCurveDisplay",
    "check_cv",
    "cross_val_predict",
    "cross_val_score",
    "cross_validate",
    "learning_curve",
    "permutation_test_score",
    "train_test_split",
    "validation_curve",
]


# TODO: remove this check once the estimator is no longer experimental.
def __getattr__(name):
    # [FUNCTION DOC]
    # Module-level attribute fallback (PEP 562), called only for names not found
    #   normally.
    # Inputs: name - str.
    # Exceptions: ImportError with instructions to import enable_halving_search_cv for
    #   the two experimental halving search names; AttributeError for any other name.
    if name in {"HalvingGridSearchCV", "HalvingRandomSearchCV"}:
        raise ImportError(
            f"{name} is experimental and the API might change without any "
            "deprecation cycle. To use it, you need to explicitly import "
            "enable_halving_search_cv:\n"
            "from sklearn.experimental import enable_halving_search_cv"
        )
    raise AttributeError(f"module {__name__} has no attribute {name}")
