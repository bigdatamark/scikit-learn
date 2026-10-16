# [MODULE DOC]
# Discovery of public scikit-learn objects by walking all importable sklearn
# submodules (skipping private modules and those in _MODULE_TO_IGNORE): estimators,
# displays and functions. Calling these functions imports every public submodule.
"""Utilities to discover scikit-learn objects."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import pkgutil
from importlib import import_module
from operator import itemgetter
from pathlib import Path

_MODULE_TO_IGNORE = {
    "tests",
    "externals",
    "setup",
    "conftest",
    "experimental",
    "estimator_checks",
}


def all_estimators(type_filter=None):
    # [FUNCTION DOC]
    # Behavior: walk all sklearn packages, skipping module names that contain an
    # ignored part or a "._" segment; import each module under
    # ignore_warnings(FutureWarning); collect public classes; keep subclasses of
    # BaseEstimator (excluding BaseEstimator itself) that are not abstract. If
    # type_filter is given (str or list), keep estimators matching any of the
    # requested mixins (classifier, regressor, transformer, cluster).
    # Inputs: type_filter - None, str or list of str.
    # Outputs: list of (name, class) tuples, deduplicated, sorted by name.
    # Exceptions: ValueError for unknown type_filter entries; ImportError from modules
    # that fail to import.
    # Side effects: imports many modules; the input list is copied, not mutated.
    """Get a list of all estimators from `sklearn`.

    This function crawls the module and gets all classes that inherit
    from BaseEstimator. Classes that are defined in test-modules are not
    included.

    Parameters
    ----------
    type_filter : {"classifier", "regressor", "cluster", "transformer"} \
            or list of such str, default=None
        Which kind of estimators should be returned. If None, no filter is
        applied and all estimators are returned.  Possible values are
        'classifier', 'regressor', 'cluster' and 'transformer' to get
        estimators only of these specific types, or a list of these to
        get the estimators that fit at least one of the types.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.

    Examples
    --------
    >>> from sklearn.utils.discovery import all_estimators
    >>> estimators = all_estimators()
    >>> type(estimators)
    <class 'list'>
    >>> type(estimators[0])
    <class 'tuple'>
    >>> estimators[:2]
    [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>),
     ('AdaBoostClassifier',
      <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>)]
    >>> classifiers = all_estimators(type_filter="classifier")
    >>> classifiers[:2]
    [('AdaBoostClassifier',
      <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>),
     ('BaggingClassifier', <class 'sklearn.ensemble._bagging.BaggingClassifier'>)]
    >>> regressors = all_estimators(type_filter="regressor")
    >>> regressors[:2]
    [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>),
     ('AdaBoostRegressor',
      <class 'sklearn.ensemble._weight_boosting.AdaBoostRegressor'>)]
    >>> both = all_estimators(type_filter=["classifier", "regressor"])
    >>> both[:2]
    [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>),
     ('AdaBoostClassifier',
      <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>)]
    """
    # lazy import to avoid circular imports from sklearn.base
    from sklearn.base import (
        BaseEstimator,
        ClassifierMixin,
        ClusterMixin,
        RegressorMixin,
        TransformerMixin,
    )
    from sklearn.utils._testing import ignore_warnings

    def is_abstract(c):
        # [FUNCTION DOC]
        # HELPER: True when the class has a non-empty __abstractmethods__ set.
        if not (hasattr(c, "__abstractmethods__")):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True

    all_classes = []
    root = str(Path(__file__).parent.parent)  # sklearn package
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(path=[root], prefix="sklearn."):
            module_parts = module_name.split(".")
            if (
                any(part in _MODULE_TO_IGNORE for part in module_parts)
                or "._" in module_name
            ):
                continue
            module = import_module(module_name)
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [
                (name, est_cls) for name, est_cls in classes if not name.startswith("_")
            ]

            all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [
        c
        for c in all_classes
        if (issubclass(c[1], BaseEstimator) and c[0] != "BaseEstimator")
    ]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []
        filters = {
            "classifier": ClassifierMixin,
            "regressor": RegressorMixin,
            "transformer": TransformerMixin,
            "cluster": ClusterMixin,
        }
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend(
                    [est for est in estimators if issubclass(est[1], mixin)]
                )
        estimators = filtered_estimators
        if type_filter:
            raise ValueError(
                "Parameter type_filter must be 'classifier', "
                "'regressor', 'transformer', 'cluster' or "
                "None, got"
                f" {type_filter!r}."
            )

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))


def all_displays():
    # [FUNCTION DOC]
    # Behavior: same module walk as all_estimators, collecting public classes whose
    # name ends with "Display".
    # Outputs: list of (name, class) tuples, deduplicated, sorted by name.
    # Side effects: imports many modules; FutureWarnings are suppressed.
    """Get a list of all displays from `sklearn`.

    Returns
    -------
    displays : list of tuples
        List of (name, class), where ``name`` is the display class name as
        string and ``class`` is the actual type of the class.

    Examples
    --------
    >>> from sklearn.utils.discovery import all_displays
    >>> displays = all_displays()
    >>> displays[0]
    ('CalibrationDisplay', <class 'sklearn.calibration.CalibrationDisplay'>)
    """
    # lazy import to avoid circular imports from sklearn.base
    from sklearn.utils._testing import ignore_warnings

    all_classes = []
    root = str(Path(__file__).parent.parent)  # sklearn package
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(path=[root], prefix="sklearn."):
            module_parts = module_name.split(".")
            if (
                any(part in _MODULE_TO_IGNORE for part in module_parts)
                or "._" in module_name
            ):
                continue
            module = import_module(module_name)
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [
                (name, display_class)
                for name, display_class in classes
                if not name.startswith("_") and name.endswith("Display")
            ]
            all_classes.extend(classes)

    return sorted(set(all_classes), key=itemgetter(0))


def _is_checked_function(item):
    # [FUNCTION DOC]
    # HELPER: Predicate for inspect.getmembers: True for plain Python functions with a
    # public name defined in an sklearn.* module other than one ending with
    # "estimator_checks".
    if not inspect.isfunction(item):
        return False

    if item.__name__.startswith("_"):
        return False

    mod = item.__module__
    if not mod.startswith("sklearn.") or mod.endswith("estimator_checks"):
        return False

    return True


def all_functions():
    # [FUNCTION DOC]
    # Behavior: same module walk, collecting public functions accepted by
    # _is_checked_function, keyed by their __name__.
    # Outputs: list of (name, function), deduplicated, sorted by name.
    # Side effects: imports many modules; FutureWarnings are suppressed.
    # KNOWN ISSUE: the attribute name is filtered for a leading underscore but the
    # function's __name__ is returned, so aliases appear under their original name.
    """Get a list of all functions from `sklearn`.

    Returns
    -------
    functions : list of tuples
        List of (name, function), where ``name`` is the function name as
        string and ``function`` is the actual function.

    Examples
    --------
    >>> from sklearn.utils.discovery import all_functions
    >>> functions = all_functions()
    >>> name, function = functions[0]
    >>> name
    'accuracy_score'
    """
    # lazy import to avoid circular imports from sklearn.base
    from sklearn.utils._testing import ignore_warnings

    all_functions = []
    root = str(Path(__file__).parent.parent)  # sklearn package
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(path=[root], prefix="sklearn."):
            module_parts = module_name.split(".")
            if (
                any(part in _MODULE_TO_IGNORE for part in module_parts)
                or "._" in module_name
            ):
                continue

            module = import_module(module_name)
            functions = inspect.getmembers(module, _is_checked_function)
            functions = [
                (func.__name__, func)
                for name, func in functions
                if not name.startswith("_")
            ]
            all_functions.extend(functions)

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(all_functions), key=itemgetter(0))
