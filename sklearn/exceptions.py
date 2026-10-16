# [MODULE DOC]
# Defines the custom exception and warning classes shared by all of scikit-learn.
# Contains only class definitions; no functions and no import-time side effects. __all__
# lists the public names.
# KNOWN ISSUE: InconsistentVersionWarning is defined and public but missing from
# __all__, so star-imports do not export it.
"""Custom warnings and errors used across scikit-learn."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "ConvergenceWarning",
    "DataConversionWarning",
    "DataDimensionalityWarning",
    "EfficiencyWarning",
    "EstimatorCheckFailedWarning",
    "FitFailedWarning",
    "NotFittedError",
    "PositiveSpectrumWarning",
    "SkipTestWarning",
    "UndefinedMetricWarning",
    "UnsetMetadataPassedError",
]


class UnsetMetadataPassedError(ValueError):
    # [CLASS DOC]
    # ValueError raised by metadata routing when metadata is passed that a consumer has
    # neither requested nor explicitly declined. Carries the offending and routed
    # parameters as attributes so callers can inspect them.
    """Exception class to raise if a metadata is passed which is not explicitly \
        requested (metadata=True) or not requested (metadata=False).

    .. versionadded:: 1.3

    Parameters
    ----------
    message : str
        The message

    unrequested_params : dict
        A dictionary of parameters and their values which are provided but not
        requested.

    routed_params : dict
        A dictionary of routed parameters.
    """

    def __init__(self, *, message, unrequested_params, routed_params):
        # [FUNCTION DOC]
        # Inputs (keyword-only): message - str used as the exception message;
        # unrequested_params - dict of param name to value passed but not requested;
        # routed_params - dict of params that were routed.
        # Side effects: sets self.unrequested_params and self.routed_params; message
        # stored via ValueError init (available as args[0] and str()).
        super().__init__(message)
        self.unrequested_params = unrequested_params
        self.routed_params = routed_params


class NotFittedError(ValueError, AttributeError):
    # [CLASS DOC]
    # Raised when a method needing a fitted estimator is called before fit. Inherits
    # from both ValueError and AttributeError so that hasattr-style checks and older
    # ValueError handlers both catch it. No extra behavior beyond the base classes.
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    Examples
    --------
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.exceptions import NotFittedError
    >>> try:
    ...     LinearSVC().predict([[1, 2], [2, 3], [3, 4]])
    ... except NotFittedError as e:
    ...     print(repr(e))
    NotFittedError("This LinearSVC instance is not fitted yet. Call 'fit' with
    appropriate arguments before using this estimator."...)

    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation.
    """


class ConvergenceWarning(UserWarning):
    # [CLASS DOC]
    # UserWarning subclass emitted when an iterative solver stops before converging.
    # Marker class only.
    """Custom warning to capture convergence problems

    .. versionchanged:: 0.18
       Moved from sklearn.utils.
    """


class DataConversionWarning(UserWarning):
    # [CLASS DOC]
    # UserWarning subclass emitted when input data is implicitly converted (dtype
    # change, forced copy, ambiguous shape such as a column-vector y). Marker class
    # only.
    """Warning used to notify implicit data conversions happening in the code.

    This warning occurs when some input data needs to be converted or
    interpreted in a way that may not match the user's expectations.

    For example, this warning may occur when the user
        - passes an integer array to a function which expects float input and
          will convert the input
        - requests a non-copying operation, but a copy is required to meet the
          implementation's data-type expectations;
        - passes an input whose shape can be interpreted ambiguously.

    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation.
    """


class DataDimensionalityWarning(UserWarning):
    # [CLASS DOC]
    # UserWarning subclass for questionable dimensionality, e.g. random projection to
    # more components than features. Marker class only.
    """Custom warning to notify potential issues with data dimensionality.

    For example, in random projection, this warning is raised when the
    number of components, which quantifies the dimensionality of the target
    projection space, is higher than the number of features, which quantifies
    the dimensionality of the original source space, to imply that the
    dimensionality of the problem will not be reduced.

    .. versionchanged:: 0.18
       Moved from sklearn.utils.
    """


class EfficiencyWarning(UserWarning):
    # [CLASS DOC]
    # UserWarning subclass flagging computations that will be slow or memory-
    # inefficient; meant to be subclassed for specific cases. Marker class only.
    """Warning used to notify the user of inefficient computation.

    This warning notifies the user that the efficiency may not be optimal due
    to some reason which may be included as a part of the warning message.
    This may be subclassed into a more specific Warning class.

    .. versionadded:: 0.18
    """


class FitFailedWarning(RuntimeWarning):
    # [CLASS DOC]
    # RuntimeWarning subclass emitted by search / cross-validation helpers when fitting
    # the estimator on a split raised an error (and error_score is not "raise"). Marker
    # class only.
    """Warning class used if there is an error while fitting the estimator.

    This Warning is used in meta estimators GridSearchCV and RandomizedSearchCV
    and the cross-validation helper function cross_val_score to warn when there
    is an error while fitting the estimator.

    .. versionchanged:: 0.18
       Moved from sklearn.cross_validation.
    """


class SkipTestWarning(UserWarning):
    # [CLASS DOC]
    # UserWarning subclass used by the estimator check suite to report a skipped check
    # (e.g. an optional dependency such as pandas is missing). Marker class only.
    """Warning class used to notify the user of a test that was skipped.

    For example, one of the estimator checks requires a pandas import.
    If the pandas package cannot be imported, the test will be skipped rather
    than register as a failure.
    """


class UndefinedMetricWarning(UserWarning):
    # [CLASS DOC]
    # UserWarning subclass emitted when a metric is ill-defined for the given input
    # (e.g. zero division in precision) and a fallback value is returned. Marker class
    # only.
    """Warning used when the metric is invalid

    .. versionchanged:: 0.18
       Moved from sklearn.base.
    """


class PositiveSpectrumWarning(UserWarning):
    # [CLASS DOC]
    # UserWarning subclass emitted by _check_psd_eigenvalues when a supposedly PSD
    # matrix has significant negative or badly conditioned tiny eigenvalues. Marker
    # class only.
    """Warning raised when the eigenvalues of a PSD matrix have issues

    This warning is typically raised by ``_check_psd_eigenvalues`` when the
    eigenvalues of a positive semidefinite (PSD) matrix such as a gram matrix
    (kernel) present significant negative eigenvalues, or bad conditioning i.e.
    very small non-zero eigenvalues compared to the largest eigenvalue.

    .. versionadded:: 0.22
    """


class InconsistentVersionWarning(UserWarning):
    # [CLASS DOC]
    # UserWarning emitted when unpickling an estimator pickled with a different scikit-
    # learn version. Stores the versions as attributes and formats its own message in
    # __str__.
    """Warning raised when an estimator is unpickled with an inconsistent version.

    Parameters
    ----------
    estimator_name : str
        Estimator name.

    current_sklearn_version : str
        Current scikit-learn version.

    original_sklearn_version : str
        Original scikit-learn version.
    """

    def __init__(
        self, *, estimator_name, current_sklearn_version, original_sklearn_version
    ):
        # [FUNCTION DOC]
        # Inputs (keyword-only): estimator_name - str class name;
        # current_sklearn_version - str running version; original_sklearn_version - str
        # version used when pickling.
        # Side effects: stores the three values as same-named attributes.
        # KNOWN ISSUE: does not call the base __init__, so the warning's args tuple is
        # empty; message text is only available through __str__.
        self.estimator_name = estimator_name
        self.current_sklearn_version = current_sklearn_version
        self.original_sklearn_version = original_sklearn_version

    def __str__(self):
        # [FUNCTION DOC]
        # Outputs: str message naming the estimator, both versions, a risk note and a
        # link to the model persistence security docs.
        return (
            f"Trying to unpickle estimator {self.estimator_name} from version"
            f" {self.original_sklearn_version} when "
            f"using version {self.current_sklearn_version}. This might lead to breaking"
            " code or "
            "invalid results. Use at your own risk. "
            "For more info please refer to:\n"
            "https://scikit-learn.org/stable/model_persistence.html"
            "#security-maintainability-limitations"
        )


class EstimatorCheckFailedWarning(UserWarning):
    # [CLASS DOC]
    # UserWarning emitted by the common estimator checks (when configured to warn
    # instead of raise) for a failed check; carries the details as attributes.
    """Warning raised when an estimator check from the common tests fails.

    Parameters
    ----------
    estimator : estimator object
        Estimator instance for which the test failed.

    check_name : str
        Name of the check that failed.

    exception : Exception
        Exception raised by the failed check.

    status : str
        Status of the check.

    expected_to_fail : bool
        Whether the check was expected to fail.

    expected_to_fail_reason : str
        Reason for the expected failure.
    """

    def __init__(
        self,
        *,
        estimator,
        check_name: str,
        exception: Exception,
        status: str,
        expected_to_fail: bool,
        expected_to_fail_reason: str,
    ):
        # [FUNCTION DOC]
        # Inputs (keyword-only): estimator - estimator instance under test; check_name -
        # str; exception - the Exception raised by the check; status - str check status;
        # expected_to_fail - bool; expected_to_fail_reason - str.
        # Side effects: stores all six as same-named attributes.
        # KNOWN ISSUE: base __init__ not called, so args is empty.
        self.estimator = estimator
        self.check_name = check_name
        self.exception = exception
        self.status = status
        self.expected_to_fail = expected_to_fail
        self.expected_to_fail_reason = expected_to_fail_reason

    def __repr__(self):
        # [FUNCTION DOC]
        # Outputs: three-line str: check name and estimator repr, expected-failure info,
        # and the exception text.
        # KNOWN ISSUE: the second line is prefixed "Expected to fail reason:" and then
        # the value is either "Expected to fail: <reason>" or "Not expected to fail",
        # producing a redundant / misleading label.
        expected_to_fail_str = (
            f"Expected to fail: {self.expected_to_fail_reason}"
            if self.expected_to_fail
            else "Not expected to fail"
        )
        return (
            f"Test {self.check_name} failed for estimator {self.estimator!r}.\n"
            f"Expected to fail reason: {expected_to_fail_str}\n"
            f"Exception: {self.exception}"
        )

    def __str__(self):
        # [FUNCTION DOC]
        # Outputs: same string as __repr__.
        return self.__repr__()
