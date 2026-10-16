# [MODULE DOC]
# Meta-estimators that reduce multiclass (and, for OvR, multilabel) classification to
# a set of binary problems: OneVsRestClassifier (one estimator per class),
# OneVsOneClassifier (one estimator per class pair, voting) and OutputCodeClassifier
# (random error-correcting code book, nearest-code decoding).
# Module-level private helpers fit/predict single binary sub-estimators and handle the
# degenerate single-label case through _ConstantPredictor.
# Public names: those listed in __all__. No import-time side effects.
"""Multiclass learning algorithms.

- one-vs-the-rest / one-vs-all
- one-vs-one
- error correcting output codes

The estimators provided in this module are meta-estimators: they require a base
estimator to be provided in their constructor. For example, it is possible to
use these estimators to turn a binary classifier or a regressor into a
multiclass classifier. It is also possible to use these estimators with
multiclass estimators in the hope that their accuracy or runtime performance
improves.

All classifiers in scikit-learn implement multiclass classification; you
only need to use this module if you want to experiment with custom multiclass
strategies.

The one-vs-the-rest meta-classifier also implements a `predict_proba` method,
so long as such a method is implemented by the base classifier. This method
returns probabilities of class membership in both the single label and
multilabel case.  Note that in the multilabel case, probabilities are the
marginal probability that a given sample falls in the given class. As such, in
the multilabel case the sum of these probabilities over all possible labels
for a given sample *will not* sum to unity, as they do in the single label
case.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import array
import itertools
import warnings
from numbers import Integral, Real

import numpy as np
import scipy.sparse as sp

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    MultiOutputMixin,
    _fit_context,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import HasMethods, Interval
from sklearn.utils._tags import get_tags
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    process_routing,
)
from sklearn.utils.metaestimators import _safe_split, available_if
from sklearn.utils.multiclass import (
    _check_partial_fit_first_call,
    _ovr_decision_function,
    check_classification_targets,
)
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_method_params,
    _num_samples,
    check_is_fitted,
    validate_data,
)

__all__ = [
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
]


def _fit_binary(estimator, X, y, fit_params, classes=None):
    # [FUNCTION DOC]
    # HELPER: Fits one binary sub-problem. If y contains a single unique value, a
    # _ConstantPredictor is fitted instead of cloning the estimator.
    # Inputs: estimator - unfitted template estimator; X - training data; y - 1-D binary
    # (or code) targets; fit_params - dict forwarded to estimator.fit; classes - optional
    # pair of display labels used only for the warning message.
    # Outputs: a fitted clone of estimator, or a fitted _ConstantPredictor.
    # Side effects: emits a UserWarning "Label ... is present in all training examples"
    # when y is constant and classes is given; the label shown is classes[0] if y[0] is
    # -1 else classes[y[0]].
    """Fit a single binary estimator."""
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not None:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn(
                "Label %s is present in all training examples." % str(classes[c])
            )
        estimator = _ConstantPredictor().fit(X, unique_y)
    else:
        estimator = clone(estimator)
        estimator.fit(X, y, **fit_params)
    return estimator


def _partial_fit_binary(estimator, X, y, partial_fit_params):
    # [FUNCTION DOC]
    # HELPER: Calls estimator.partial_fit on X, y with classes fixed to [0, 1] and the
    # given extra params.
    # Inputs: estimator (already cloned/fitted, mutated in place); X; y - binary 0/1 targets;
    # partial_fit_params - dict.
    # Outputs: the same estimator object.
    # Side effects: updates the estimator's state.
    """Partially fit a single binary estimator."""
    estimator.partial_fit(X, y, classes=np.array((0, 1)), **partial_fit_params)
    return estimator


def _predict_binary(estimator, X):
    # [FUNCTION DOC]
    # HELPER: Computes a continuous score for the positive class of one binary estimator.
    # Regressors: return predict(X). Otherwise try decision_function (raveled to 1-D) and
    # fall back to the second column of predict_proba if that raises AttributeError or
    # NotImplementedError.
    # Inputs: estimator - fitted binary estimator; X - samples.
    # Outputs: 1-D float array of length n_samples.
    """Make predictions using a single binary estimator."""
    if is_regressor(estimator):
        return estimator.predict(X)
    try:
        score = np.ravel(estimator.decision_function(X))
    except (AttributeError, NotImplementedError):
        # probabilities of the positive class
        score = estimator.predict_proba(X)[:, 1]
    return score


def _threshold_for_binary_predict(estimator):
    # [FUNCTION DOC]
    # HELPER: Returns the decision threshold matching the score from _predict_binary:
    # 0.0 for classifiers exposing decision_function, otherwise 0.5 (probability).
    # Inputs: estimator - fitted sub-estimator. Outputs: float.
    # KNOWN ISSUE: for a regressor that has decision_function this returns 0.5 while
    # _predict_binary uses predict; for a _ConstantPredictor (has decision_function, not a
    # classifier) it returns 0.5, which is consistent with constant 0/1 outputs.
    """Threshold for predictions from binary estimator."""
    if hasattr(estimator, "decision_function") and is_classifier(estimator):
        return 0.0
    else:
        # predict_proba threshold
        return 0.5


class _ConstantPredictor(BaseEstimator):
    # [CLASS DOC]
    # HELPER: Degenerate estimator used when a binary sub-problem has only one label.
    # It memorizes that label and returns it for every sample from predict and
    # decision_function, and the matching one-hot probabilities from predict_proba.
    # Fitted attribute: y_ (array of the unique label(s) passed to fit).
    """Helper predictor to be used when only one class is present."""

    def fit(self, X, y):
        # [FUNCTION DOC]
        # Validates X and y leniently (no finiteness check, any dtype, sparse allowed, 1-D
        # allowed), sets n_features_in_ etc. via validate_data, and stores y as y_.
        # Inputs: X - samples; y - array containing the single unique label.
        # Outputs: self. Side effects: sets y_ and validation attributes.
        check_params = dict(
            ensure_all_finite=False, dtype=None, ensure_2d=False, accept_sparse=True
        )
        validate_data(
            self, X, y, reset=True, validate_separately=(check_params, check_params)
        )
        self.y_ = y
        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # Checks fitted state, validates X without resetting, then repeats y_ once per sample.
        # Outputs: 1-D array of length n_samples (if y_ has length 1).
        # Exceptions: NotFittedError if not fitted; validation errors on feature mismatch.
        check_is_fitted(self)
        validate_data(
            self,
            X,
            ensure_all_finite=False,
            dtype=None,
            accept_sparse=True,
            ensure_2d=False,
            reset=False,
        )

        return np.repeat(self.y_, _num_samples(X))

    def decision_function(self, X):
        # [FUNCTION DOC]
        # Same behavior as predict: returns y_ repeated n_samples times, so the constant label
        # (0 or 1) serves as the decision score.
        check_is_fitted(self)
        validate_data(
            self,
            X,
            ensure_all_finite=False,
            dtype=None,
            accept_sparse=True,
            ensure_2d=False,
            reset=False,
        )

        return np.repeat(self.y_, _num_samples(X))

    def predict_proba(self, X):
        # [FUNCTION DOC]
        # Returns an (n_samples, 2) float array whose rows are [1 - y_, y_], i.e. probability 1
        # for the memorized label when it is 0 or 1.
        # Exceptions: NotFittedError if not fitted.
        # KNOWN ISSUE: assumes y_ is a single 0/1 value; other labels give meaningless
        # "probabilities" (e.g. -1 from OutputCodeClassifier code books).
        check_is_fitted(self)
        validate_data(
            self,
            X,
            ensure_all_finite=False,
            dtype=None,
            accept_sparse=True,
            ensure_2d=False,
            reset=False,
        )
        y_ = self.y_.astype(np.float64)
        return np.repeat([np.hstack([1 - y_, y_])], _num_samples(X), axis=0)


def _estimators_has(attr):
    # [FUNCTION DOC]
    # HELPER: Builds a predicate for available_if that checks that the fitted first
    # sub-estimator (if estimators_ exists) or else the unfitted estimator has attr.
    # Inputs: attr - method name. Outputs: callable taking self.
    """Check if self.estimator or self.estimators_[0] has attr.

    If `self.estimators_[0]` has the attr, then its safe to assume that other
    estimators have it too. We raise the original `AttributeError` if `attr`
    does not exist. This function is used together with `available_if`.
    """

    def check(self):
        # [FUNCTION DOC]
        # HELPER: Accesses attr on estimators_[0] when fitted, otherwise on self.estimator,
        # letting AttributeError propagate; returns True if access succeeds.
        if hasattr(self, "estimators_"):
            getattr(self.estimators_[0], attr)
        else:
            getattr(self.estimator, attr)

        return True

    return check


class OneVsRestClassifier(
    MultiOutputMixin,
    ClassifierMixin,
    MetaEstimatorMixin,
    BaseEstimator,
):
    # [CLASS DOC]
    # One-vs-rest meta-classifier. A sparse LabelBinarizer turns y (labels or indicator
    # matrix) into n_classes binary columns, and one clone of estimator is fitted per
    # column (in parallel via joblib). Multiclass prediction takes the argmax of the
    # per-class scores; multilabel prediction thresholds each score independently.
    # Fitted attributes: label_binarizer_, classes_, estimators_, and optionally
    # n_features_in_ / feature_names_in_ copied from the first sub-estimator.
    """One-vs-the-rest (OvR) multiclass strategy.

    Also known as one-vs-all, this strategy consists in fitting one classifier
    per class. For each classifier, the class is fitted against all the other
    classes. In addition to its computational efficiency (only `n_classes`
    classifiers are needed), one advantage of this approach is its
    interpretability. Since each class is represented by one and one classifier
    only, it is possible to gain knowledge about the class by inspecting its
    corresponding classifier. This is the most commonly used strategy for
    multiclass classification and is a fair default choice.

    OneVsRestClassifier can also be used for multilabel classification. To use
    this feature, provide an indicator matrix for the target `y` when calling
    `.fit`. In other words, the target labels should be formatted as a 2D
    binary (0/1) matrix, where [i, j] == 1 indicates the presence of label j
    in sample i. This estimator uses the binary relevance method to perform
    multilabel classification, which involves training one binary classifier
    independently for each label.

    Read more in the :ref:`User Guide <ovr_classification>`.

    Parameters
    ----------
    estimator : estimator object
        A regressor or a classifier that implements :term:`fit`.
        When a classifier is passed, :term:`decision_function` will be used
        in priority and it will fallback to :term:`predict_proba` if it is not
        available.
        When a regressor is passed, :term:`predict` is used.

    n_jobs : int, default=None
        The number of jobs to use for the computation: the `n_classes`
        one-vs-rest problems are computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: 0.20
           `n_jobs` default changed from 1 to None

    verbose : int, default=0
        The verbosity level, if non zero, progress messages are printed.
        Below 50, the output is sent to stderr. Otherwise, the output is sent
        to stdout. The frequency of the messages increases with the verbosity
        level, reporting all iterations at 10. See :class:`joblib.Parallel` for
        more details.

        .. versionadded:: 1.1

    Attributes
    ----------
    estimators_ : list of `n_classes` estimators
        Estimators used for predictions.

    classes_ : array, shape = [`n_classes`]
        Class labels.

    n_classes_ : int
        Number of classes.

    label_binarizer_ : LabelBinarizer object
        Object used to transform multiclass labels to binary labels and
        vice-versa.

    multilabel_ : boolean
        Whether a OneVsRestClassifier is a multilabel classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    OneVsOneClassifier : One-vs-one multiclass strategy.
    OutputCodeClassifier : (Error-Correcting) Output-Code multiclass strategy.
    sklearn.multioutput.MultiOutputClassifier : Alternate way of extending an
        estimator for multilabel classification.
    sklearn.preprocessing.MultiLabelBinarizer : Transform iterable of iterables
        to binary indicator matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.multiclass import OneVsRestClassifier
    >>> from sklearn.svm import SVC
    >>> X = np.array([
    ...     [10, 10],
    ...     [8, 10],
    ...     [-5, 5.5],
    ...     [-5.4, 5.5],
    ...     [-20, -20],
    ...     [-15, -20]
    ... ])
    >>> y = np.array([0, 0, 1, 1, 2, 2])
    >>> clf = OneVsRestClassifier(SVC()).fit(X, y)
    >>> clf.predict([[-19, -20], [9, 9], [-5, 5]])
    array([2, 0, 1])
    """

    _parameter_constraints = {
        "estimator": [HasMethods(["fit"])],
        "n_jobs": [Integral, None],
        "verbose": ["verbose"],
    }

    def __init__(self, estimator, *, n_jobs=None, verbose=0):
        # [FUNCTION DOC]
        # Stores estimator, n_jobs and verbose unchanged (sklearn convention, no validation).
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.verbose = verbose

    @_fit_context(
        # OneVsRestClassifier.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, **fit_params):
        # [FUNCTION DOC]
        # Steps: reject unrouted metadata (_raise_for_params), route fit_params; fit a sparse
        # LabelBinarizer on y and convert to CSC; set classes_; for each column i fit
        # _fit_binary on a dense 0/1 vector, with warning labels "not <class>" / <class>; copy
        # n_features_in_ and feature_names_in_ from estimators_[0] when present.
        # Inputs: X; y - 1-D labels or 2-D indicator; fit_params - routed metadata.
        # Outputs: self.
        # Exceptions: ValueError from _raise_for_params when metadata is passed without
        # routing enabled; LabelBinarizer errors for unsupported target types.
        # Side effects: sets fitted attributes; parallel work with n_jobs/verbose.
        """Fit underlying estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        y : {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_classes)
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        **fit_params : dict
            Parameters passed to the ``estimator.fit`` method of each
            sub-estimator.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        _raise_for_params(fit_params, self, "fit")

        routed_params = process_routing(
            self,
            "fit",
            **fit_params,
        )
        # A sparse LabelBinarizer, with sparse_output=True, has been shown to
        # outperform or match a dense label binarizer in all cases and has also
        # resulted in less or equal memory consumption in the fit_ovr function
        # overall.
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.
        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_binary)(
                self.estimator,
                X,
                column,
                fit_params=routed_params.estimator.fit,
                classes=[
                    "not %s" % self.label_binarizer_.classes_[i],
                    self.label_binarizer_.classes_[i],
                ],
            )
            for i, column in enumerate(columns)
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    @available_if(_estimators_has("partial_fit"))
    @_fit_context(
        # OneVsRestClassifier.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def partial_fit(self, X, y, classes=None, **partial_fit_params):
        # [FUNCTION DOC]
        # Incremental fit. On the first call (classes given) creates n_classes clones and a
        # sparse LabelBinarizer fitted on classes_. Rejects batches containing labels outside
        # classes_, binarizes y, then calls _partial_fit_binary on each sub-estimator with its
        # column. Copies n_features_in_ from the first sub-estimator.
        # Inputs: X; y; classes - required on the first call; partial_fit_params - routed.
        # Outputs: self.
        # Exceptions: ValueError if y contains unseen labels, or if classes is missing on the
        # first call / inconsistent later (from _check_partial_fit_first_call).
        # Side effects: sets classes_ (via helper), estimators_, label_binarizer_.
        # KNOWN ISSUE: unlike fit, verbose is not forwarded to Parallel and
        # feature_names_in_ is never set.
        """Partially fit underlying estimators.

        Should be used when memory is inefficient to train all data.
        Chunks of data can be passed in several iterations.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        y : {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_classes)
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        classes : array, shape (n_classes, )
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is only required in the first call of partial_fit
            and can be omitted in the subsequent calls.

        **partial_fit_params : dict
            Parameters passed to the ``estimator.partial_fit`` method of each
            sub-estimator.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        self : object
            Instance of partially fitted estimator.
        """
        _raise_for_params(partial_fit_params, self, "partial_fit")

        routed_params = process_routing(
            self,
            "partial_fit",
            **partial_fit_params,
        )

        if _check_partial_fit_first_call(self, classes):
            self.estimators_ = [clone(self.estimator) for _ in range(self.n_classes_)]

            # A sparse LabelBinarizer, with sparse_output=True, has been
            # shown to outperform or match a dense label binarizer in all
            # cases and has also resulted in less or equal memory consumption
            # in the fit_ovr function overall.
            self.label_binarizer_ = LabelBinarizer(sparse_output=True)
            self.label_binarizer_.fit(self.classes_)

        if len(np.setdiff1d(y, self.classes_)):
            raise ValueError(
                (
                    "Mini-batch contains {0} while classes " + "must be subset of {1}"
                ).format(np.unique(y), self.classes_)
            )

        Y = self.label_binarizer_.transform(y)
        Y = Y.tocsc()
        columns = (col.toarray().ravel() for col in Y.T)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit_binary)(
                estimator,
                X,
                column,
                partial_fit_params=routed_params.estimator.partial_fit,
            )
            for estimator, column in zip(self.estimators_, columns)
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_

        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # Multiclass target type: iterate the estimators in reverse, keep a running elementwise
        # maximum of scores and record the index wherever the score equals the current max, so
        # ties resolve to the lowest class index (like argmax); return classes_ at those
        # indices. Multilabel: threshold each estimator's score (0 or 0.5 depending on the
        # first estimator), build a sparse CSC indicator (n_samples, n_classes) and
        # inverse-transform it with the label binarizer.
        # Inputs: X. Outputs: 1-D label array or indicator matrix (sparse/dense as the
        # binarizer returns).
        # Exceptions: NotFittedError.
        """Predict multi-class targets using underlying estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        Returns
        -------
        y : {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_classes)
            Predicted multi-class targets.
        """
        check_is_fitted(self)

        n_samples = _num_samples(X)
        if self.label_binarizer_.y_type_ == "multiclass":
            maxima = np.empty(n_samples, dtype=float)
            maxima.fill(-np.inf)
            argmaxima = np.zeros(n_samples, dtype=int)
            n_classes = len(self.estimators_)
            # Iterate in reverse order to match np.argmax tie-breaking behavior
            for i, e in enumerate(reversed(self.estimators_)):
                pred = _predict_binary(e, X)
                np.maximum(maxima, pred, out=maxima)
                argmaxima[maxima == pred] = n_classes - i - 1
            return self.classes_[argmaxima]
        else:
            thresh = _threshold_for_binary_predict(self.estimators_[0])
            indices = array.array("i")
            indptr = array.array("i", [0])
            for e in self.estimators_:
                indices.extend(np.where(_predict_binary(e, X) > thresh)[0])
                indptr.append(len(indices))
            data = np.ones(len(indices), dtype=int)
            indicator = sp.csc_array(
                (data, indices, indptr), shape=(n_samples, len(self.estimators_))
            )
            return self.label_binarizer_.inverse_transform(indicator)

    @available_if(_estimators_has("predict_proba"))
    def predict_proba(self, X):
        # [FUNCTION DOC]
        # Stacks each sub-estimator's positive-class probability into (n_samples, n_classes).
        # With a single estimator (binary problem) prepends the complement column. In the
        # non-multilabel case, rows with a nonzero sum are normalized to sum to 1 (in place).
        # Inputs: X. Outputs: float array (n_samples, n_classes) (2 columns for binary).
        # Exceptions: NotFittedError; only available if the sub-estimators have predict_proba.
        """Probability estimates.

        The returned estimates for all classes are ordered by label of classes.

        Note that in the multilabel case, each sample can have any number of
        labels. This returns the marginal probability that the given sample has
        the label in question. For example, it is entirely consistent that two
        labels both have a 90% probability of applying to a given sample.

        In the single label multiclass case, the rows of the returned matrix
        sum to 1.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.
        """
        check_is_fitted(self)
        # Y[i, j] gives the probability that sample i has the label j.
        # In the multi-label case, these are not disjoint.
        Y = np.array([e.predict_proba(X)[:, 1] for e in self.estimators_]).T

        if len(self.estimators_) == 1:
            # Only one estimator, but we still want to return probabilities
            # for two classes.
            Y = np.concatenate(((1 - Y), Y), axis=1)

        if not self.multilabel_:
            # Then, (nonzero) sample probability distributions should be normalized.
            row_sums = np.sum(Y, axis=1)[:, np.newaxis]
            np.divide(Y, row_sums, out=Y, where=row_sums != 0)

        return Y

    @available_if(_estimators_has("decision_function"))
    def decision_function(self, X):
        # [FUNCTION DOC]
        # Returns estimators_[0].decision_function(X) unchanged for a single estimator
        # (binary), otherwise stacks each estimator's raveled decision values column-wise.
        # Outputs: (n_samples,) or (n_samples, n_classes).
        # Exceptions: NotFittedError; available only if sub-estimators have decision_function.
        """Decision function for the OneVsRestClassifier.

        Return the distance of each sample from the decision boundary for each
        class. This can only be used with estimators which implement the
        `decision_function` method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes) or (n_samples,) for \
            binary classification.
            Result of calling `decision_function` on the final estimator.

            .. versionchanged:: 0.19
                output shape changed to ``(n_samples,)`` to conform to
                scikit-learn conventions for binary classification.
        """
        check_is_fitted(self)
        if len(self.estimators_) == 1:
            return self.estimators_[0].decision_function(X)
        return np.array(
            [est.decision_function(X).ravel() for est in self.estimators_]
        ).T

    @property
    def multilabel_(self):
        # [FUNCTION DOC]
        # Property: True when the label binarizer saw a "multilabel-*" target type.
        # Exceptions: AttributeError if not fitted.
        """Whether this is a multilabel classifier."""
        return self.label_binarizer_.y_type_.startswith("multilabel")

    @property
    def n_classes_(self):
        # [FUNCTION DOC]
        # Property: number of entries in classes_.
        """Number of classes."""
        return len(self.classes_)

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Extends parent tags by copying the wrapped estimator's pairwise and sparse input tags.
        # Outputs: Tags object.
        """Indicate if wrapped estimator is using a precomputed Gram matrix"""
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = get_tags(self.estimator).input_tags.pairwise
        tags.input_tags.sparse = get_tags(self.estimator).input_tags.sparse
        return tags

    def get_metadata_routing(self):
        # [FUNCTION DOC]
        # Builds a MetadataRouter that includes self requests and routes fit->fit and
        # partial_fit->partial_fit to the wrapped estimator.
        # Outputs: MetadataRouter.
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """

        router = (
            MetadataRouter(owner=self)
            .add_self_request(self)
            .add(
                estimator=self.estimator,
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="partial_fit", callee="partial_fit"),
            )
        )
        return router


def _fit_ovo_binary(estimator, X, y, i, j, fit_params):
    # [FUNCTION DOC]
    # HELPER: Fits the classifier for one class pair (i, j). Selects samples whose label is
    # i or j, encodes i as 0 and j as 1, subsets X with _safe_split (which also handles
    # pairwise/precomputed kernels by selecting rows and columns) and subsets the
    # sample-aligned fit params, then delegates to _fit_binary.
    # Inputs: estimator; X; y - full label array; i, j - the two class labels; fit_params.
    # Outputs: tuple (fitted estimator, integer indices of the selected samples).
    """Fit a single binary estimator (one-vs-one)."""
    cond = np.logical_or(y == i, y == j)
    y = y[cond]
    y_binary = np.empty(y.shape, int)
    y_binary[y == i] = 0
    y_binary[y == j] = 1
    indcond = np.arange(_num_samples(X))[cond]

    fit_params_subset = _check_method_params(X, params=fit_params, indices=indcond)
    return (
        _fit_binary(
            estimator,
            _safe_split(estimator, X, None, indices=indcond)[0],
            y_binary,
            fit_params=fit_params_subset,
            classes=[i, j],
        ),
        indcond,
    )


def _partial_fit_ovo_binary(estimator, X, y, i, j, partial_fit_params):
    # [FUNCTION DOC]
    # HELPER: Incremental counterpart of _fit_ovo_binary. Selects the samples labelled i or
    # j in this batch; if there are any, encodes j as 1 and others as 0, subsets params by
    # the boolean mask and calls _partial_fit_binary; otherwise returns the estimator
    # untouched.
    # Inputs: estimator - current sub-estimator; X; y; i, j; partial_fit_params.
    # Outputs: the (possibly updated) estimator.
    # KNOWN ISSUE: subsets X with plain boolean indexing rather than _safe_split, so
    # pairwise/precomputed inputs are not column-subset as in fit (partial_fit sets
    # pairwise_indices_ to None accordingly).
    """Partially fit a single binary estimator(one-vs-one)."""

    cond = np.logical_or(y == i, y == j)
    y = y[cond]
    if len(y) != 0:
        y_binary = np.zeros_like(y)
        y_binary[y == j] = 1
        partial_fit_params_subset = _check_method_params(
            X, params=partial_fit_params, indices=cond
        )
        return _partial_fit_binary(
            estimator, X[cond], y_binary, partial_fit_params=partial_fit_params_subset
        )
    return estimator


class OneVsOneClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    # [CLASS DOC]
    # One-vs-one meta-classifier: fits n_classes*(n_classes-1)/2 binary estimators, one per
    # ordered pair (i < j) of classes, each on the samples of those two classes only.
    # Decision values are votes plus normalized confidences (_ovr_decision_function).
    # Fitted attributes: classes_, estimators_, pairwise_indices_, n_features_in_,
    # feature_names_in_ (from validate_data).
    """One-vs-one multiclass strategy.

    This strategy consists in fitting one classifier per class pair.
    At prediction time, the class which received the most votes is selected.
    Since it requires to fit `n_classes * (n_classes - 1) / 2` classifiers,
    this method is usually slower than one-vs-the-rest, due to its
    O(n_classes^2) complexity. However, this method may be advantageous for
    algorithms such as kernel algorithms which don't scale well with
    `n_samples`. This is because each individual learning problem only involves
    a small subset of the data whereas, with one-vs-the-rest, the complete
    dataset is used `n_classes` times.

    Read more in the :ref:`User Guide <ovo_classification>`.

    Parameters
    ----------
    estimator : estimator object
        A regressor or a classifier that implements :term:`fit`.
        When a classifier is passed, :term:`decision_function` will be used
        in priority and it will fallback to :term:`predict_proba` if it is not
        available.
        When a regressor is passed, :term:`predict` is used.

    n_jobs : int, default=None
        The number of jobs to use for the computation: the `n_classes * (
        n_classes - 1) / 2` OVO problems are computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    estimators_ : list of ``n_classes * (n_classes - 1) / 2`` estimators
        Estimators used for predictions.

    classes_ : numpy array of shape [n_classes]
        Array containing labels.

    n_classes_ : int
        Number of classes.

    pairwise_indices_ : list, length = ``len(estimators_)``, or ``None``
        Indices of samples used when training the estimators.
        ``None`` when ``estimator``'s `pairwise` tag is False.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    OneVsRestClassifier : One-vs-all multiclass strategy.
    OutputCodeClassifier : (Error-Correcting) Output-Code multiclass strategy.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.multiclass import OneVsOneClassifier
    >>> from sklearn.svm import LinearSVC
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.33, shuffle=True, random_state=0)
    >>> clf = OneVsOneClassifier(
    ...     LinearSVC(random_state=0)).fit(X_train, y_train)
    >>> clf.predict(X_test[:10])
    array([2, 1, 0, 2, 0, 2, 0, 1, 1, 1])
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit"])],
        "n_jobs": [Integral, None],
    }

    def __init__(self, estimator, *, n_jobs=None):
        # [FUNCTION DOC]
        # Stores estimator and n_jobs unchanged.
        self.estimator = estimator
        self.n_jobs = n_jobs

    @_fit_context(
        # OneVsOneClassifier.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, **fit_params):
        # [FUNCTION DOC]
        # Steps: reject/route metadata; validate X (csr/csc sparse allowed, non-finite allowed)
        # and y; check classification targets; set classes_ (sorted unique); in parallel fit
        # _fit_ovo_binary for every pair (i < j) in that order; split the results into
        # estimators_ (tuple) and pairwise_indices_ (kept only if this meta-estimator's
        # pairwise tag is set, else None).
        # Inputs: X; y - 1-D labels; fit_params.
        # Outputs: self.
        # Exceptions: ValueError if only one class is present or targets are not
        # classification-like; ValueError for unrouted metadata.
        # Side effects: sets fitted attributes.
        """Fit underlying estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        y : array-like of shape (n_samples,)
            Multi-class targets.

        **fit_params : dict
            Parameters passed to the ``estimator.fit`` method of each
            sub-estimator.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        self : object
            The fitted underlying estimator.
        """
        _raise_for_params(fit_params, self, "fit")

        routed_params = process_routing(
            self,
            "fit",
            **fit_params,
        )

        # We need to validate the data because we do a safe_indexing later.
        X, y = validate_data(
            self, X, y, accept_sparse=["csr", "csc"], ensure_all_finite=False
        )
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError(
                "OneVsOneClassifier can not be fit when only one class is present."
            )
        n_classes = self.classes_.shape[0]
        estimators_indices = list(
            zip(
                *(
                    Parallel(n_jobs=self.n_jobs)(
                        delayed(_fit_ovo_binary)(
                            self.estimator,
                            X,
                            y,
                            self.classes_[i],
                            self.classes_[j],
                            fit_params=routed_params.estimator.fit,
                        )
                        for i in range(n_classes)
                        for j in range(i + 1, n_classes)
                    )
                )
            )
        )

        self.estimators_ = estimators_indices[0]

        pairwise = self.__sklearn_tags__().input_tags.pairwise
        self.pairwise_indices_ = estimators_indices[1] if pairwise else None

        return self

    @available_if(_estimators_has("partial_fit"))
    @_fit_context(
        # OneVsOneClassifier.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def partial_fit(self, X, y, classes=None, **partial_fit_params):
        # [FUNCTION DOC]
        # Incremental fit. On the first call creates one clone per class pair. Rejects batches
        # with labels not in classes_, validates X/y (resetting feature info only on the first
        # call), then for each pair from itertools combinations calls _partial_fit_ovo_binary
        # in parallel. Sets pairwise_indices_ to None and copies n_features_in_ from the first
        # estimator.
        # Inputs: X; y; classes - required on first call; partial_fit_params.
        # Outputs: self.
        # Exceptions: ValueError for unseen labels, missing classes on the first call, or
        # unrouted metadata.
        # KNOWN ISSUE: the unseen-label check runs on raw y before validate_data.
        """Partially fit underlying estimators.

        Should be used when memory is inefficient to train all data. Chunks
        of data can be passed in several iteration, where the first call
        should have an array of all target variables.

        Parameters
        ----------
        X : {array-like, sparse matrix) of shape (n_samples, n_features)
            Data.

        y : array-like of shape (n_samples,)
            Multi-class targets.

        classes : array, shape (n_classes, )
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is only required in the first call of partial_fit
            and can be omitted in the subsequent calls.

        **partial_fit_params : dict
            Parameters passed to the ``estimator.partial_fit`` method of each
            sub-estimator.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        self : object
            The partially fitted underlying estimator.
        """
        _raise_for_params(partial_fit_params, self, "partial_fit")

        routed_params = process_routing(
            self,
            "partial_fit",
            **partial_fit_params,
        )

        first_call = _check_partial_fit_first_call(self, classes)
        if first_call:
            self.estimators_ = [
                clone(self.estimator)
                for _ in range(self.n_classes_ * (self.n_classes_ - 1) // 2)
            ]

        if len(np.setdiff1d(y, self.classes_)):
            raise ValueError(
                "Mini-batch contains {0} while it must be subset of {1}".format(
                    np.unique(y), self.classes_
                )
            )

        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=["csr", "csc"],
            ensure_all_finite=False,
            reset=first_call,
        )
        check_classification_targets(y)
        combinations = itertools.combinations(range(self.n_classes_), 2)
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit_ovo_binary)(
                estimator,
                X,
                y,
                self.classes_[i],
                self.classes_[j],
                partial_fit_params=routed_params.estimator.partial_fit,
            )
            for estimator, (i, j) in zip(self.estimators_, (combinations))
        )

        self.pairwise_indices_ = None

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_

        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # Computes decision_function. For two classes, thresholds the 1-D result with
        # _threshold_for_binary_predict of the first estimator and maps 0/1 to classes_;
        # otherwise returns classes_ at the row-wise argmax.
        # Outputs: 1-D label array.
        # KNOWN ISSUE: in the binary case decision_function returns the vote-plus-confidence
        # value of class 1 (in [0, 1] style ranges, not a raw margin), yet it is compared with
        # 0.0 or 0.5 depending on the base estimator; behavior relies on the vote term
        # dominating.
        """Estimate the best class label for each sample in X.

        This is implemented as ``argmax(decision_function(X), axis=1)`` which
        will return the label of the class with most votes by estimators
        predicting the outcome of a decision for each possible class pair.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        Returns
        -------
        y : numpy array of shape [n_samples]
            Predicted multi-class targets.
        """
        Y = self.decision_function(X)
        if self.n_classes_ == 2:
            thresh = _threshold_for_binary_predict(self.estimators_[0])
            return self.classes_[(Y > thresh).astype(int)]
        return self.classes_[Y.argmax(axis=1)]

    def decision_function(self, X):
        # [FUNCTION DOC]
        # Validates X (no reset). For each estimator uses X itself, or the column subset given
        # by pairwise_indices_ when stored (precomputed kernels). Stacks predictions (0/1) and
        # confidences (_predict_binary) into (n_samples, n_estimators) arrays and combines them
        # with _ovr_decision_function into (n_samples, n_classes). Returns only column 1 for
        # binary problems.
        # Exceptions: NotFittedError; feature-count mismatch errors.
        """Decision function for the OneVsOneClassifier.

        The decision values for the samples are computed by adding the
        normalized sum of pair-wise classification confidence levels to the
        votes in order to disambiguate between the decision values when the
        votes for all the classes are equal leading to a tie.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        Y : array-like of shape (n_samples, n_classes) or (n_samples,)
            Result of calling `decision_function` on the final estimator.

            .. versionchanged:: 0.19
                output shape changed to ``(n_samples,)`` to conform to
                scikit-learn conventions for binary classification.
        """
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            accept_sparse=True,
            ensure_all_finite=False,
            reset=False,
        )

        indices = self.pairwise_indices_
        if indices is None:
            Xs = [X] * len(self.estimators_)
        else:
            Xs = [X[:, idx] for idx in indices]

        predictions = np.vstack(
            [est.predict(Xi) for est, Xi in zip(self.estimators_, Xs)]
        ).T
        confidences = np.vstack(
            [_predict_binary(est, Xi) for est, Xi in zip(self.estimators_, Xs)]
        ).T
        Y = _ovr_decision_function(predictions, confidences, len(self.classes_))
        if self.n_classes_ == 2:
            return Y[:, 1]
        return Y

    @property
    def n_classes_(self):
        # [FUNCTION DOC]
        # Property: number of entries in classes_.
        """Number of classes."""
        return len(self.classes_)

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Copies the wrapped estimator's pairwise and sparse input tags onto the parent tags.
        """Indicate if wrapped estimator is using a precomputed Gram matrix"""
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = get_tags(self.estimator).input_tags.pairwise
        tags.input_tags.sparse = get_tags(self.estimator).input_tags.sparse
        return tags

    def get_metadata_routing(self):
        # [FUNCTION DOC]
        # MetadataRouter with self requests, routing fit->fit and partial_fit->partial_fit to
        # the wrapped estimator.
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """

        router = (
            MetadataRouter(owner=self)
            .add_self_request(self)
            .add(
                estimator=self.estimator,
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="partial_fit", callee="partial_fit"),
            )
        )
        return router


class OutputCodeClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    # [CLASS DOC]
    # Error-correcting output-code meta-classifier. A random code book of shape
    # (n_classes, int(n_classes * code_size)) is drawn; one binary estimator is fitted per
    # column; prediction picks the class whose code row is closest (Euclidean) to the
    # vector of sub-estimator scores.
    # Fitted attributes: classes_, code_book_, estimators_, optional n_features_in_ /
    # feature_names_in_ from the first sub-estimator.
    """(Error-Correcting) Output-Code multiclass strategy.

    Output-code based strategies consist in representing each class with a
    binary code (an array of 0s and 1s). At fitting time, one binary
    classifier per bit in the code book is fitted.  At prediction time, the
    classifiers are used to project new points in the class space and the class
    closest to the points is chosen. The main advantage of these strategies is
    that the number of classifiers used can be controlled by the user, either
    for compressing the model (0 < `code_size` < 1) or for making the model more
    robust to errors (`code_size` > 1). See the documentation for more details.

    Read more in the :ref:`User Guide <ecoc>`.

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing :term:`fit` and one of
        :term:`decision_function` or :term:`predict_proba`.

    code_size : float, default=1.5
        Percentage of the number of classes to be used to create the code book.
        A number between 0 and 1 will require fewer classifiers than
        one-vs-the-rest. A number greater than 1 will require more classifiers
        than one-vs-the-rest.

    random_state : int, RandomState instance, default=None
        The generator used to initialize the codebook.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_jobs : int, default=None
        The number of jobs to use for the computation: the multiclass problems
        are computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    estimators_ : list of `int(n_classes * code_size)` estimators
        Estimators used for predictions.

    classes_ : ndarray of shape (n_classes,)
        Array containing labels.

    code_book_ : ndarray of shape (n_classes, `len(estimators_)`)
        Binary array containing the code of each class.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    OneVsRestClassifier : One-vs-all multiclass strategy.
    OneVsOneClassifier : One-vs-one multiclass strategy.

    References
    ----------

    .. [1] "Solving multiclass learning problems via error-correcting output
       codes",
       Dietterich T., Bakiri G.,
       Journal of Artificial Intelligence Research 2,
       1995.

    .. [2] "The error coding method and PICTs",
       James G., Hastie T.,
       Journal of Computational and Graphical statistics 7,
       1998.

    .. [3] "The Elements of Statistical Learning",
       Hastie T., Tibshirani R., Friedman J., page 606 (second-edition)
       2008.

    Examples
    --------
    >>> from sklearn.multiclass import OutputCodeClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = OutputCodeClassifier(
    ...     estimator=RandomForestClassifier(random_state=0),
    ...     random_state=0).fit(X, y)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])
    """

    _parameter_constraints: dict = {
        "estimator": [
            HasMethods(["fit", "decision_function"]),
            HasMethods(["fit", "predict_proba"]),
        ],
        "code_size": [Interval(Real, 0.0, None, closed="neither")],
        "random_state": ["random_state"],
        "n_jobs": [Integral, None],
    }

    def __init__(self, estimator, *, code_size=1.5, random_state=None, n_jobs=None):
        # [FUNCTION DOC]
        # Stores estimator, code_size, random_state and n_jobs unchanged.
        self.estimator = estimator
        self.code_size = code_size
        self.random_state = random_state
        self.n_jobs = n_jobs

    @_fit_context(
        # OutputCodeClassifier.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y, **fit_params):
        # [FUNCTION DOC]
        # Steps: reject/route metadata; validate y only (X is not validated here); create the
        # RNG from random_state; check targets; set classes_; draw a uniform
        # (n_classes, n_estimators) matrix, set entries > 0.5 to 1 and the rest to -1 if the
        # estimator has decision_function else 0; build the per-sample integer code matrix Y by
        # looking up each sample's class row; fit _fit_binary on each column in parallel; copy
        # n_features_in_ / feature_names_in_ from the first estimator.
        # Inputs: X; y - 1-D labels; fit_params.
        # Outputs: self.
        # Exceptions: ValueError for non-classification targets or unrouted metadata.
        # Side effects: consumes randomness; sets fitted attributes.
        # KNOWN ISSUE: the "no class is present" check (n_classes == 0) is effectively
        # unreachable since y validation fails earlier on empty input. Also if
        # int(n_classes * code_size) is 0 there are no estimators and estimators_[0] raises
        # IndexError. A random code column can be constant, which silently yields a
        # _ConstantPredictor (no warning since classes is not passed).
        """Fit underlying estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        y : array-like of shape (n_samples,)
            Multi-class targets.

        **fit_params : dict
            Parameters passed to the ``estimator.fit`` method of each
            sub-estimator.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        _raise_for_params(fit_params, self, "fit")

        routed_params = process_routing(
            self,
            "fit",
            **fit_params,
        )

        y = validate_data(self, X="no_validation", y=y)

        random_state = check_random_state(self.random_state)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        if n_classes == 0:
            raise ValueError(
                "OutputCodeClassifier can not be fit when no class is present."
            )
        n_estimators = int(n_classes * self.code_size)

        # FIXME: there are more elaborate methods than generating the codebook
        # randomly.
        self.code_book_ = random_state.uniform(size=(n_classes, n_estimators))
        self.code_book_[self.code_book_ > 0.5] = 1.0

        if hasattr(self.estimator, "decision_function"):
            self.code_book_[self.code_book_ != 1] = -1.0
        else:
            self.code_book_[self.code_book_ != 1] = 0.0

        classes_index = {c: i for i, c in enumerate(self.classes_)}

        Y = np.array(
            [self.code_book_[classes_index[y[i]]] for i in range(_num_samples(y))],
            dtype=int,
        )

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_binary)(
                self.estimator, X, Y[:, i], fit_params=routed_params.estimator.fit
            )
            for i in range(Y.shape[1])
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # Builds an F-ordered float64 matrix of _predict_binary scores (one row per estimator),
        # transposes it to C-contiguous (n_samples, n_estimators), finds for each sample the
        # nearest code-book row by Euclidean distance and returns the corresponding class.
        # Exceptions: NotFittedError.
        """Predict multi-class targets using underlying estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted multi-class targets.
        """
        check_is_fitted(self)
        # ArgKmin only accepts C-contiguous array. The aggregated predictions need to be
        # transposed. We therefore create an F-contiguous array to avoid a copy and have
        # a C-contiguous array after the transpose operation.
        Y = np.array(
            [_predict_binary(e, X) for e in self.estimators_],
            order="F",
            dtype=np.float64,
        ).T
        pred = pairwise_distances_argmin(Y, self.code_book_, metric="euclidean")
        return self.classes_[pred]

    def get_metadata_routing(self):
        # [FUNCTION DOC]
        # MetadataRouter routing fit->fit to the wrapped estimator (no self requests).
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """

        router = MetadataRouter(owner=self).add(
            estimator=self.estimator,
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Copies the wrapped estimator's sparse input tag onto the parent tags.
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = get_tags(self.estimator).input_tags.sparse
        return tags
