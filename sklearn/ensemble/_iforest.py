# [MODULE DOC]
# Isolation Forest outlier detector. IsolationForest builds a bagged ensemble of
# fully random ExtraTreeRegressor trees on subsamples and scores each sample by its
# average normalized path length to a leaf; shorter paths mean more anomalous.
# Also contains the module-private helpers for per-tree depth accumulation and the
# average path length of an unsuccessful BST search. Public API: IsolationForest.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numbers
import threading
from numbers import Integral, Real
from warnings import warn

import numpy as np
from scipy.sparse import issparse

from sklearn.base import OutlierMixin, _fit_context
from sklearn.ensemble._bagging import BaseBagging
from sklearn.tree import ExtraTreeRegressor
from sklearn.utils import check_array, check_random_state, gen_batches
from sklearn.utils._chunking import get_chunk_n_rows
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_sample_weight,
    _num_samples,
    check_is_fitted,
    validate_data,
)

__all__ = ["IsolationForest"]


def _parallel_compute_tree_depths(
    tree,
    X,
    features,
    tree_decision_path_lengths,
    tree_avg_path_lengths,
    depths,
    lock,
):
    # [FUNCTION DOC]
    # HELPER: Accumulates, for one tree, each sample's isolation depth into a shared
    # array.
    # Behavior: restricts X to the tree's feature subset (if features is not None),
    # finds each sample's leaf with tree.apply, and under the lock adds leaf depth plus
    # the leaf's average path length correction minus 1 to depths.
    # Inputs: tree - fitted ExtraTreeRegressor; X - (n_samples, n_features) array or
    # CSR; features - index array or None; tree_decision_path_lengths - depth per node;
    # tree_avg_path_lengths - average path length per node; depths - float array
    # (n_samples,) accumulated in place; lock - threading.Lock.
    # Outputs: None.
    # Side effects: mutates depths in place (thread-safe via lock).
    """Parallel computation of isolation tree depth."""
    if features is None:
        X_subset = X
    else:
        X_subset = X[:, features]

    leaves_index = tree.apply(X_subset, check_input=False)

    with lock:
        depths += (
            tree_decision_path_lengths[leaves_index]
            + tree_avg_path_lengths[leaves_index]
            - 1.0
        )


class IsolationForest(OutlierMixin, BaseBagging):
    # [CLASS DOC]
    # Unsupervised anomaly detector built on BaseBagging with ExtraTreeRegressor base
    # trees (max_features=1, random splitter) fit on random uniform pseudo-targets so
    # splits are random. Tree depth is capped at ceil(log2(max_samples)). Scores are 2
    # ** (-mean_depth / c(max_samples)) negated, and the decision threshold offset_ is
    # -0.5 ("auto") or the contamination percentile of training scores.
    # Fitted attributes (beyond BaseBagging's): max_samples_, offset_,
    # _average_path_length_per_tree, _decision_path_lengths. NaN inputs are allowed.
    """
    Isolation Forest Algorithm.

    Return the anomaly score of each sample using the IsolationForest algorithm

    The IsolationForest 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.

    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function.

    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.

    Read more in the :ref:`User Guide <isolation_forest>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators in the ensemble.

    max_samples : "auto", int or float, default="auto"
        The number of samples to draw from X to train each base estimator.

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.
        - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.

        - If 'auto', the threshold is determined as in the
          original paper.
        - If float, the contamination should be in the range (0, 0.5].

        .. versionchanged:: 0.22
           The default value of ``contamination`` changed from 0.1
           to ``'auto'``.

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.

        - If int, then draw `max_features` features.
        - If float, then draw `max(1, int(max_features * n_features_in_))` features.

        Note: using a float number less than 1.0 or integer less than number of
        features will enable feature subsampling and leads to a longer runtime.

    bootstrap : bool, default=False
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    n_jobs : int, default=None
        The number of jobs to run in parallel for :meth:`fit`. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using
        all processors. See :term:`Glossary <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : int, default=0
        Controls the verbosity of the tree building process.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.21

    Attributes
    ----------
    estimator_ : :class:`~sklearn.tree.ExtraTreeRegressor` instance
        The child estimator template used to create the collection of
        fitted sub-estimators.

        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.

    estimators_ : list of ExtraTreeRegressor instances
        The collection of fitted sub-estimators.

    estimators_features_ : list of ndarray
        The subset of drawn features for each base estimator.

    estimators_samples_ : list of ndarray
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    max_samples_ : int
        The actual number of samples.

    offset_ : float
        Offset used to define the decision function from the raw scores. We
        have the relation: ``decision_function = score_samples - offset_``.
        ``offset_`` is defined as follows. When the contamination parameter is
        set to "auto", the offset is equal to -0.5 as the scores of inliers are
        close to 0 and the scores of outliers are close to -1. When a
        contamination parameter different than "auto" is provided, the offset
        is defined in such a way we obtain the expected number of outliers
        (samples with decision function < 0) in training.

        .. versionadded:: 0.20

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.covariance.EllipticEnvelope : An object for detecting outliers in a
        Gaussian distributed dataset.
    sklearn.svm.OneClassSVM : Unsupervised Outlier Detection.
        Estimate the support of a high-dimensional distribution.
        The implementation is based on libsvm.
    sklearn.neighbors.LocalOutlierFactor : Unsupervised Outlier Detection
        using Local Outlier Factor (LOF).

    Notes
    -----
    The implementation is based on an ensemble of ExtraTreeRegressor. The
    maximum depth of each tree is set to ``ceil(log_2(n))`` where
    :math:`n` is the number of samples used to build the tree
    (see [1]_ for more details).

    References
    ----------
    .. [1] F. T. Liu, K. M. Ting and Z. -H. Zhou.
           :doi:`"Isolation forest." <10.1109/ICDM.2008.17>`
           2008 Eighth IEEE International Conference on Data Mining (ICDM),
           2008, pp. 413-422.
    .. [2] F. T. Liu, K. M. Ting and Z. -H. Zhou.
           :doi:`"Isolation-based anomaly detection."
           <10.1145/2133360.2133363>` ACM Transactions on
           Knowledge Discovery from Data (TKDD) 6.1 (2012): 1-39.

    Examples
    --------
    >>> from sklearn.ensemble import IsolationForest
    >>> X = [[-1.1], [0.3], [0.5], [100]]
    >>> clf = IsolationForest(random_state=0).fit(X)
    >>> clf.predict([[0.1], [0], [90]])
    array([ 1,  1, -1])

    For an example of using isolation forest for anomaly detection see
    :ref:`sphx_glr_auto_examples_ensemble_plot_isolation_forest.py`.
    """

    _parameter_constraints: dict = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "max_samples": [
            StrOptions({"auto"}),
            Interval(Integral, 1, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="right"),
        ],
        "contamination": [
            StrOptions({"auto"}),
            Interval(Real, 0, 0.5, closed="right"),
        ],
        "max_features": [
            Integral,
            Interval(Real, 0, 1, closed="right"),
        ],
        "bootstrap": ["boolean"],
        "n_jobs": [Integral, None],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
        "warm_start": ["boolean"],
    }

    def __init__(
        self,
        *,
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
    ):
        # [FUNCTION DOC]
        # Stores hyperparameters via BaseBagging.__init__ (estimator=None,
        # bootstrap_features=False, the rest forwarded) and stores contamination. No
        # validation here (done by _fit_context in fit).
        super().__init__(
            estimator=None,
            # here above max_features has no links with self.max_features
            bootstrap=bootstrap,
            bootstrap_features=False,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

        self.contamination = contamination

    def _get_estimator(self):
        # [FUNCTION DOC]
        # HELPER: Returns the base estimator template: ExtraTreeRegressor with
        # max_features=1, splitter="random" and the forest's random_state.
        return ExtraTreeRegressor(
            # here max_features has no links with self.max_features
            max_features=1,
            splitter="random",
            random_state=self.random_state,
        )

    def _set_oob_score(self, X, y):
        # [FUNCTION DOC]
        # HELPER: Overrides BaseBagging hook; always raises NotImplementedError since
        # OOB scoring is not supported.
        raise NotImplementedError("OOB score not supported by iforest")

    def _parallel_args(self):
        # [FUNCTION DOC]
        # HELPER: Returns joblib Parallel kwargs {"prefer": "threads"} used during fit,
        # since tree fitting releases the GIL.
        # ExtraTreeRegressor releases the GIL, so it's more efficient to use
        # a thread-based backend rather than a process-based backend so as
        # to avoid suffering from communication overhead and extra memory
        # copies. This is only used in the fit method.
        return {"prefer": "threads"}

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        # [FUNCTION DOC]
        # Fits the forest.
        # Behavior: validates X (float32, CSC sparse allowed, NaN/inf allowed),
        # validates sample_weight, sorts sparse indices, draws a uniform random
        # pseudo-target y of length n_samples from random_state, resolves max_samples
        # ("auto" -> min(256, n), int larger than n -> n with a warning, float ->
        # int(fraction * n)), sets max_depth to ceil(log2(max(max_samples, 2))) and
        # calls BaseBagging._fit. Then precomputes for each tree the average path length
        # per node (from n_node_samples) and node depths. Sets offset_ to -0.5 for
        # contamination "auto", else to the contamination percentile of the training
        # score_samples (X converted to CSR if sparse).
        # Inputs: X - (n_samples, n_features) array-like or sparse; y - ignored;
        # sample_weight - (n_samples,) or None.
        # Outputs: self.
        # Exceptions: parameter validation errors (InvalidParameterError) via
        # _fit_context; ValueError from validate_data/_check_sample_weight.
        # Side effects: sets estimators_, estimators_features_, max_samples_, offset_
        # and the private per-tree arrays; emits UserWarning when int max_samples >
        # n_samples; consumes randomness from random_state.
        # KNOWN ISSUE: a float max_samples small enough that fraction * n_samples < 1
        # yields max_samples 0, contradicting the comment that it lies in [1,
        # n_samples].
        """
        Fit estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = validate_data(
            self, X, accept_sparse=["csc"], dtype=np.float32, ensure_all_finite=False
        )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=None)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        rnd = check_random_state(self.random_state)
        y = rnd.uniform(size=X.shape[0])

        # ensure that max_sample is in [1, n_samples]:
        n_samples = X.shape[0]

        if isinstance(self.max_samples, str) and self.max_samples == "auto":
            max_samples = min(256, n_samples)

        elif isinstance(self.max_samples, numbers.Integral):
            if self.max_samples > n_samples:
                warn(
                    "max_samples (%s) is greater than the "
                    "total number of samples (%s). max_samples "
                    "will be set to n_samples for estimation."
                    % (self.max_samples, n_samples)
                )
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:  # max_samples is float
            max_samples = int(self.max_samples * X.shape[0])

        self.max_samples_ = max_samples
        max_depth = int(np.ceil(np.log2(max(max_samples, 2))))
        super()._fit(
            X,
            y,
            max_samples=max_samples,
            max_depth=max_depth,
            sample_weight=sample_weight,
            check_input=False,
        )

        self._average_path_length_per_tree, self._decision_path_lengths = zip(
            *[
                (
                    _average_path_length(tree.tree_.n_node_samples),
                    tree.tree_.compute_node_depths(),
                )
                for tree in self.estimators_
            ]
        )

        if self.contamination == "auto":
            # 0.5 plays a special role as described in the original paper.
            # we take the opposite as we consider the opposite of their score.
            self.offset_ = -0.5
            return self

        # Else, define offset_ wrt contamination parameter
        # To avoid performing input validation a second time we call
        # _score_samples rather than score_samples.
        # _score_samples expects a CSR matrix, so we convert if necessary.
        if issparse(X):
            X = X.tocsr()
        self.offset_ = np.percentile(self._score_samples(X), 100.0 * self.contamination)

        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # Returns an int array (n_samples,) with +1 for inliers and -1 where
        # decision_function(X) < 0.
        # Exceptions: NotFittedError if not fitted; validation errors from
        # score_samples.
        """
        Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            For each observation, tells whether or not (+1 or -1) it should
            be considered as an inlier according to the fitted model.

        Notes
        -----
        The predict method can be parallelized by setting a joblib context. This
        inherently does NOT use the ``n_jobs`` parameter initialized in the class,
        which is used during ``fit``. This is because, predict may actually be faster
        without parallelization for a small number of samples,
        such as for 1000 samples or less. The user can set the
        number of jobs in the joblib context to control the number of parallel jobs.

        .. code-block:: python

            from joblib import parallel_backend

            # Note, we use threading here as the predict method is not CPU bound.
            with parallel_backend("threading", n_jobs=4):
                model.predict(X)
        """
        check_is_fitted(self)
        decision_func = self.decision_function(X)
        is_inlier = np.ones_like(decision_func, dtype=int)
        is_inlier[decision_func < 0] = -1
        return is_inlier

    def decision_function(self, X):
        # [FUNCTION DOC]
        # Returns score_samples(X) minus offset_, so that negative values are outliers.
        # Output: float array (n_samples,).
        # KNOWN ISSUE: the docstring calls the trees "base classifiers"; they are
        # regressors.
        """
        Average anomaly score of X of the base classifiers.

        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.

        The measure of normality of an observation given a tree is the depth
        of the leaf containing this observation, which is equivalent to
        the number of splittings required to isolate this point. In case of
        several observations n_left in the leaf, the average path length of
        an n_left samples isolation tree is added.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal. Negative scores represent outliers,
            positive scores represent inliers.

        Notes
        -----
        The decision_function method can be parallelized by setting a joblib context.
        This inherently does NOT use the ``n_jobs`` parameter initialized in the class,
        which is used during ``fit``. This is because, calculating the score may
        actually be faster without parallelization for a small number of samples,
        such as for 1000 samples or less.
        The user can set the number of jobs in the joblib context to control the
        number of parallel jobs.

        .. code-block:: python

            from joblib import parallel_backend

            # Note, we use threading here as the decision_function method is
            # not CPU bound.
            with parallel_backend("threading", n_jobs=4):
                model.decision_function(X)
        """
        # We subtract self.offset_ to make 0 be the threshold value for being
        # an outlier:

        return self.score_samples(X) - self.offset_

    def score_samples(self, X):
        # [FUNCTION DOC]
        # Validates X (float32, CSR sparse allowed, NaN allowed, reset=False so feature
        # count and names are checked against fit) and returns _score_samples(X): float
        # array (n_samples,), around -1 for anomalies and closer to 0 for inliers.
        # Exceptions: ValueError on feature mismatch; NotFittedError if not fitted.
        # KNOWN ISSUE: the docstring example calls model.score(X), which IsolationForest
        # does not define; score_samples was meant.
        """
        Opposite of the anomaly score defined in the original paper.

        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.

        The measure of normality of an observation given a tree is the depth
        of the leaf containing this observation, which is equivalent to
        the number of splittings required to isolate this point. In case of
        several observations n_left in the leaf, the average path length of
        an n_left samples isolation tree is added.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal.

        Notes
        -----
        The score function method can be parallelized by setting a joblib context. This
        inherently does NOT use the ``n_jobs`` parameter initialized in the class,
        which is used during ``fit``. This is because, calculating the score may
        actually be faster without parallelization for a small number of samples,
        such as for 1000 samples or less.
        The user can set the number of jobs in the joblib context to control the
        number of parallel jobs.

        .. code-block:: python

            from joblib import parallel_backend

            # Note, we use threading here as the score_samples method is not CPU bound.
            with parallel_backend("threading", n_jobs=4):
                model.score(X)
        """
        # Check data
        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=np.float32,
            reset=False,
            ensure_all_finite=False,
        )

        return self._score_samples(X)

    def _score_samples(self, X):
        # [FUNCTION DOC]
        # HELPER: Unvalidated scoring: checks fitted state and returns the negation of
        # _compute_chunked_score_samples(X), so higher means more normal.
        """Private version of score_samples without input validation.

        Input validation would remove feature names, so we disable it.
        """
        # Code structure from ForestClassifier/predict_proba

        check_is_fitted(self)

        # Take the opposite of the scores as bigger is better (here less abnormal)
        return -self._compute_chunked_score_samples(X)

    def _compute_chunked_score_samples(self, X):
        # [FUNCTION DOC]
        # HELPER: Computes raw anomaly scores in row chunks to bound temporary memory.
        # Behavior: decides whether feature subsetting is needed (_max_features differs
        # from X's column count), computes chunk size from working_memory with 16 bytes
        # per selected feature per row (at least one row), and fills a float array of
        # scores by calling _compute_score_samples on each row slice.
        # Inputs: X - validated array or CSR matrix. Outputs: float array (n_samples,)
        # in (0, 1], higher = more anomalous.
        n_samples = _num_samples(X)

        if self._max_features == X.shape[1]:
            subsample_features = False
        else:
            subsample_features = True

        # We get as many rows as possible within our working_memory budget
        # (defined by sklearn.get_config()['working_memory']) to store
        # self._max_features in each row during computation.
        #
        # Note:
        #  - this will get at least 1 row, even if 1 row of score will
        #    exceed working_memory.
        #  - this does only account for temporary memory usage while loading
        #    the data needed to compute the scores -- the returned scores
        #    themselves are 1D.

        chunk_n_rows = get_chunk_n_rows(
            row_bytes=16 * self._max_features, max_n_rows=n_samples
        )
        slices = gen_batches(n_samples, chunk_n_rows)

        scores = np.zeros(n_samples, order="f")

        for sl in slices:
            # compute score on the slices of test samples:
            scores[sl] = self._compute_score_samples(X[sl], subsample_features)

        return scores

    def _compute_score_samples(self, X, subsample_features):
        # [FUNCTION DOC]
        # HELPER: Computes raw anomaly scores 2 ** (-E[h(x)] / c(max_samples)) for a
        # chunk.
        # Behavior: zero-initializes depths, then in a shared-memory joblib Parallel
        # (default n_jobs, overridable via a parallel_backend context) runs
        # _parallel_compute_tree_depths for every tree with its feature subset (when
        # subsample_features) and precomputed node arrays. Divides summed depths by
        # n_estimators times the average path length of _max_samples, using 1 where the
        # denominator is 0 (single training sample case, giving score 0.5), and returns
        # 2 to the negated ratio.
        # Inputs: X - array/CSR chunk; subsample_features - bool.
        # Outputs: float array (n_samples,).
        # Side effects: none on self; uses a thread lock.
        """
        Compute the score of each samples in X going through the extra trees.

        Parameters
        ----------
        X : array-like or sparse matrix
            Data matrix.

        subsample_features : bool
            Whether features should be subsampled.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The score of each sample in X.
        """
        n_samples = X.shape[0]

        depths = np.zeros(n_samples, order="f")

        average_path_length_max_samples = _average_path_length([self._max_samples])

        # Note: we use default n_jobs value, i.e. sequential computation, which
        # we expect to be more performant that parallelizing for small number
        # of samples, e.g. < 1k samples. Default n_jobs value can be overridden
        # by using joblib.parallel_backend context manager around
        # ._compute_score_samples. Using a higher n_jobs may speed up the
        # computation of the scores, e.g. for > 1k samples. See
        # https://github.com/scikit-learn/scikit-learn/pull/28622 for more
        # details.
        lock = threading.Lock()
        Parallel(
            verbose=self.verbose,
            require="sharedmem",
        )(
            delayed(_parallel_compute_tree_depths)(
                tree,
                X,
                features if subsample_features else None,
                self._decision_path_lengths[tree_idx],
                self._average_path_length_per_tree[tree_idx],
                depths,
                lock,
            )
            for tree_idx, (tree, features) in enumerate(
                zip(self.estimators_, self.estimators_features_)
            )
        )

        denominator = len(self.estimators_) * average_path_length_max_samples
        scores = 2 ** (
            # For a single training sample, denominator and depth are 0.
            # Therefore, we set the score manually to 1.
            -np.divide(
                depths, denominator, out=np.ones_like(depths), where=denominator != 0
            )
        )
        return scores

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Returns the parent's tags with input_tags.allow_nan set to True.
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


def _average_path_length(n_samples_leaf):
    # [FUNCTION DOC]
    # HELPER: Computes c(n), the average path length of an unsuccessful BST search in a
    # tree built on n samples, elementwise: 0 for n <= 1, 1 for n == 2, else 2 * (ln(n -
    # 1) + Euler's gamma) - 2 * (n - 1) / n.
    # Inputs: n_samples_leaf - array-like of counts (any shape, validated with
    # check_array, not required to be 2D).
    # Outputs: float ndarray with the same shape as the input.
    # Exceptions: ValueError from check_array for invalid input.
    """
    The average path length in an n_samples iTree, which is equal to
    the average path length of an unsuccessful BST search since the
    latter has the same structure as an isolation tree.
    Parameters
    ----------
    n_samples_leaf : array-like of shape (n_samples,)
        The number of training samples in each test sample leaf, for
        each estimators.

    Returns
    -------
    average_path_length : ndarray of shape (n_samples,)
    """

    n_samples_leaf = check_array(n_samples_leaf, ensure_2d=False)

    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.0
    average_path_length[mask_2] = 1.0
    average_path_length[not_mask] = (
        2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
        - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)
