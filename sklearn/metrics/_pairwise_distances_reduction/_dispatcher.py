# [MODULE DOC]
# Python-level dispatchers for the Cython pairwise-distances-reduction backends.
# Each public dispatcher class exposes classmethods that check whether the fast backend
# applies (is_usable_for, valid_metrics) and a compute classmethod that picks the
# float64 or float32 specialised Cython class from the dtype of X and Y and forwards
# all arguments to it. Also provides sqeuclidean_row_norms with the same dtype dispatch.
# No state is kept; classes are used only through their classmethods.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from abc import abstractmethod
from typing import List

import numpy as np
from scipy.sparse import issparse

from sklearn import get_config
from sklearn.metrics._dist_metrics import BOOL_METRICS, METRIC_MAPPING64, DistanceMetric
from sklearn.metrics._pairwise_distances_reduction._argkmin import ArgKmin32, ArgKmin64
from sklearn.metrics._pairwise_distances_reduction._argkmin_classmode import (
    ArgKminClassMode32,
    ArgKminClassMode64,
)
from sklearn.metrics._pairwise_distances_reduction._base import (
    _sqeuclidean_row_norms32,
    _sqeuclidean_row_norms64,
)
from sklearn.metrics._pairwise_distances_reduction._radius_neighbors import (
    RadiusNeighbors32,
    RadiusNeighbors64,
)
from sklearn.metrics._pairwise_distances_reduction._radius_neighbors_classmode import (
    RadiusNeighborsClassMode32,
    RadiusNeighborsClassMode64,
)


def sqeuclidean_row_norms(X, num_threads):
    # [FUNCTION DOC]
    # Computes the squared Euclidean norm of each row of X using the compiled
    # routine matching X.dtype (float64 or float32), run with num_threads OpenMP
    #   threads.
    # Inputs: X - C-contiguous ndarray or CSR matrix (n_samples, n_features);
    #   num_threads -
    # int.
    # Outputs: ndarray of shape (n_samples,) with the same float dtype as X.
    # Exceptions: ValueError if X.dtype is neither float64 nor float32.
    # Side effects: none.
    """Compute the squared euclidean norm of the rows of X in parallel.

    Parameters
    ----------
    X : ndarray or CSR matrix of shape (n_samples, n_features)
        Input data. Must be c-contiguous.

    num_threads : int
        The number of OpenMP threads to use.

    Returns
    -------
    sqeuclidean_row_norms : ndarray of shape (n_samples,)
        Arrays containing the squared euclidean norm of each row of X.
    """
    if X.dtype == np.float64:
        return np.asarray(_sqeuclidean_row_norms64(X, num_threads))
    if X.dtype == np.float32:
        return np.asarray(_sqeuclidean_row_norms32(X, num_threads))

    raise ValueError(
        "Only float64 or float32 datasets are supported at this time, "
        f"got: X.dtype={X.dtype}."
    )


class BaseDistancesReductionDispatcher:
    # [CLASS DOC]
    # Abstract base for the dispatchers: defines the default set of supported metrics,
    # the usability test for the Cython backend, and the abstract compute classmethod.
    # Not meant to be instantiated; all members are classmethods.
    # KNOWN ISSUE: compute is decorated with abstractmethod but the class does not use
    # ABCMeta, so the abstractness is not enforced at instantiation time.
    """Abstract base dispatcher for pairwise distance computation & reduction.

    Each dispatcher extending the base :class:`BaseDistancesReductionDispatcher`
    dispatcher must implement the :meth:`compute` classmethod.
    """

    @classmethod
    def valid_metrics(cls) -> List[str]:
        # [FUNCTION DOC]
        # Returns the sorted list of metric names accepted by the backend: 'sqeuclidean'
        #   plus
        # every key of METRIC_MAPPING64, minus 'pyfunc', 'mahalanobis', 'hamming' and
        #   all
        # boolean metrics (BOOL_METRICS).
        # Outputs: list of str. Side effects: none.
        excluded = {
            # PyFunc cannot be supported because it necessitates interacting with
            # the CPython interpreter to call user defined functions.
            "pyfunc",
            "mahalanobis",  # is numerically unstable
            # In order to support discrete distance metrics, we need to have a
            # stable simultaneous sort which preserves the order of the indices
            # because there generally is a lot of occurrences for a given values
            # of distances in this case.
            # TODO: implement a stable simultaneous_sort.
            "hamming",
            *BOOL_METRICS,
        }
        return sorted(({"sqeuclidean"} | set(METRIC_MAPPING64.keys())) - excluded)

    @classmethod
    def is_usable_for(cls, X, Y, metric) -> bool:
        # [FUNCTION DOC]
        # Decides whether the Cython reduction can handle the given inputs.
        # Returns False immediately when both X and Y are sparse and metric is a string
        # containing 'euclidean' (temporary fallback to SciPy for speed reasons).
        #   Otherwise
        # returns True only if all of these hold: the global config flag
        # enable_cython_pairwise_dist is truthy (default True); each of X and Y is
        #   either a
        # C-contiguous array or a valid CSR matrix (CSR format, at least one stored
        #   element,
        # int32 indices and indptr); X and Y share a dtype which is float32 or float64;
        #   and
        # metric is in cls.valid_metrics() or is a DistanceMetric instance.
        # Inputs: X, Y - arrays or sparse matrices (n_samples_*, n_features); metric -
        #   str or
        # DistanceMetric.
        # Outputs: bool. Exceptions: none raised directly (inputs lacking dtype would
        #   raise
        # AttributeError). Side effects: reads global config only.
        """Return True if the dispatcher can be used for the
        given parameters.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples_X, n_features)
            Input data.

        Y : {ndarray, sparse matrix} of shape (n_samples_Y, n_features)
            Input data.

        metric : str, default='euclidean'
            The distance metric to use.
            For a list of available metrics, see the documentation of
            :class:`~sklearn.metrics.DistanceMetric`.

        Returns
        -------
        True if the dispatcher can be used, else False.
        """

        # FIXME: the current Cython implementation is too slow for a large number of
        # features. We temporarily disable it to fallback on SciPy's implementation.
        # See: https://github.com/scikit-learn/scikit-learn/issues/28191
        if (
            issparse(X)
            and issparse(Y)
            and isinstance(metric, str)
            and "euclidean" in metric
        ):
            return False

        def is_numpy_c_ordered(X):
            # [FUNCTION DOC]
            # HELPER: True if the object has a flags attribute whose c_contiguous flag
            #   is true.
            # Inputs: X - any object. Outputs: bool. Side effects: none.
            return hasattr(X, "flags") and getattr(X.flags, "c_contiguous", False)

        def is_valid_sparse_matrix(X):
            # [FUNCTION DOC]
            # HELPER: True if X is a scipy sparse matrix in CSR format with nnz > 0 and
            #   both
            # indices and indptr of dtype int32.
            # Inputs: X - any object. Outputs: bool. Side effects: none.
            return (
                issparse(X)
                and X.format == "csr"
                and
                # TODO: support CSR matrices without non-zeros elements
                X.nnz > 0
                and
                # TODO: support CSR matrices with int64 indices and indptr
                # See: https://github.com/scikit-learn/scikit-learn/issues/23653
                X.indices.dtype == X.indptr.dtype == np.int32
            )

        is_usable = (
            get_config().get("enable_cython_pairwise_dist", True)
            and (is_numpy_c_ordered(X) or is_valid_sparse_matrix(X))
            and (is_numpy_c_ordered(Y) or is_valid_sparse_matrix(Y))
            and X.dtype == Y.dtype
            and X.dtype in (np.float32, np.float64)
            and (metric in cls.valid_metrics() or isinstance(metric, DistanceMetric))
        )

        return is_usable

    @classmethod
    @abstractmethod
    def compute(
        cls,
        X,
        Y,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Abstract hook: subclasses compute the reduction of X against Y with
        # reduction-specific keyword arguments. The base body is only a docstring and
        #   returns
        # None.
        # Inputs: X, Y - arrays or CSR matrices; kwargs - reduction parameters.
        """Compute the reduction.

        Parameters
        ----------
        X : ndarray or CSR matrix of shape (n_samples_X, n_features)
            Input data.

        Y : ndarray or CSR matrix of shape (n_samples_Y, n_features)
            Input data.

        **kwargs : additional parameters for the reduction

        Notes
        -----
        This method is an abstract class method: it has to be implemented
        for all subclasses.
        """


class ArgKmin(BaseDistancesReductionDispatcher):
    # [CLASS DOC]
    # Dispatcher for the k-smallest-distance (argkmin) reduction used by brute-force
    # k-nearest-neighbor queries. Inherits the default metric list and usability check.
    """Compute the argkmin of row vectors of X on the ones of Y.

    For each row vector of X, computes the indices of k first the rows
    vectors of Y with the smallest distances.

    ArgKmin is typically used to perform
    bruteforce k-nearest neighbors queries.

    This class is not meant to be instantiated, one should only use
    its :meth:`compute` classmethod which handles allocation and
    deallocation consistently.
    """

    @classmethod
    def compute(
        cls,
        X,
        Y,
        k,
        metric="euclidean",
        chunk_size=None,
        metric_kwargs=None,
        strategy=None,
        return_distance=False,
    ):
        # [FUNCTION DOC]
        # Forwards all arguments unchanged to ArgKmin64.compute when X and Y are both
        # float64, or to ArgKmin32.compute when both are float32.
        # Inputs: see docstring (k int, metric str, chunk_size/metric_kwargs/strategy
        #   optional,
        # return_distance bool).
        # Outputs: indices array (n_samples_X, k), or a (distances, indices) pair when
        # return_distance is True, as returned by the backend.
        # Exceptions: ValueError if the dtypes differ or are not float32/float64.
        # Side effects: none beyond backend allocation and multithreaded computation.
        """Compute the argkmin reduction.

        Parameters
        ----------
        X : ndarray or CSR matrix of shape (n_samples_X, n_features)
            Input data.

        Y : ndarray or CSR matrix of shape (n_samples_Y, n_features)
            Input data.

        k : int
            The k for the argkmin reduction.

        metric : str, default='euclidean'
            The distance metric to use for argkmin.
            For a list of available metrics, see the documentation of
            :class:`~sklearn.metrics.DistanceMetric`.

        chunk_size : int, default=None,
            The number of vectors per chunk. If None (default) looks-up in
            scikit-learn configuration for `pairwise_dist_chunk_size`,
            and use 256 if it is not set.

        metric_kwargs : dict, default=None
            Keyword arguments to pass to specified metric function.

        strategy : str, {'auto', 'parallel_on_X', 'parallel_on_Y'}, default=None
            The chunking strategy defining which dataset parallelization are made on.

            For both strategies the computations happens with two nested loops,
            respectively on chunks of X and chunks of Y.
            Strategies differs on which loop (outer or inner) is made to run
            in parallel with the Cython `prange` construct:

              - 'parallel_on_X' dispatches chunks of X uniformly on threads.
                Each thread then iterates on all the chunks of Y. This strategy is
                embarrassingly parallel and comes with no datastructures
                synchronisation.

              - 'parallel_on_Y' dispatches chunks of Y uniformly on threads.
                Each thread processes all the chunks of X in turn. This strategy is
                a sequence of embarrassingly parallel subtasks (the inner loop on Y
                chunks) with intermediate datastructures synchronisation at each
                iteration of the sequential outer loop on X chunks.

              - 'auto' relies on a simple heuristic to choose between
                'parallel_on_X' and 'parallel_on_Y': when `X.shape[0]` is large enough,
                'parallel_on_X' is usually the most efficient strategy.
                When `X.shape[0]` is small but `Y.shape[0]` is large, 'parallel_on_Y'
                brings more opportunity for parallelism and is therefore more efficient

              - None (default) looks-up in scikit-learn configuration for
                `pairwise_dist_parallel_strategy`, and use 'auto' if it is not set.

        return_distance : boolean, default=False
            Return distances between each X vector and its
            argkmin if set to True.

        Returns
        -------
        If return_distance=False:
          - argkmin_indices : ndarray of shape (n_samples_X, k)
            Indices of the argkmin for each vector in X.

        If return_distance=True:
          - argkmin_distances : ndarray of shape (n_samples_X, k)
            Distances to the argkmin for each vector in X.
          - argkmin_indices : ndarray of shape (n_samples_X, k)
            Indices of the argkmin for each vector in X.

        Notes
        -----
        This classmethod inspects the arguments values to dispatch to the
        dtype-specialized implementation of :class:`ArgKmin`.

        This allows decoupling the API entirely from the implementation details
        whilst maintaining RAII: all temporarily allocated datastructures necessary
        for the concrete implementation are therefore freed when this classmethod
        returns.
        """
        if X.dtype == Y.dtype == np.float64:
            return ArgKmin64.compute(
                X=X,
                Y=Y,
                k=k,
                metric=metric,
                chunk_size=chunk_size,
                metric_kwargs=metric_kwargs,
                strategy=strategy,
                return_distance=return_distance,
            )

        if X.dtype == Y.dtype == np.float32:
            return ArgKmin32.compute(
                X=X,
                Y=Y,
                k=k,
                metric=metric,
                chunk_size=chunk_size,
                metric_kwargs=metric_kwargs,
                strategy=strategy,
                return_distance=return_distance,
            )

        raise ValueError(
            "Only float64 or float32 datasets pairs are supported at this time, "
            f"got: X.dtype={X.dtype} and Y.dtype={Y.dtype}."
        )


class RadiusNeighbors(BaseDistancesReductionDispatcher):
    # [CLASS DOC]
    # Dispatcher for the radius-neighborhood reduction: for each row of X, all rows of Y
    # within distance radius. Inherits the default metric list and usability check.
    """Compute radius-based neighbors for two sets of vectors.

    For each row-vector X[i] of the queries X, find all the indices j of
    row-vectors in Y such that:

                        dist(X[i], Y[j]) <= radius

    The distance function `dist` depends on the values of the `metric`
    and `metric_kwargs` parameters.

    This class is not meant to be instantiated, one should only use
    its :meth:`compute` classmethod which handles allocation and
    deallocation consistently.
    """

    @classmethod
    def compute(
        cls,
        X,
        Y,
        radius,
        metric="euclidean",
        chunk_size=None,
        metric_kwargs=None,
        strategy=None,
        return_distance=False,
        sort_results=False,
    ):
        # [FUNCTION DOC]
        # Forwards all arguments to RadiusNeighbors64.compute or
        #   RadiusNeighbors32.compute
        # according to the common float dtype of X and Y.
        # Inputs: radius float; sort_results bool; return_distance bool; others as in
        #   ArgKmin.
        # Outputs: object array of per-query index arrays, or (distances, indices) pair
        #   when
        # return_distance is True, as returned by the backend.
        # KNOWN ISSUE: the docstring lists the return_distance=True outputs as
        # (indices, distances); the backend returns distances first, matching ArgKmin.
        # Exceptions: ValueError if the dtypes differ or are not float32/float64.
        # Side effects: none.
        """Return the results of the reduction for the given arguments.

        Parameters
        ----------
        X : ndarray or CSR matrix of shape (n_samples_X, n_features)
            Input data.

        Y : ndarray or CSR matrix of shape (n_samples_Y, n_features)
            Input data.

        radius : float
            The radius defining the neighborhood.

        metric : str, default='euclidean'
            The distance metric to use.
            For a list of available metrics, see the documentation of
            :class:`~sklearn.metrics.DistanceMetric`.

        chunk_size : int, default=None,
            The number of vectors per chunk. If None (default) looks-up in
            scikit-learn configuration for `pairwise_dist_chunk_size`,
            and use 256 if it is not set.

        metric_kwargs : dict, default=None
            Keyword arguments to pass to specified metric function.

        strategy : str, {'auto', 'parallel_on_X', 'parallel_on_Y'}, default=None
            The chunking strategy defining which dataset parallelization are made on.

            For both strategies the computations happens with two nested loops,
            respectively on chunks of X and chunks of Y.
            Strategies differs on which loop (outer or inner) is made to run
            in parallel with the Cython `prange` construct:

              - 'parallel_on_X' dispatches chunks of X uniformly on threads.
                Each thread then iterates on all the chunks of Y. This strategy is
                embarrassingly parallel and comes with no datastructures
                synchronisation.

              - 'parallel_on_Y' dispatches chunks of Y uniformly on threads.
                Each thread processes all the chunks of X in turn. This strategy is
                a sequence of embarrassingly parallel subtasks (the inner loop on Y
                chunks) with intermediate datastructures synchronisation at each
                iteration of the sequential outer loop on X chunks.

              - 'auto' relies on a simple heuristic to choose between
                'parallel_on_X' and 'parallel_on_Y': when `X.shape[0]` is large enough,
                'parallel_on_X' is usually the most efficient strategy.
                When `X.shape[0]` is small but `Y.shape[0]` is large, 'parallel_on_Y'
                brings more opportunity for parallelism and is therefore more efficient
                despite the synchronization step at each iteration of the outer loop
                on chunks of `X`.

              - None (default) looks-up in scikit-learn configuration for
                `pairwise_dist_parallel_strategy`, and use 'auto' if it is not set.

        return_distance : boolean, default=False
            Return distances between each X vector and its neighbors if set to True.

        sort_results : boolean, default=False
            Sort results with respect to distances between each X vector and its
            neighbors if set to True.

        Returns
        -------
        If return_distance=False:
          - neighbors_indices : ndarray of n_samples_X ndarray
            Indices of the neighbors for each vector in X.

        If return_distance=True:
          - neighbors_indices : ndarray of n_samples_X ndarray
            Indices of the neighbors for each vector in X.
          - neighbors_distances : ndarray of n_samples_X ndarray
            Distances to the neighbors for each vector in X.

        Notes
        -----
        This classmethod inspects the arguments values to dispatch to the
        dtype-specialized implementation of :class:`RadiusNeighbors`.

        This allows decoupling the API entirely from the implementation details
        whilst maintaining RAII: all temporarily allocated datastructures necessary
        for the concrete implementation are therefore freed when this classmethod
        returns.
        """
        if X.dtype == Y.dtype == np.float64:
            return RadiusNeighbors64.compute(
                X=X,
                Y=Y,
                radius=radius,
                metric=metric,
                chunk_size=chunk_size,
                metric_kwargs=metric_kwargs,
                strategy=strategy,
                sort_results=sort_results,
                return_distance=return_distance,
            )

        if X.dtype == Y.dtype == np.float32:
            return RadiusNeighbors32.compute(
                X=X,
                Y=Y,
                radius=radius,
                metric=metric,
                chunk_size=chunk_size,
                metric_kwargs=metric_kwargs,
                strategy=strategy,
                sort_results=sort_results,
                return_distance=return_distance,
            )

        raise ValueError(
            "Only float64 or float32 datasets pairs are supported at this time, "
            f"got: X.dtype={X.dtype} and Y.dtype={Y.dtype}."
        )


class ArgKminClassMode(BaseDistancesReductionDispatcher):
    # [CLASS DOC]
    # Dispatcher for argkmin followed by a (weighted) vote over neighbor labels, used by
    # KNeighborsClassifier.predict/predict_proba. Excludes euclidean metrics.
    """Compute the argkmin of row vectors of X on the ones of Y with labels.

    For each row vector of X, computes the indices of k first the rows
    vectors of Y with the smallest distances. Computes weighted mode of labels.

    ArgKminClassMode is typically used to perform bruteforce k-nearest neighbors
    queries when the weighted mode of the labels for the k-nearest neighbors
    are required, such as in `predict` methods.

    This class is not meant to be instantiated, one should only use
    its :meth:`compute` classmethod which handles allocation and
    deallocation consistently.
    """

    @classmethod
    def valid_metrics(cls) -> List[str]:
        # [FUNCTION DOC]
        # Returns the base valid metrics minus 'euclidean' and 'sqeuclidean'.
        # Outputs: list of str.
        # KNOWN ISSUE: unlike the base and RadiusNeighborsClassMode versions, the list
        #   is built
        # from a set and is not sorted, so its order is not deterministic across runs.
        excluded = {
            # Euclidean is technically usable for ArgKminClassMode
            # but its current implementation would not be competitive.
            # TODO: implement Euclidean specialization using GEMM.
            "euclidean",
            "sqeuclidean",
        }
        return list(set(BaseDistancesReductionDispatcher.valid_metrics()) - excluded)

    @classmethod
    def compute(
        cls,
        X,
        Y,
        k,
        weights,
        Y_labels,
        unique_Y_labels,
        metric="euclidean",
        chunk_size=None,
        metric_kwargs=None,
        strategy=None,
    ):
        # [FUNCTION DOC]
        # Validates weights then forwards to ArgKminClassMode64/32.compute according to
        #   the
        # common float dtype, converting Y_labels and unique_Y_labels to intp numpy
        #   arrays.
        # Inputs: k int; weights - str 'uniform' or 'distance' (despite the docstring
        #   saying
        # ndarray); Y_labels - label index per row of Y; unique_Y_labels - all label
        #   indices;
        # metric, chunk_size, metric_kwargs, strategy as in ArgKmin.
        # Outputs: probabilities ndarray (n_samples_X, n_classes).
        # Exceptions: ValueError if weights is not 'uniform'/'distance', or dtypes
        #   unsupported.
        # Side effects: none.
        """Compute the argkmin reduction.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            The input array to be labelled.

        Y : ndarray of shape (n_samples_Y, n_features)
            The input array whose class membership are provided through the
            `Y_labels` parameter.

        k : int
            The number of nearest neighbors to consider.

        weights : ndarray
            The weights applied over the `Y_labels` of `Y` when computing the
            weighted mode of the labels.

        Y_labels : ndarray
            An array containing the index of the class membership of the
            associated samples in `Y`. This is used in labeling `X`.

        unique_Y_labels : ndarray
            An array containing all unique indices contained in the
            corresponding `Y_labels` array.

        metric : str, default='euclidean'
            The distance metric to use. For a list of available metrics, see
            the documentation of :class:`~sklearn.metrics.DistanceMetric`.
            Currently does not support `'precomputed'`.

        chunk_size : int, default=None,
            The number of vectors per chunk. If None (default) looks-up in
            scikit-learn configuration for `pairwise_dist_chunk_size`,
            and use 256 if it is not set.

        metric_kwargs : dict, default=None
            Keyword arguments to pass to specified metric function.

        strategy : str, {'auto', 'parallel_on_X', 'parallel_on_Y'}, default=None
            The chunking strategy defining which dataset parallelization are made on.

            For both strategies the computations happens with two nested loops,
            respectively on chunks of X and chunks of Y.
            Strategies differs on which loop (outer or inner) is made to run
            in parallel with the Cython `prange` construct:

              - 'parallel_on_X' dispatches chunks of X uniformly on threads.
                Each thread then iterates on all the chunks of Y. This strategy is
                embarrassingly parallel and comes with no datastructures
                synchronisation.

              - 'parallel_on_Y' dispatches chunks of Y uniformly on threads.
                Each thread processes all the chunks of X in turn. This strategy is
                a sequence of embarrassingly parallel subtasks (the inner loop on Y
                chunks) with intermediate datastructures synchronisation at each
                iteration of the sequential outer loop on X chunks.

              - 'auto' relies on a simple heuristic to choose between
                'parallel_on_X' and 'parallel_on_Y': when `X.shape[0]` is large enough,
                'parallel_on_X' is usually the most efficient strategy.
                When `X.shape[0]` is small but `Y.shape[0]` is large, 'parallel_on_Y'
                brings more opportunity for parallelism and is therefore more efficient
                despite the synchronization step at each iteration of the outer loop
                on chunks of `X`.

              - None (default) looks-up in scikit-learn configuration for
                `pairwise_dist_parallel_strategy`, and use 'auto' if it is not set.

        Returns
        -------
        probabilities : ndarray of shape (n_samples_X, n_classes)
            An array containing the class probabilities for each sample.

        Notes
        -----
        This classmethod is responsible for introspecting the arguments
        values to dispatch to the most appropriate implementation of
        :class:`PairwiseDistancesArgKmin`.

        This allows decoupling the API entirely from the implementation details
        whilst maintaining RAII: all temporarily allocated datastructures necessary
        for the concrete implementation are therefore freed when this classmethod
        returns.
        """
        if weights not in {"uniform", "distance"}:
            raise ValueError(
                "Only the 'uniform' or 'distance' weights options are supported"
                f" at this time. Got: {weights=}."
            )
        if X.dtype == Y.dtype == np.float64:
            return ArgKminClassMode64.compute(
                X=X,
                Y=Y,
                k=k,
                weights=weights,
                Y_labels=np.array(Y_labels, dtype=np.intp),
                unique_Y_labels=np.array(unique_Y_labels, dtype=np.intp),
                metric=metric,
                chunk_size=chunk_size,
                metric_kwargs=metric_kwargs,
                strategy=strategy,
            )

        if X.dtype == Y.dtype == np.float32:
            return ArgKminClassMode32.compute(
                X=X,
                Y=Y,
                k=k,
                weights=weights,
                Y_labels=np.array(Y_labels, dtype=np.intp),
                unique_Y_labels=np.array(unique_Y_labels, dtype=np.intp),
                metric=metric,
                chunk_size=chunk_size,
                metric_kwargs=metric_kwargs,
                strategy=strategy,
            )

        raise ValueError(
            "Only float64 or float32 datasets pairs are supported at this time, "
            f"got: X.dtype={X.dtype} and Y.dtype={Y.dtype}."
        )


class RadiusNeighborsClassMode(BaseDistancesReductionDispatcher):
    # [CLASS DOC]
    # Dispatcher for radius neighbors followed by a (weighted) vote over neighbor
    #   labels,
    # used by RadiusNeighborsClassifier. Excludes euclidean metrics.
    """Compute radius-based class modes of row vectors of X using the
    those of Y.

    For each row-vector X[i] of the queries X, find all the indices j of
    row-vectors in Y such that:

                        dist(X[i], Y[j]) <= radius

    RadiusNeighborsClassMode is typically used to perform bruteforce
    radius neighbors queries when the weighted mode of the labels for
    the nearest neighbors within the specified radius are required,
    such as in `predict` methods.

    This class is not meant to be instantiated, one should only use
    its :meth:`compute` classmethod which handles allocation and
    deallocation consistently.
    """

    @classmethod
    def valid_metrics(cls) -> List[str]:
        # [FUNCTION DOC]
        # Returns the sorted base valid metrics minus 'euclidean' and 'sqeuclidean'.
        # Outputs: list of str.
        excluded = {
            # Euclidean is technically usable for RadiusNeighborsClassMode
            # but it would not be competitive.
            # TODO: implement Euclidean specialization using GEMM.
            "euclidean",
            "sqeuclidean",
        }
        return sorted(set(BaseDistancesReductionDispatcher.valid_metrics()) - excluded)

    @classmethod
    def compute(
        cls,
        X,
        Y,
        radius,
        weights,
        Y_labels,
        unique_Y_labels,
        outlier_label,
        metric="euclidean",
        chunk_size=None,
        metric_kwargs=None,
        strategy=None,
    ):
        # [FUNCTION DOC]
        # Validates weights then forwards to RadiusNeighborsClassMode64/32.compute
        #   according
        # to the common float dtype, converting Y_labels and unique_Y_labels to intp
        #   arrays and
        # passing outlier_label through.
        # Inputs: radius float; weights - 'uniform' or 'distance'; Y_labels,
        #   unique_Y_labels;
        # outlier_label - label for queries without neighbors (handled by backend);
        #   remaining
        # as in RadiusNeighbors.
        # Outputs: probabilities ndarray (n_samples_X, n_classes).
        # Exceptions: ValueError for invalid weights or unsupported dtypes.
        # Side effects: none.
        """Return the results of the reduction for the given arguments.
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            The input array to be labelled.
        Y : ndarray of shape (n_samples_Y, n_features)
            The input array whose class membership is provided through
            the `Y_labels` parameter.
        radius : float
            The radius defining the neighborhood.
        weights : ndarray
            The weights applied to the `Y_labels` when computing the
            weighted mode of the labels.
        Y_labels : ndarray
            An array containing the index of the class membership of the
            associated samples in `Y`. This is used in labeling `X`.
        unique_Y_labels : ndarray
            An array containing all unique class labels.
        outlier_label : int, default=None
            Label for outlier samples (samples with no neighbors in given
            radius). In the default case when the value is None if any
            outlier is detected, a ValueError will be raised. The outlier
            label should be selected from among the unique 'Y' labels. If
            it is specified with a different value a warning will be raised
            and all class probabilities of outliers will be assigned to be 0.
        metric : str, default='euclidean'
            The distance metric to use. For a list of available metrics, see
            the documentation of :class:`~sklearn.metrics.DistanceMetric`.
            Currently does not support `'precomputed'`.
        chunk_size : int, default=None,
            The number of vectors per chunk. If None (default) looks-up in
            scikit-learn configuration for `pairwise_dist_chunk_size`,
            and use 256 if it is not set.
        metric_kwargs : dict, default=None
            Keyword arguments to pass to specified metric function.
        strategy : str, {'auto', 'parallel_on_X', 'parallel_on_Y'}, default=None
            The chunking strategy defining which dataset parallelization are made on.
            For both strategies the computations happens with two nested loops,
            respectively on chunks of X and chunks of Y.
            Strategies differs on which loop (outer or inner) is made to run
            in parallel with the Cython `prange` construct:
              - 'parallel_on_X' dispatches chunks of X uniformly on threads.
                Each thread then iterates on all the chunks of Y. This strategy is
                embarrassingly parallel and comes with no datastructures
                synchronisation.
              - 'parallel_on_Y' dispatches chunks of Y uniformly on threads.
                Each thread processes all the chunks of X in turn. This strategy is
                a sequence of embarrassingly parallel subtasks (the inner loop on Y
                chunks) with intermediate datastructures synchronisation at each
                iteration of the sequential outer loop on X chunks.
              - 'auto' relies on a simple heuristic to choose between
                'parallel_on_X' and 'parallel_on_Y': when `X.shape[0]` is large enough,
                'parallel_on_X' is usually the most efficient strategy.
                When `X.shape[0]` is small but `Y.shape[0]` is large, 'parallel_on_Y'
                brings more opportunity for parallelism and is therefore more efficient
                despite the synchronization step at each iteration of the outer loop
                on chunks of `X`.
              - None (default) looks-up in scikit-learn configuration for
                `pairwise_dist_parallel_strategy`, and use 'auto' if it is not set.
        Returns
        -------
        probabilities : ndarray of shape (n_samples_X, n_classes)
            An array containing the class probabilities for each sample.
        """
        if weights not in {"uniform", "distance"}:
            raise ValueError(
                "Only the 'uniform' or 'distance' weights options are supported"
                f" at this time. Got: {weights=}."
            )
        if X.dtype == Y.dtype == np.float64:
            return RadiusNeighborsClassMode64.compute(
                X=X,
                Y=Y,
                radius=radius,
                weights=weights,
                Y_labels=np.array(Y_labels, dtype=np.intp),
                unique_Y_labels=np.array(unique_Y_labels, dtype=np.intp),
                outlier_label=outlier_label,
                metric=metric,
                chunk_size=chunk_size,
                metric_kwargs=metric_kwargs,
                strategy=strategy,
            )

        if X.dtype == Y.dtype == np.float32:
            return RadiusNeighborsClassMode32.compute(
                X=X,
                Y=Y,
                radius=radius,
                weights=weights,
                Y_labels=np.array(Y_labels, dtype=np.intp),
                unique_Y_labels=np.array(unique_Y_labels, dtype=np.intp),
                outlier_label=outlier_label,
                metric=metric,
                chunk_size=chunk_size,
                metric_kwargs=metric_kwargs,
                strategy=strategy,
            )

        raise ValueError(
            "Only float64 or float32 datasets pairs are supported at this time, "
            f"got: X.dtype={X.dtype} and Y.dtype={Y.dtype}."
        )
