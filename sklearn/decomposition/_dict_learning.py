# [MODULE DOC]
# Dictionary learning and sparse coding. Learns a dictionary D (n_components x
# n_features) and sparse codes U (n_samples x n_components) minimizing half the squared
# reconstruction error of X by U times D plus alpha times the L1 norm of U, with atoms
# constrained to norm at most 1. Provides the sparse_encode function (solve for U given
# D with LARS, Lasso-LARS, Lasso-CD, OMP or soft thresholding), the batch solver
# dict_learning (alternating minimization), the online solver dict_learning_online, and
# the estimators SparseCoder (fixed dictionary), DictionaryLearning and
# MiniBatchDictionaryLearning, which share _BaseSparseCoding for transform /
# inverse_transform.
"""Dictionary learning."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import sys
import time
from numbers import Integral, Real

import numpy as np
from joblib import effective_n_jobs
from scipy import linalg

from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.linear_model import Lars, Lasso, LassoLars, orthogonal_mp_gram
from sklearn.utils import check_array, check_random_state, gen_batches, gen_even_slices
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.extmath import _randomized_svd, row_norms, svd_flip
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted, validate_data


def _check_positive_coding(method, positive):
    # [FUNCTION DOC]
    # HELPER: Rejects positive=True for coding methods that do not support it.
    # Inputs: method - coding algorithm name; positive - bool.
    # Exceptions: ValueError if positive and method is "omp" or "lars".
    # Side effects: none.
    if positive and method in ["omp", "lars"]:
        raise ValueError(
            "Positive constraint not supported for '{}' coding method.".format(method)
        )


def _sparse_encode_precomputed(
    X,
    dictionary,
    *,
    gram=None,
    cov=None,
    algorithm="lasso_lars",
    regularization=None,
    copy_cov=True,
    init=None,
    max_iter=1000,
    verbose=0,
    positive=False,
):
    # [FUNCTION DOC]
    # HELPER: Solves the sparse coding problem for all rows of X with one algorithm,
    # using precomputed gram (dictionary times its transpose) and/or cov (dictionary
    # times X transposed). Per algorithm:
    # - lasso_lars: LassoLars with alpha equal to regularization divided by n_features
    # (the linear models scale the loss by the number of rows of the design, here
    # n_features), no intercept, precomputed gram, cov passed as Xy, single solution
    # (fit_path False); numpy floating errors are silenced during fit and restored
    # after.
    # - lasso_cd: Lasso with the same alpha scaling, tol 1e-8, warm start from init
    # (copied if read-only); cov is not used; fits without input checks.
    # - lars: Lars limited to int(regularization) nonzero coefficients, same error
    # handling.
    # - threshold: soft-thresholds cov by regularization (sign times max of absolute
    # value minus regularization and 0), transposed; clipped at 0 if positive.
    # - omp: orthogonal_mp_gram with int(regularization) nonzero coefficients, the
    # squared row norms of X and copy_Xy equal to copy_cov.
    # Inputs: X (n_samples, n_features); dictionary (n_components, n_features); gram
    # (n_components, n_components) or None; cov (n_components, n_samples) or None;
    # algorithm str (default lasso_lars); regularization - alpha or number of nonzero
    # coefficients depending on algorithm; copy_cov bool; init (n_samples, n_components)
    # or None (lasso_cd only); max_iter int; verbose; positive bool.
    # Outputs: code array reshaped to (n_samples, n_components).
    # Exceptions: errors from the underlying linear models; an unknown algorithm would
    # leave new_code undefined (NameError), but callers validate the name.
    # Side effects: omp may overwrite cov when copy_cov is False; temporarily changes
    # numpy error state.
    """Generic sparse coding with precomputed Gram and/or covariance matrices.

    Each row of the result is the solution to a Lasso problem.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix.

    dictionary : ndarray of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows.

    gram : ndarray of shape (n_components, n_components), default=None
        Precomputed Gram matrix, `dictionary * dictionary'`
        gram can be `None` if method is 'threshold'.

    cov : ndarray of shape (n_components, n_samples), default=None
        Precomputed covariance, `dictionary * X'`.

    algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}, \
            default='lasso_lars'
        The algorithm used:

        * `'lars'`: uses the least angle regression method
          (`linear_model.lars_path`);
        * `'lasso_lars'`: uses Lars to compute the Lasso solution;
        * `'lasso_cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). lasso_lars will be faster if
          the estimated components are sparse;
        * `'omp'`: uses orthogonal matching pursuit to estimate the sparse
          solution;
        * `'threshold'`: squashes to zero all coefficients less than
          regularization from the projection `dictionary * data'`.

    regularization : int or float, default=None
        The regularization parameter. It corresponds to alpha when
        algorithm is `'lasso_lars'`, `'lasso_cd'` or `'threshold'`.
        Otherwise it corresponds to `n_nonzero_coefs`.

    init : ndarray of shape (n_samples, n_components), default=None
        Initialization value of the sparse code. Only used if
        `algorithm='lasso_cd'`.

    max_iter : int, default=1000
        Maximum number of iterations to perform if `algorithm='lasso_cd'` or
        `'lasso_lars'`.

    copy_cov : bool, default=True
        Whether to copy the precomputed covariance matrix; if `False`, it may
        be overwritten.

    verbose : int, default=0
        Controls the verbosity; the higher, the more messages.

    positive: bool, default=False
        Whether to enforce a positivity constraint on the sparse code.

        .. versionadded:: 0.20

    Returns
    -------
    code : ndarray of shape (n_components, n_features)
        The sparse codes.
    """
    n_samples, n_features = X.shape
    n_components = dictionary.shape[0]

    if algorithm == "lasso_lars":
        alpha = float(regularization) / n_features  # account for scaling
        try:
            err_mgt = np.seterr(all="ignore")

            # Not passing in verbose=max(0, verbose-1) because Lars.fit already
            # corrects the verbosity level.
            lasso_lars = LassoLars(
                alpha=alpha,
                fit_intercept=False,
                verbose=verbose,
                precompute=gram,
                fit_path=False,
                positive=positive,
                max_iter=max_iter,
            )
            lasso_lars.fit(dictionary.T, X.T, Xy=cov)
            new_code = lasso_lars.coef_
        finally:
            np.seterr(**err_mgt)

    elif algorithm == "lasso_cd":
        alpha = float(regularization) / n_features  # account for scaling

        # TODO: Make verbosity argument for Lasso?
        # sklearn.linear_model.coordinate_descent.enet_path has a verbosity
        # argument that we could pass in from Lasso.
        clf = Lasso(
            alpha=alpha,
            fit_intercept=False,
            precompute=gram,
            tol=1e-8,  # TODO: This parameter should be exposed.
            max_iter=max_iter,
            warm_start=True,
            positive=positive,
        )

        if init is not None:
            # In some workflows using coordinate descent algorithms:
            #  - users might provide NumPy arrays with read-only buffers
            #  - `joblib` might memmap arrays making their buffer read-only
            # TODO: move this handling (which is currently too broad)
            # closer to the actual private function which need buffers to be writable.
            if not init.flags["WRITEABLE"]:
                init = np.array(init)
            clf.coef_ = init

        clf.fit(dictionary.T, X.T, check_input=False)
        new_code = clf.coef_

    elif algorithm == "lars":
        try:
            err_mgt = np.seterr(all="ignore")

            # Not passing in verbose=max(0, verbose-1) because Lars.fit already
            # corrects the verbosity level.
            lars = Lars(
                fit_intercept=False,
                verbose=verbose,
                precompute=gram,
                n_nonzero_coefs=int(regularization),
                fit_path=False,
            )
            lars.fit(dictionary.T, X.T, Xy=cov)
            new_code = lars.coef_
        finally:
            np.seterr(**err_mgt)

    elif algorithm == "threshold":
        new_code = (np.sign(cov) * np.maximum(np.abs(cov) - regularization, 0)).T
        if positive:
            np.clip(new_code, 0, None, out=new_code)

    elif algorithm == "omp":
        new_code = orthogonal_mp_gram(
            Gram=gram,
            Xy=cov,
            n_nonzero_coefs=int(regularization),
            tol=None,
            norms_squared=row_norms(X, squared=True),
            copy_Xy=copy_cov,
        ).T

    return new_code.reshape(n_samples, n_components)


@validate_params(
    {
        "X": ["array-like"],
        "dictionary": ["array-like"],
        "gram": ["array-like", None],
        "cov": ["array-like", None],
        "algorithm": [
            StrOptions({"lasso_lars", "lasso_cd", "lars", "omp", "threshold"})
        ],
        "n_nonzero_coefs": [Interval(Integral, 1, None, closed="left"), None],
        "alpha": [Interval(Real, 0, None, closed="left"), None],
        "copy_cov": ["boolean"],
        "init": ["array-like", None],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "n_jobs": [Integral, None],
        "check_input": ["boolean"],
        "verbose": ["verbose"],
        "positive": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
# XXX : could be moved to the linear_model module
def sparse_encode(
    X,
    dictionary,
    *,
    gram=None,
    cov=None,
    algorithm="lasso_lars",
    n_nonzero_coefs=None,
    alpha=None,
    copy_cov=True,
    init=None,
    max_iter=1000,
    n_jobs=None,
    check_input=True,
    verbose=0,
    positive=False,
):
    # [FUNCTION DOC]
    # Public sparse coding entry point (parameters validated by validate_params). If
    # check_input, converts dictionary and X to float64/float32 arrays (C order for
    # lasso_cd). Checks feature counts and positive support, then delegates to
    # _sparse_encode with all options.
    # Inputs: see _sparse_encode; n_nonzero_coefs and alpha select the regularization;
    # n_jobs parallelizes over samples; check_input bool (default True).
    # Outputs: code (n_samples, n_components).
    # Exceptions: ValueError if dictionary and X feature counts differ or positive is
    # unsupported for the algorithm; InvalidParameterError from validation.
    """Sparse coding.

    Each row of the result is the solution to a sparse coding problem.
    The goal is to find a sparse array `code` such that::

        X ~= code * dictionary

    Read more in the :ref:`User Guide <SparseCoder>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.

    dictionary : array-like of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows for meaningful
        output.

    gram : array-like of shape (n_components, n_components), default=None
        Precomputed Gram matrix, `dictionary * dictionary'`.

    cov : array-like of shape (n_components, n_samples), default=None
        Precomputed covariance, `dictionary' * X`.

    algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}, \
            default='lasso_lars'
        The algorithm used:

        * `'lars'`: uses the least angle regression method
          (`linear_model.lars_path`);
        * `'lasso_lars'`: uses Lars to compute the Lasso solution;
        * `'lasso_cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). lasso_lars will be faster if
          the estimated components are sparse;
        * `'omp'`: uses orthogonal matching pursuit to estimate the sparse
          solution;
        * `'threshold'`: squashes to zero all coefficients less than
          regularization from the projection `dictionary * data'`.

    n_nonzero_coefs : int, default=None
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
        and is overridden by `alpha` in the `omp` case. If `None`, then
        `n_nonzero_coefs=int(n_features / 10)`.

    alpha : float, default=None
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
        the reconstruction error targeted. In this case, it overrides
        `n_nonzero_coefs`.
        If `None`, default to 1.

    copy_cov : bool, default=True
        Whether to copy the precomputed covariance matrix; if `False`, it may
        be overwritten.

    init : ndarray of shape (n_samples, n_components), default=None
        Initialization value of the sparse codes. Only used if
        `algorithm='lasso_cd'`.

    max_iter : int, default=1000
        Maximum number of iterations to perform if `algorithm='lasso_cd'` or
        `'lasso_lars'`.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    check_input : bool, default=True
        If `False`, the input arrays X and dictionary will not be checked.

    verbose : int, default=0
        Controls the verbosity; the higher, the more messages.

    positive : bool, default=False
        Whether to enforce positivity when finding the encoding.

        .. versionadded:: 0.20

    Returns
    -------
    code : ndarray of shape (n_samples, n_components)
        The sparse codes.

    See Also
    --------
    sklearn.linear_model.lars_path : Compute Least Angle Regression or Lasso
        path using LARS algorithm.
    sklearn.linear_model.orthogonal_mp : Solves Orthogonal Matching Pursuit problems.
    sklearn.linear_model.Lasso : Train Linear Model with L1 prior as regularizer.
    SparseCoder : Find a sparse representation of data from a fixed precomputed
        dictionary.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.decomposition import sparse_encode
    >>> X = np.array([[-1, -1, -1], [0, 0, 3]])
    >>> dictionary = np.array(
    ...     [[0, 1, 0],
    ...      [-1, -1, 2],
    ...      [1, 1, 1],
    ...      [0, 1, 1],
    ...      [0, 2, 1]],
    ...    dtype=np.float64
    ... )
    >>> sparse_encode(X, dictionary, alpha=1e-10)
    array([[ 0.,  0., -1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.]])
    """
    if check_input:
        order = "C" if algorithm == "lasso_cd" else None
        dictionary = check_array(
            dictionary, order=order, dtype=[np.float64, np.float32]
        )
        X = check_array(X, order=order, dtype=[np.float64, np.float32])

    if dictionary.shape[1] != X.shape[1]:
        raise ValueError(
            "Dictionary and X have different numbers of features:"
            "dictionary.shape: {} X.shape{}".format(dictionary.shape, X.shape)
        )

    _check_positive_coding(algorithm, positive)

    return _sparse_encode(
        X,
        dictionary,
        gram=gram,
        cov=cov,
        algorithm=algorithm,
        n_nonzero_coefs=n_nonzero_coefs,
        alpha=alpha,
        copy_cov=copy_cov,
        init=init,
        max_iter=max_iter,
        n_jobs=n_jobs,
        verbose=verbose,
        positive=positive,
    )


def _sparse_encode(
    X,
    dictionary,
    *,
    gram=None,
    cov=None,
    algorithm="lasso_lars",
    n_nonzero_coefs=None,
    alpha=None,
    copy_cov=True,
    init=None,
    max_iter=1000,
    n_jobs=None,
    verbose=0,
    positive=False,
):
    # [FUNCTION DOC]
    # HELPER: Unvalidated sparse coding. Chooses regularization: for lars/omp it is
    # n_nonzero_coefs, defaulting to n_features/10 clipped to [1, n_components];
    # otherwise alpha, defaulting to 1.0. Computes gram (cast to X dtype) unless given
    # or the algorithm is threshold, and cov unless given or the algorithm is lasso_cd
    # (then copy_cov is turned off since cov is a private temporary). With one effective
    # job or threshold, runs _sparse_encode_precomputed on everything; else splits
    # samples into even slices (one per job), encodes each slice in parallel with joblib
    # (slicing cov columns and init rows), and assembles results into a new array.
    # Outputs: (n_samples, n_components) code.
    # KNOWN ISSUE: the parallel path allocates the result as float64 regardless of X
    # dtype, so float32 inputs are upcast only when n_jobs is greater than 1.
    """Sparse coding without input/parameter validation."""

    n_samples, n_features = X.shape
    n_components = dictionary.shape[0]

    if algorithm in ("lars", "omp"):
        regularization = n_nonzero_coefs
        if regularization is None:
            regularization = min(max(n_features / 10, 1), n_components)
    else:
        regularization = alpha
        if regularization is None:
            regularization = 1.0

    if gram is None and algorithm != "threshold":
        gram = np.dot(dictionary, dictionary.T).astype(X.dtype, copy=False)

    if cov is None and algorithm != "lasso_cd":
        copy_cov = False
        cov = np.dot(dictionary, X.T)

    if effective_n_jobs(n_jobs) == 1 or algorithm == "threshold":
        code = _sparse_encode_precomputed(
            X,
            dictionary,
            gram=gram,
            cov=cov,
            algorithm=algorithm,
            regularization=regularization,
            copy_cov=copy_cov,
            init=init,
            max_iter=max_iter,
            verbose=verbose,
            positive=positive,
        )
        return code

    # Enter parallel code block
    n_samples = X.shape[0]
    n_components = dictionary.shape[0]
    code = np.empty((n_samples, n_components))
    slices = list(gen_even_slices(n_samples, effective_n_jobs(n_jobs)))

    code_views = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_sparse_encode_precomputed)(
            X[this_slice],
            dictionary,
            gram=gram,
            cov=cov[:, this_slice] if cov is not None else None,
            algorithm=algorithm,
            regularization=regularization,
            copy_cov=copy_cov,
            init=init[this_slice] if init is not None else None,
            max_iter=max_iter,
            verbose=verbose,
            positive=positive,
        )
        for this_slice in slices
    )
    for this_slice, this_view in zip(slices, code_views):
        code[this_slice] = this_view
    return code


def _update_dict(
    dictionary,
    Y,
    code,
    A=None,
    B=None,
    verbose=False,
    random_state=None,
    positive=False,
):
    # [FUNCTION DOC]
    # HELPER: One block-coordinate-descent pass updating each dictionary atom in place.
    # With A equal to code transposed times code and B equal to Y transposed times code
    # (computed if not given; the online solver passes running statistics), for each
    # atom k: if A[k, k] exceeds 1e-6, add (column k of B minus row k of A times the
    # dictionary) divided by A[k, k]; otherwise the atom is unused, so replace it by a
    # random sample of Y plus Gaussian noise with standard deviation 1 percent of that
    # sample's std (or 0.01 if std is 0), and zero column k of code. Then clip to
    # nonnegative if positive and divide the atom by max(its norm, 1) to project onto
    # the unit ball.
    # Inputs: dictionary (n_components, n_features) writable; Y (n_samples, n_features);
    # code (n_samples, n_components); A, B optional; verbose; random_state; positive
    # bool.
    # Outputs: None.
    # Side effects: mutates dictionary and code in place; draws random numbers when
    # resampling; prints the number of resampled atoms if verbose.
    """Update the dense dictionary factor in place.

    Parameters
    ----------
    dictionary : ndarray of shape (n_components, n_features)
        Value of the dictionary at the previous iteration.

    Y : ndarray of shape (n_samples, n_features)
        Data matrix.

    code : ndarray of shape (n_samples, n_components)
        Sparse coding of the data against which to optimize the dictionary.

    A : ndarray of shape (n_components, n_components), default=None
        Together with `B`, sufficient stats of the online model to update the
        dictionary.

    B : ndarray of shape (n_features, n_components), default=None
        Together with `A`, sufficient stats of the online model to update the
        dictionary.

    verbose: bool, default=False
        Degree of output the procedure will print.

    random_state : int, RandomState instance or None, default=None
        Used for randomly initializing the dictionary. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    positive : bool, default=False
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20
    """
    n_samples, n_components = code.shape
    random_state = check_random_state(random_state)

    if A is None:
        A = code.T @ code
    if B is None:
        B = Y.T @ code

    n_unused = 0

    for k in range(n_components):
        if A[k, k] > 1e-6:
            # 1e-6 is arbitrary but consistent with the spams implementation
            dictionary[k] += (B[:, k] - A[k] @ dictionary) / A[k, k]
        else:
            # kth atom is almost never used -> sample a new one from the data
            newd = Y[random_state.choice(n_samples)]

            # add small noise to avoid making the sparse coding ill conditioned
            noise_level = 0.01 * (newd.std() or 1)  # avoid 0 std
            noise = random_state.normal(0, noise_level, size=len(newd))

            dictionary[k] = newd + noise
            code[:, k] = 0
            n_unused += 1

        if positive:
            np.clip(dictionary[k], 0, None, out=dictionary[k])

        # Projection on the constraint set ||V_k|| <= 1
        dictionary[k] /= max(linalg.norm(dictionary[k]), 1)

    if verbose and n_unused > 0:
        print(f"{n_unused} unused atoms resampled.")


def _dict_learning(
    X,
    n_components,
    *,
    alpha,
    max_iter,
    tol,
    method,
    n_jobs,
    dict_init,
    code_init,
    callback,
    verbose,
    random_state,
    return_n_iter,
    positive_dict,
    positive_code,
    method_max_iter,
):
    # [FUNCTION DOC]
    # HELPER: Batch dictionary learning by alternating minimization. Initialization: if
    # both code_init and dict_init are given, uses them (code copied in Fortran order);
    # otherwise takes the thin SVD of X with sign flipping for determinism, the left
    # factor as code and the singular values times the right factor as dictionary.
    # Truncates to n_components or pads code columns and dictionary rows with zeros, and
    # makes the dictionary Fortran-ordered. Then for up to max_iter iterations: encode X
    # with sparse_encode (algorithm method, warm-started from current code), update the
    # dictionary with _update_dict, compute cost (half squared residual plus alpha times
    # L1 norm of the code) and append it to errors; from the second iteration, stop when
    # the decrease is smaller than tol times the current cost. Every 5 iterations calls
    # callback with the local variables. Verbose 1 prints dots, higher prints per-
    # iteration timing and cost.
    # Inputs: X (n_samples, n_features); n_components int; alpha float; max_iter int;
    # tol float; method sparse_encode algorithm name; n_jobs; dict_init, code_init
    # arrays or None; callback callable or None; verbose; random_state RandomState;
    # return_n_iter bool; positive_dict, positive_code bool; method_max_iter int.
    # Outputs: (code, dictionary, errors) plus the number of iterations run if
    # return_n_iter (0 when max_iter is 0).
    # Side effects: prints to stdout when verbose; calls the callback.
    # KNOWN ISSUE: the comment on the truncation test says it holds for n_components
    # None, but comparing None to an int raises TypeError; callers always pass an int.
    # If only one of code_init / dict_init is provided it is silently ignored. When
    # dict_init is already Fortran-ordered and float64, it is not copied and is modified
    # in place by the updates.
    """Main dictionary learning algorithm"""
    t0 = time.time()
    # Init the code and the dictionary with SVD of Y
    if code_init is not None and dict_init is not None:
        code = np.array(code_init, order="F")
        # Don't copy V, it will happen below
        dictionary = dict_init
    else:
        code, S, dictionary = linalg.svd(X, full_matrices=False)
        # flip the initial code's sign to enforce deterministic output
        code, dictionary = svd_flip(code, dictionary)
        dictionary = S[:, np.newaxis] * dictionary
    r = len(dictionary)
    if n_components <= r:  # True even if n_components=None
        code = code[:, :n_components]
        dictionary = dictionary[:n_components, :]
    else:
        code = np.c_[code, np.zeros((len(code), n_components - r))]
        dictionary = np.r_[
            dictionary, np.zeros((n_components - r, dictionary.shape[1]))
        ]

    # Fortran-order dict better suited for the sparse coding which is the
    # bottleneck of this algorithm.
    dictionary = np.asfortranarray(dictionary)

    errors = []
    current_cost = np.nan

    if verbose == 1:
        print("[dict_learning]", end=" ")

    # If max_iter is 0, number of iterations returned should be zero
    ii = -1

    for ii in range(max_iter):
        dt = time.time() - t0
        if verbose == 1:
            sys.stdout.write(".")
            sys.stdout.flush()
        elif verbose:
            print(
                "Iteration % 3i (elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)"
                % (ii, dt, dt / 60, current_cost)
            )

        # Update code
        code = sparse_encode(
            X,
            dictionary,
            algorithm=method,
            alpha=alpha,
            init=code,
            n_jobs=n_jobs,
            positive=positive_code,
            max_iter=method_max_iter,
            verbose=verbose,
        )

        # Update dictionary in place
        _update_dict(
            dictionary,
            X,
            code,
            verbose=verbose,
            random_state=random_state,
            positive=positive_dict,
        )

        # Cost function
        current_cost = 0.5 * np.sum((X - code @ dictionary) ** 2) + alpha * np.sum(
            np.abs(code)
        )
        errors.append(current_cost)

        if ii > 0:
            dE = errors[-2] - errors[-1]
            # assert(dE >= -tol * errors[-1])
            if dE < tol * errors[-1]:
                if verbose == 1:
                    # A line return
                    print("")
                elif verbose:
                    print("--- Convergence reached after %d iterations" % ii)
                break
        if ii % 5 == 0 and callback is not None:
            callback(locals())

    if return_n_iter:
        return code, dictionary, errors, ii + 1
    else:
        return code, dictionary, errors


@validate_params(
    {
        "X": ["array-like"],
        "return_code": ["boolean"],
        "method": [StrOptions({"cd", "lars"})],
        "method_max_iter": [Interval(Integral, 0, None, closed="left")],
    },
    prefer_skip_nested_validation=False,
)
def dict_learning_online(
    X,
    n_components=2,
    *,
    alpha=1,
    max_iter=100,
    return_code=True,
    dict_init=None,
    callback=None,
    batch_size=256,
    verbose=False,
    shuffle=True,
    n_jobs=None,
    method="lars",
    random_state=None,
    positive_dict=False,
    positive_code=False,
    method_max_iter=1000,
    tol=1e-3,
    max_no_improvement=10,
):
    # [FUNCTION DOC]
    # Functional wrapper around MiniBatchDictionaryLearning: builds the estimator with
    # fit_algorithm=method, transform_algorithm "lasso_" plus method,
    # transform_alpha=alpha, transform_max_iter=method_max_iter and the remaining
    # options, and fits it on X.
    # Outputs: components_ (n_components, n_features) if not return_code, else (code
    # from transform(X), components_).
    # Exceptions: parameter validation errors from validate_params and the estimator.
    # Side effects: as the estimator (printing, callback, randomness).
    """Solve a dictionary learning matrix factorization problem online.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_Fro^2 + alpha * || U ||_1,1
                     (U,V)
                     with || V_k ||_2 = 1 for all  0 <= k < n_components

    where V is the dictionary and U is the sparse code. ||.||_Fro stands for
    the Frobenius norm and ||.||_1,1 stands for the entry-wise matrix norm
    which is the sum of the absolute values of all the entries in the matrix.
    This is accomplished by repeatedly iterating over mini-batches by slicing
    the input data.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.

    n_components : int or None, default=2
        Number of dictionary atoms to extract. If None, then ``n_components``
        is set to ``n_features``.

    alpha : float, default=1
        Sparsity controlling parameter.

    max_iter : int, default=100
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.

        .. versionadded:: 1.1

    return_code : bool, default=True
        Whether to also return the code U or just the dictionary `V`.

    dict_init : ndarray of shape (n_components, n_features), default=None
        Initial values for the dictionary for warm restart scenarios.
        If `None`, the initial values for the dictionary are created
        with an SVD decomposition of the data via
        :func:`~sklearn.utils.extmath.randomized_svd`.

    callback : callable, default=None
        A callable that gets invoked at the end of each iteration.

    batch_size : int, default=256
        The number of samples to take in each batch.

        .. versionchanged:: 1.3
           The default value of `batch_size` changed from 3 to 256 in version 1.3.

    verbose : bool, default=False
        To control the verbosity of the procedure.

    shuffle : bool, default=True
        Whether to shuffle the data before splitting it in batches.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    method : {'lars', 'cd'}, default='lars'
        * `'lars'`: uses the least angle regression method to solve the lasso
          problem (`linear_model.lars_path`);
        * `'cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). Lars will be faster if
          the estimated components are sparse.

    random_state : int, RandomState instance or None, default=None
        Used for initializing the dictionary when ``dict_init`` is not
        specified, randomly shuffling the data when ``shuffle`` is set to
        ``True``, and updating the dictionary. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    positive_dict : bool, default=False
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    positive_code : bool, default=False
        Whether to enforce positivity when finding the code.

        .. versionadded:: 0.20

    method_max_iter : int, default=1000
        Maximum number of iterations to perform when solving the lasso problem.

        .. versionadded:: 0.22

    tol : float, default=1e-3
        Control early stopping based on the norm of the differences in the
        dictionary between 2 steps.

        To disable early stopping based on changes in the dictionary, set
        `tol` to 0.0.

        .. versionadded:: 1.1

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini batches
        that does not yield an improvement on the smoothed cost function.

        To disable convergence detection based on cost function, set
        `max_no_improvement` to None.

        .. versionadded:: 1.1

    Returns
    -------
    code : ndarray of shape (n_samples, n_components),
        The sparse code (only returned if `return_code=True`).

    dictionary : ndarray of shape (n_components, n_features),
        The solutions to the dictionary learning problem.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to `True`.

    See Also
    --------
    dict_learning : Solve a dictionary learning matrix factorization problem.
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    MiniBatchDictionaryLearning : A faster, less accurate, version of the dictionary
        learning algorithm.
    SparsePCA : Sparse Principal Components Analysis.
    MiniBatchSparsePCA : Mini-batch Sparse Principal Components Analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_sparse_coded_signal
    >>> from sklearn.decomposition import dict_learning_online
    >>> X, _, _ = make_sparse_coded_signal(
    ...     n_samples=30, n_components=15, n_features=20, n_nonzero_coefs=10,
    ...     random_state=42,
    ... )
    >>> U, V = dict_learning_online(
    ...     X, n_components=15, alpha=0.2, max_iter=20, batch_size=3, random_state=42
    ... )

    We can check the level of sparsity of `U`:

    >>> np.mean(U == 0)
    np.float64(0.53)

    We can compare the average squared euclidean norm of the reconstruction
    error of the sparse coded signal relative to the squared euclidean norm of
    the original signal:

    >>> X_hat = U @ V
    >>> np.mean(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X ** 2, axis=1))
    np.float64(0.053)
    """
    transform_algorithm = "lasso_" + method

    est = MiniBatchDictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        max_iter=max_iter,
        n_jobs=n_jobs,
        fit_algorithm=method,
        batch_size=batch_size,
        shuffle=shuffle,
        dict_init=dict_init,
        random_state=random_state,
        transform_algorithm=transform_algorithm,
        transform_alpha=alpha,
        positive_code=positive_code,
        positive_dict=positive_dict,
        transform_max_iter=method_max_iter,
        verbose=verbose,
        callback=callback,
        tol=tol,
        max_no_improvement=max_no_improvement,
    ).fit(X)

    if not return_code:
        return est.components_
    else:
        code = est.transform(X)
        return code, est.components_


@validate_params(
    {
        "X": ["array-like"],
        "method": [StrOptions({"lars", "cd"})],
        "return_n_iter": ["boolean"],
        "method_max_iter": [Interval(Integral, 0, None, closed="left")],
    },
    prefer_skip_nested_validation=False,
)
def dict_learning(
    X,
    n_components,
    *,
    alpha,
    max_iter=100,
    tol=1e-8,
    method="lars",
    n_jobs=None,
    dict_init=None,
    code_init=None,
    callback=None,
    verbose=False,
    random_state=None,
    return_n_iter=False,
    positive_dict=False,
    positive_code=False,
    method_max_iter=1000,
):
    # [FUNCTION DOC]
    # Functional wrapper around DictionaryLearning: builds it with fit_algorithm=method,
    # transform_max_iter=method_max_iter and the given options, forces default (numpy)
    # output, and calls fit_transform.
    # Outputs: (code, components_, error_) and also n_iter_ if return_n_iter.
    # Exceptions: validation errors.
    """Solve a dictionary learning matrix factorization problem.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_Fro^2 + alpha * || U ||_1,1
                     (U,V)
                    with || V_k ||_2 = 1 for all  0 <= k < n_components

    where V is the dictionary and U is the sparse code. ||.||_Fro stands for
    the Frobenius norm and ||.||_1,1 stands for the entry-wise matrix norm
    which is the sum of the absolute values of all the entries in the matrix.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.

    n_components : int
        Number of dictionary atoms to extract.

    alpha : int or float
        Sparsity controlling parameter.

    max_iter : int, default=100
        Maximum number of iterations to perform.

    tol : float, default=1e-8
        Tolerance for the stopping condition.

    method : {'lars', 'cd'}, default='lars'
        The method used:

        * `'lars'`: uses the least angle regression method to solve the lasso
           problem (`linear_model.lars_path`);
        * `'cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). Lars will be faster if
          the estimated components are sparse.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    dict_init : ndarray of shape (n_components, n_features), default=None
        Initial value for the dictionary for warm restart scenarios. Only used
        if `code_init` and `dict_init` are not None.

    code_init : ndarray of shape (n_samples, n_components), default=None
        Initial value for the sparse code for warm restart scenarios. Only used
        if `code_init` and `dict_init` are not None.

    callback : callable, default=None
        Callable that gets invoked every five iterations.

    verbose : bool, default=False
        To control the verbosity of the procedure.

    random_state : int, RandomState instance or None, default=None
        Used for randomly initializing the dictionary. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    positive_dict : bool, default=False
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    positive_code : bool, default=False
        Whether to enforce positivity when finding the code.

        .. versionadded:: 0.20

    method_max_iter : int, default=1000
        Maximum number of iterations to perform.

        .. versionadded:: 0.22

    Returns
    -------
    code : ndarray of shape (n_samples, n_components)
        The sparse code factor in the matrix factorization.

    dictionary : ndarray of shape (n_components, n_features),
        The dictionary factor in the matrix factorization.

    errors : array
        Vector of errors at each iteration.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to True.

    See Also
    --------
    dict_learning_online : Solve a dictionary learning matrix factorization
        problem online.
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    MiniBatchDictionaryLearning : A faster, less accurate version
        of the dictionary learning algorithm.
    SparsePCA : Sparse Principal Components Analysis.
    MiniBatchSparsePCA : Mini-batch Sparse Principal Components Analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_sparse_coded_signal
    >>> from sklearn.decomposition import dict_learning
    >>> X, _, _ = make_sparse_coded_signal(
    ...     n_samples=30, n_components=15, n_features=20, n_nonzero_coefs=10,
    ...     random_state=42,
    ... )
    >>> U, V, errors = dict_learning(X, n_components=15, alpha=0.1, random_state=42)

    We can check the level of sparsity of `U`:

    >>> np.mean(U == 0)
    np.float64(0.62)

    We can compare the average squared euclidean norm of the reconstruction
    error of the sparse coded signal relative to the squared euclidean norm of
    the original signal:

    >>> X_hat = U @ V
    >>> np.mean(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X ** 2, axis=1))
    np.float64(0.0192)
    """
    estimator = DictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        fit_algorithm=method,
        n_jobs=n_jobs,
        dict_init=dict_init,
        callback=callback,
        code_init=code_init,
        verbose=verbose,
        random_state=random_state,
        positive_code=positive_code,
        positive_dict=positive_dict,
        transform_max_iter=method_max_iter,
    ).set_output(transform="default")
    code = estimator.fit_transform(X)
    if return_n_iter:
        return (
            code,
            estimator.components_,
            estimator.error_,
            estimator.n_iter_,
        )
    return code, estimator.components_, estimator.error_


class _BaseSparseCoding(ClassNamePrefixFeaturesOutMixin, TransformerMixin):
    # [CLASS DOC]
    # HELPER: Mixin with shared transform logic for SparseCoder and the dictionary
    # learning estimators. Stores the transform_* options, split_sign, n_jobs and
    # positive_code. Subclasses provide the dictionary (components_ or the dictionary
    # parameter).
    """Base class from SparseCoder and DictionaryLearning algorithms."""

    def __init__(
        self,
        transform_algorithm,
        transform_n_nonzero_coefs,
        transform_alpha,
        split_sign,
        n_jobs,
        positive_code,
        transform_max_iter,
    ):
        # [FUNCTION DOC]
        # HELPER: Stores the constructor arguments unchanged as attributes (estimator
        # API).
        self.transform_algorithm = transform_algorithm
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.transform_alpha = transform_alpha
        self.transform_max_iter = transform_max_iter
        self.split_sign = split_sign
        self.n_jobs = n_jobs
        self.positive_code = positive_code

    def _transform(self, X, dictionary):
        # [FUNCTION DOC]
        # HELPER: Validates X against the fitted feature count, picks the transform
        # alpha (self.alpha when the estimator has one and transform_alpha is None, else
        # transform_alpha), and encodes X with sparse_encode using the transform_*
        # options. If split_sign, returns a code with twice the columns: positive parts
        # first, then the magnitudes of the negative parts.
        # Inputs: X (n_samples, n_features); dictionary (n_components, n_features).
        # Outputs: (n_samples, n_components) or (n_samples, 2 * n_components) array.
        # KNOWN ISSUE: the split_sign array is allocated as float64, so float32 inputs
        # are upcast although the estimators declare float32 is preserved.
        """Private method allowing to accommodate both DictionaryLearning and
        SparseCoder."""
        X = validate_data(self, X, reset=False)

        if hasattr(self, "alpha") and self.transform_alpha is None:
            transform_alpha = self.alpha
        else:
            transform_alpha = self.transform_alpha

        code = sparse_encode(
            X,
            dictionary,
            algorithm=self.transform_algorithm,
            n_nonzero_coefs=self.transform_n_nonzero_coefs,
            alpha=transform_alpha,
            max_iter=self.transform_max_iter,
            n_jobs=self.n_jobs,
            positive=self.positive_code,
        )

        if self.split_sign:
            # feature vector is split into a positive and negative side
            n_samples, n_features = code.shape
            split_code = np.empty((n_samples, 2 * n_features))
            split_code[:, :n_features] = np.maximum(code, 0)
            split_code[:, n_features:] = -np.minimum(code, 0)
            code = split_code

        return code

    def transform(self, X):
        # [FUNCTION DOC]
        # Checks the estimator is fitted and encodes X with components_.
        # Exceptions: NotFittedError.
        """Encode the data as a sparse combination of the dictionary atoms.

        Coding method is determined by the object parameter
        `transform_algorithm`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data to be transformed, must have the same number of
            features as the data used to train the model.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        return self._transform(X, self.components_)

    def _inverse_transform(self, code, dictionary):
        # [FUNCTION DOC]
        # HELPER: Validates code, checks its column count equals the number of atoms
        # (doubled if split_sign), recombines split codes as positive part minus
        # negative part, and returns code times dictionary.
        # Outputs: (n_samples, n_features) reconstruction.
        # Exceptions: ValueError on column count mismatch.
        """Private method allowing to accommodate both DictionaryLearning and
        SparseCoder."""
        code = check_array(code)
        # compute number of expected features in code
        expected_n_components = dictionary.shape[0]
        if self.split_sign:
            expected_n_components += expected_n_components
        if not code.shape[1] == expected_n_components:
            raise ValueError(
                "The number of components in the code is different from the "
                "number of components in the dictionary."
                f"Expected {expected_n_components}, got {code.shape[1]}."
            )
        if self.split_sign:
            n_samples, n_features = code.shape
            n_features //= 2
            code = code[:, :n_features] - code[:, n_features:]

        return code @ dictionary

    def inverse_transform(self, X):
        # [FUNCTION DOC]
        # Checks fitted state and reconstructs X from codes using components_.
        """Transform data back to its original space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Data to be transformed back. Must have the same number of
            components as the data used to train the model.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)
        return self._inverse_transform(X, self.components_)


class SparseCoder(_BaseSparseCoding, BaseEstimator):
    # [CLASS DOC]
    # Estimator that sparse-codes data against a fixed, user-provided dictionary. Needs
    # no fitting (requires_fit tag False); fit only validates. Default
    # transform_algorithm is omp.
    """Sparse coding.

    Finds a sparse representation of data against a fixed, precomputed
    dictionary.

    Each row of the result is the solution to a sparse coding problem.
    The goal is to find a sparse array `code` such that::

        X ~= code * dictionary

    Read more in the :ref:`User Guide <SparseCoder>`.

    Parameters
    ----------
    dictionary : ndarray of shape (n_components, n_features)
        The dictionary atoms used for sparse coding. Lines are assumed to be
        normalized to unit norm.

    transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', \
            'threshold'}, default='omp'
        Algorithm used to transform the data:

        - `'lars'`: uses the least angle regression method
          (`linear_model.lars_path`);
        - `'lasso_lars'`: uses Lars to compute the Lasso solution;
        - `'lasso_cd'`: uses the coordinate descent method to compute the
          Lasso solution (linear_model.Lasso). `'lasso_lars'` will be faster if
          the estimated components are sparse;
        - `'omp'`: uses orthogonal matching pursuit to estimate the sparse
          solution;
        - `'threshold'`: squashes to zero all coefficients less than alpha from
          the projection ``dictionary * X'``.

    transform_n_nonzero_coefs : int, default=None
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
        and is overridden by `alpha` in the `omp` case. If `None`, then
        `transform_n_nonzero_coefs=int(n_features / 10)`.

    transform_alpha : float, default=None
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
        the reconstruction error targeted. In this case, it overrides
        `n_nonzero_coefs`.
        If `None`, default to 1.

    split_sign : bool, default=False
        Whether to split the sparse feature vector into the concatenation of
        its negative part and its positive part. This can improve the
        performance of downstream classifiers.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    positive_code : bool, default=False
        Whether to enforce positivity when finding the code.

        .. versionadded:: 0.20

    transform_max_iter : int, default=1000
        Maximum number of iterations to perform if `algorithm='lasso_cd'` or
        `lasso_lars`.

        .. versionadded:: 0.22

    Attributes
    ----------
    n_components_ : int
        Number of atoms.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    MiniBatchDictionaryLearning : A faster, less accurate, version of the
        dictionary learning algorithm.
    MiniBatchSparsePCA : Mini-batch Sparse Principal Components Analysis.
    SparsePCA : Sparse Principal Components Analysis.
    sparse_encode : Sparse coding where each row of the result is the solution
        to a sparse coding problem.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.decomposition import SparseCoder
    >>> X = np.array([[-1, -1, -1], [0, 0, 3]])
    >>> dictionary = np.array(
    ...     [[0, 1, 0],
    ...      [-1, -1, 2],
    ...      [1, 1, 1],
    ...      [0, 1, 1],
    ...      [0, 2, 1]],
    ...    dtype=np.float64
    ... )
    >>> coder = SparseCoder(
    ...     dictionary=dictionary, transform_algorithm='lasso_lars',
    ...     transform_alpha=1e-10,
    ... )
    >>> coder.transform(X)
    array([[ 0.,  0., -1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.]])
    """

    _parameter_constraints: dict = {
        "dictionary": ["array-like"],
        "transform_algorithm": [
            StrOptions({"lasso_lars", "lasso_cd", "lars", "omp", "threshold"})
        ],
        "transform_n_nonzero_coefs": [Interval(Integral, 1, None, closed="left"), None],
        "transform_alpha": [Interval(Real, 0, None, closed="left"), None],
        "split_sign": ["boolean"],
        "n_jobs": [Integral, None],
        "positive_code": ["boolean"],
        "transform_max_iter": [Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
        self,
        dictionary,
        *,
        transform_algorithm="omp",
        transform_n_nonzero_coefs=None,
        transform_alpha=None,
        split_sign=False,
        n_jobs=None,
        positive_code=False,
        transform_max_iter=1000,
    ):
        # [FUNCTION DOC]
        # Stores dictionary (array (n_components, n_features)) and the transform options
        # via the base initializer.
        super().__init__(
            transform_algorithm,
            transform_n_nonzero_coefs,
            transform_alpha,
            split_sign,
            n_jobs,
            positive_code,
            transform_max_iter,
        )
        self.dictionary = dictionary

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Validates X, sets n_features_in_ and n_components_ (rows of the dictionary).
        # Outputs: self.
        # Exceptions: ValueError if X and dictionary feature counts differ.
        # Side effects: sets n_components_ and validation attributes.
        """Only validate the parameters of the estimator.

        This method allows to: (i) validate the parameters of the estimator and
        (ii) be consistent with the scikit-learn transformer API.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Only used for input validation.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = validate_data(self, X)
        self.n_components_ = self.dictionary.shape[0]
        if X.shape[1] != self.dictionary.shape[1]:
            raise ValueError(
                "Dictionary and X have different numbers of features:"
                f"dictionary.shape: {self.dictionary.shape} X.shape: {X.shape}"
            )
        return self

    def transform(self, X, y=None):
        # [FUNCTION DOC]
        # Encodes X with the stored dictionary via _transform (y ignored). Works without
        # fit.
        """Encode the data as a sparse combination of the dictionary atoms.

        Coding method is determined by the object parameter
        `transform_algorithm`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        return super()._transform(X, self.dictionary)

    def inverse_transform(self, X):
        # [FUNCTION DOC]
        # Reconstructs data from codes using the stored dictionary; works without fit.
        """Transform data back to its original space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Data to be transformed back. Must have the same number of
            components as the data used to train the model.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        return self._inverse_transform(X, self.dictionary)

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Marks the estimator as not requiring fit and as preserving float64/float32.
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags

    @property
    def _n_features_out(self):
        # [FUNCTION DOC]
        # HELPER: Number of output features, n_components_ (set by fit).
        # KNOWN ISSUE: ignores split_sign (actual output has twice as many columns) and
        # fails before fit is called although transform works without fit.
        """Number of transformed output features."""
        return self.n_components_


class DictionaryLearning(_BaseSparseCoding, BaseEstimator):
    # [CLASS DOC]
    # Batch dictionary learning estimator; fitting runs _dict_learning on the whole data
    # set. Fitted attributes: components_, error_, n_iter_ (plus n_features_in_).
    """Dictionary learning.

    Finds a dictionary (a set of atoms) that performs well at sparsely
    encoding the fitted data.

    Solves the optimization problem::

        (U^*,V^*) = argmin 0.5 || X - U V ||_Fro^2 + alpha * || U ||_1,1
                    (U,V)
                    with || V_k ||_2 <= 1 for all  0 <= k < n_components

    ||.||_Fro stands for the Frobenius norm and ||.||_1,1 stands for
    the entry-wise matrix norm which is the sum of the absolute values
    of all the entries in the matrix.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    n_components : int, default=None
        Number of dictionary elements to extract. If None, then ``n_components``
        is set to ``n_features``.

    alpha : float, default=1.0
        Sparsity controlling parameter.

    max_iter : int, default=1000
        Maximum number of iterations to perform.

    tol : float, default=1e-8
        Tolerance for numerical error.

    fit_algorithm : {'lars', 'cd'}, default='lars'
        * `'lars'`: uses the least angle regression method to solve the lasso
          problem (:func:`~sklearn.linear_model.lars_path`);
        * `'cd'`: uses the coordinate descent method to compute the
          Lasso solution (:class:`~sklearn.linear_model.Lasso`). Lars will be
          faster if the estimated components are sparse.

        .. versionadded:: 0.17
           *cd* coordinate descent method to improve speed.

    transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', \
            'threshold'}, default='omp'
        Algorithm used to transform the data:

        - `'lars'`: uses the least angle regression method
          (:func:`~sklearn.linear_model.lars_path`);
        - `'lasso_lars'`: uses Lars to compute the Lasso solution.
        - `'lasso_cd'`: uses the coordinate descent method to compute the
          Lasso solution (:class:`~sklearn.linear_model.Lasso`). `'lasso_lars'`
          will be faster if the estimated components are sparse.
        - `'omp'`: uses orthogonal matching pursuit to estimate the sparse
          solution.
        - `'threshold'`: squashes to zero all coefficients less than alpha from
          the projection ``dictionary * X'``.

        .. versionadded:: 0.17
           *lasso_cd* coordinate descent method to improve speed.

    transform_n_nonzero_coefs : int, default=None
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and
        `algorithm='omp'`. If `None`, then
        `transform_n_nonzero_coefs=int(n_features / 10)`.

    transform_alpha : float, default=None
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `None`, defaults to `alpha`.

        .. versionchanged:: 1.2
            When None, default value changed from 1.0 to `alpha`.

    n_jobs : int or None, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    code_init : ndarray of shape (n_samples, n_components), default=None
        Initial value for the code, for warm restart. Only used if `code_init`
        and `dict_init` are not None.

    dict_init : ndarray of shape (n_components, n_features), default=None
        Initial values for the dictionary, for warm restart. Only used if
        `code_init` and `dict_init` are not None.

    callback : callable, default=None
        Callable that gets invoked every five iterations.

        .. versionadded:: 1.3

    verbose : bool, default=False
        To control the verbosity of the procedure.

    split_sign : bool, default=False
        Whether to split the sparse feature vector into the concatenation of
        its negative part and its positive part. This can improve the
        performance of downstream classifiers.

    random_state : int, RandomState instance or None, default=None
        Used for initializing the dictionary when ``dict_init`` is not
        specified, randomly shuffling the data when ``shuffle`` is set to
        ``True``, and updating the dictionary. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    positive_code : bool, default=False
        Whether to enforce positivity when finding the code.

        .. versionadded:: 0.20

    positive_dict : bool, default=False
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    transform_max_iter : int, default=1000
        Maximum number of iterations to perform if `algorithm='lasso_cd'` or
        `'lasso_lars'`.

        .. versionadded:: 0.22

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        dictionary atoms extracted from the data

    error_ : array
        vector of errors at each iteration

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Number of iterations run.

    See Also
    --------
    MiniBatchDictionaryLearning: A faster, less accurate, version of the
        dictionary learning algorithm.
    MiniBatchSparsePCA : Mini-batch Sparse Principal Components Analysis.
    SparseCoder : Find a sparse representation of data from a fixed,
        precomputed dictionary.
    SparsePCA : Sparse Principal Components Analysis.

    References
    ----------

    J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009: Online dictionary learning
    for sparse coding (https://www.di.ens.fr/~fbach/mairal_icml09.pdf)

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_sparse_coded_signal
    >>> from sklearn.decomposition import DictionaryLearning
    >>> X, dictionary, code = make_sparse_coded_signal(
    ...     n_samples=30, n_components=15, n_features=20, n_nonzero_coefs=10,
    ...     random_state=42,
    ... )
    >>> dict_learner = DictionaryLearning(
    ...     n_components=15, transform_algorithm='lasso_lars', transform_alpha=0.1,
    ...     random_state=42,
    ... )
    >>> X_transformed = dict_learner.fit(X).transform(X)

    We can check the level of sparsity of `X_transformed`:

    >>> np.mean(X_transformed == 0)
    np.float64(0.527)

    We can compare the average squared euclidean norm of the reconstruction
    error of the sparse coded signal relative to the squared euclidean norm of
    the original signal:

    >>> X_hat = X_transformed @ dict_learner.components_
    >>> np.mean(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X ** 2, axis=1))
    np.float64(0.056)
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left"), None],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "fit_algorithm": [StrOptions({"lars", "cd"})],
        "transform_algorithm": [
            StrOptions({"lasso_lars", "lasso_cd", "lars", "omp", "threshold"})
        ],
        "transform_n_nonzero_coefs": [Interval(Integral, 1, None, closed="left"), None],
        "transform_alpha": [Interval(Real, 0, None, closed="left"), None],
        "n_jobs": [Integral, None],
        "code_init": [np.ndarray, None],
        "dict_init": [np.ndarray, None],
        "callback": [callable, None],
        "verbose": ["verbose"],
        "split_sign": ["boolean"],
        "random_state": ["random_state"],
        "positive_code": ["boolean"],
        "positive_dict": ["boolean"],
        "transform_max_iter": [Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
        self,
        n_components=None,
        *,
        alpha=1,
        max_iter=1000,
        tol=1e-8,
        fit_algorithm="lars",
        transform_algorithm="omp",
        transform_n_nonzero_coefs=None,
        transform_alpha=None,
        n_jobs=None,
        code_init=None,
        dict_init=None,
        callback=None,
        verbose=False,
        split_sign=False,
        random_state=None,
        positive_code=False,
        positive_dict=False,
        transform_max_iter=1000,
    ):
        # [FUNCTION DOC]
        # Stores all hyperparameters unchanged (see class docstring); transform options
        # go to the base class.
        super().__init__(
            transform_algorithm,
            transform_n_nonzero_coefs,
            transform_alpha,
            split_sign,
            n_jobs,
            positive_code,
            transform_max_iter,
        )
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_algorithm = fit_algorithm
        self.code_init = code_init
        self.dict_init = dict_init
        self.callback = callback
        self.verbose = verbose
        self.random_state = random_state
        self.positive_dict = positive_dict

    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Fits via fit_transform and discards the code. Outputs: self.
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.fit_transform(X)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        # [FUNCTION DOC]
        # Checks positive coding support for fit_algorithm, uses "lasso_" plus
        # fit_algorithm as coding method, validates X, defaults n_components to
        # n_features, and runs _dict_learning with return_n_iter.
        # Outputs: code (n_samples, n_components) of the final iteration.
        # Exceptions: ValueError from _check_positive_coding.
        # Side effects: sets components_, error_ (cost per iteration), n_iter_.
        # KNOWN ISSUE: the positivity check is done on fit_algorithm ("lars"/"cd")
        # rather than on the actual solver "lasso_lars", so positive_code with
        # fit_algorithm "lars" is rejected although Lasso-LARS supports it.
        """Fit the model from data in X and return the transformed data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        V : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        _check_positive_coding(method=self.fit_algorithm, positive=self.positive_code)

        method = "lasso_" + self.fit_algorithm

        random_state = check_random_state(self.random_state)
        X = validate_data(self, X)

        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        V, U, E, self.n_iter_ = _dict_learning(
            X,
            n_components,
            alpha=self.alpha,
            tol=self.tol,
            max_iter=self.max_iter,
            method=method,
            method_max_iter=self.transform_max_iter,
            n_jobs=self.n_jobs,
            code_init=self.code_init,
            dict_init=self.dict_init,
            callback=self.callback,
            verbose=self.verbose,
            random_state=random_state,
            return_n_iter=True,
            positive_dict=self.positive_dict,
            positive_code=self.positive_code,
        )
        self.components_ = U
        self.error_ = E

        return V

    @property
    def _n_features_out(self):
        # [FUNCTION DOC]
        # HELPER: Number of output features (rows of components_). Ignores split_sign.
        """Number of transformed output features."""
        return self.components_.shape[0]

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Declares float64/float32 dtype preservation.
        tags = super().__sklearn_tags__()
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags


class MiniBatchDictionaryLearning(_BaseSparseCoding, BaseEstimator):
    # [CLASS DOC]
    # Online (mini-batch) dictionary learning following Mairal et al.: codes each batch,
    # keeps exponentially down-weighted sufficient statistics _A (code Gram) and _B
    # (data-code cross products) and updates atoms by block coordinate descent with
    # them. Supports partial_fit. Fitted attributes: components_, n_iter_, n_steps_ plus
    # private state.
    """Mini-batch dictionary learning.

    Finds a dictionary (a set of atoms) that performs well at sparsely
    encoding the fitted data.

    Solves the optimization problem::

       (U^*,V^*) = argmin 0.5 || X - U V ||_Fro^2 + alpha * || U ||_1,1
                    (U,V)
                    with || V_k ||_2 <= 1 for all  0 <= k < n_components

    ||.||_Fro stands for the Frobenius norm and ||.||_1,1 stands for
    the entry-wise matrix norm which is the sum of the absolute values
    of all the entries in the matrix.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    n_components : int, default=None
        Number of dictionary elements to extract.

    alpha : float, default=1
        Sparsity controlling parameter.

    max_iter : int, default=1_000
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.

        .. versionadded:: 1.1

    fit_algorithm : {'lars', 'cd'}, default='lars'
        The algorithm used:

        - `'lars'`: uses the least angle regression method to solve the lasso
          problem (`linear_model.lars_path`)
        - `'cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). Lars will be faster if
          the estimated components are sparse.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    batch_size : int, default=256
        Number of samples in each mini-batch.

        .. versionchanged:: 1.3
           The default value of `batch_size` changed from 3 to 256 in version 1.3.

    shuffle : bool, default=True
        Whether to shuffle the samples before forming batches.

    dict_init : ndarray of shape (n_components, n_features), default=None
        Initial value of the dictionary for warm restart scenarios.

    transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', \
            'threshold'}, default='omp'
        Algorithm used to transform the data:

        - `'lars'`: uses the least angle regression method
          (`linear_model.lars_path`);
        - `'lasso_lars'`: uses Lars to compute the Lasso solution.
        - `'lasso_cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). `'lasso_lars'` will be faster
          if the estimated components are sparse.
        - `'omp'`: uses orthogonal matching pursuit to estimate the sparse
          solution.
        - `'threshold'`: squashes to zero all coefficients less than alpha from
          the projection ``dictionary * X'``.

    transform_n_nonzero_coefs : int, default=None
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and
        `algorithm='omp'`. If `None`, then
        `transform_n_nonzero_coefs=int(n_features / 10)`.

    transform_alpha : float, default=None
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `None`, defaults to `alpha`.

        .. versionchanged:: 1.2
            When None, default value changed from 1.0 to `alpha`.

    verbose : bool or int, default=False
        To control the verbosity of the procedure.

    split_sign : bool, default=False
        Whether to split the sparse feature vector into the concatenation of
        its negative part and its positive part. This can improve the
        performance of downstream classifiers.

    random_state : int, RandomState instance or None, default=None
        Used for initializing the dictionary when ``dict_init`` is not
        specified, randomly shuffling the data when ``shuffle`` is set to
        ``True``, and updating the dictionary. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    positive_code : bool, default=False
        Whether to enforce positivity when finding the code.

        .. versionadded:: 0.20

    positive_dict : bool, default=False
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    transform_max_iter : int, default=1000
        Maximum number of iterations to perform if `algorithm='lasso_cd'` or
        `'lasso_lars'`.

        .. versionadded:: 0.22

    callback : callable, default=None
        A callable that gets invoked at the end of each iteration.

        .. versionadded:: 1.1

    tol : float, default=1e-3
        Control early stopping based on the norm of the differences in the
        dictionary between 2 steps.

        To disable early stopping based on changes in the dictionary, set
        `tol` to 0.0.

        .. versionadded:: 1.1

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini batches
        that does not yield an improvement on the smoothed cost function.

        To disable convergence detection based on cost function, set
        `max_no_improvement` to None.

        .. versionadded:: 1.1

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Components extracted from the data.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Number of iterations over the full dataset.

    n_steps_ : int
        Number of mini-batches processed.

        .. versionadded:: 1.1

    See Also
    --------
    DictionaryLearning : Find a dictionary that sparsely encodes data.
    MiniBatchSparsePCA : Mini-batch Sparse Principal Components Analysis.
    SparseCoder : Find a sparse representation of data from a fixed,
        precomputed dictionary.
    SparsePCA : Sparse Principal Components Analysis.

    References
    ----------

    J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009: Online dictionary learning
    for sparse coding (https://www.di.ens.fr/~fbach/mairal_icml09.pdf)

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_sparse_coded_signal
    >>> from sklearn.decomposition import MiniBatchDictionaryLearning
    >>> X, dictionary, code = make_sparse_coded_signal(
    ...     n_samples=30, n_components=15, n_features=20, n_nonzero_coefs=10,
    ...     random_state=42)
    >>> dict_learner = MiniBatchDictionaryLearning(
    ...     n_components=15, batch_size=3, transform_algorithm='lasso_lars',
    ...     transform_alpha=0.1, max_iter=20, random_state=42)
    >>> X_transformed = dict_learner.fit_transform(X)

    We can check the level of sparsity of `X_transformed`:

    >>> np.mean(X_transformed == 0) > 0.5
    np.True_

    We can compare the average squared euclidean norm of the reconstruction
    error of the sparse coded signal relative to the squared euclidean norm of
    the original signal:

    >>> X_hat = X_transformed @ dict_learner.components_
    >>> np.mean(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X ** 2, axis=1))
    np.float64(0.052)

    For a more detailed example, see
    :ref:`sphx_glr_auto_examples_decomposition_plot_image_denoising.py`
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left"), None],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "fit_algorithm": [StrOptions({"cd", "lars"})],
        "n_jobs": [None, Integral],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "shuffle": ["boolean"],
        "dict_init": [None, np.ndarray],
        "transform_algorithm": [
            StrOptions({"lasso_lars", "lasso_cd", "lars", "omp", "threshold"})
        ],
        "transform_n_nonzero_coefs": [Interval(Integral, 1, None, closed="left"), None],
        "transform_alpha": [Interval(Real, 0, None, closed="left"), None],
        "verbose": ["verbose"],
        "split_sign": ["boolean"],
        "random_state": ["random_state"],
        "positive_code": ["boolean"],
        "positive_dict": ["boolean"],
        "transform_max_iter": [Interval(Integral, 0, None, closed="left")],
        "callback": [None, callable],
        "tol": [Interval(Real, 0, None, closed="left")],
        "max_no_improvement": [Interval(Integral, 0, None, closed="left"), None],
    }

    def __init__(
        self,
        n_components=None,
        *,
        alpha=1,
        max_iter=1_000,
        fit_algorithm="lars",
        n_jobs=None,
        batch_size=256,
        shuffle=True,
        dict_init=None,
        transform_algorithm="omp",
        transform_n_nonzero_coefs=None,
        transform_alpha=None,
        verbose=False,
        split_sign=False,
        random_state=None,
        positive_code=False,
        positive_dict=False,
        transform_max_iter=1000,
        callback=None,
        tol=1e-3,
        max_no_improvement=10,
    ):
        # [FUNCTION DOC]
        # Stores all hyperparameters unchanged.
        super().__init__(
            transform_algorithm,
            transform_n_nonzero_coefs,
            transform_alpha,
            split_sign,
            n_jobs,
            positive_code,
            transform_max_iter,
        )
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.fit_algorithm = fit_algorithm
        self.dict_init = dict_init
        self.verbose = verbose
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.split_sign = split_sign
        self.random_state = random_state
        self.positive_dict = positive_dict
        self.callback = callback
        self.max_no_improvement = max_no_improvement
        self.tol = tol

    def _check_params(self, X):
        # [FUNCTION DOC]
        # HELPER: Derives private settings from X: _n_components (n_features if None),
        # _fit_algorithm ("lasso_" plus fit_algorithm, after the positive coding check)
        # and _batch_size (capped at n_samples).
        # Exceptions: ValueError from _check_positive_coding.
        # Side effects: sets those private attributes.
        # n_components
        self._n_components = self.n_components
        if self._n_components is None:
            self._n_components = X.shape[1]

        # fit_algorithm
        _check_positive_coding(self.fit_algorithm, self.positive_code)
        self._fit_algorithm = "lasso_" + self.fit_algorithm

        # batch_size
        self._batch_size = min(self.batch_size, X.shape[0])

    def _initialize_dict(self, X, random_state):
        # [FUNCTION DOC]
        # HELPER: Uses dict_init if given, else a randomized SVD of X with _n_components
        # components (singular values times right vectors). Truncates or zero-pads rows
        # to _n_components, converts to Fortran order and X dtype and makes it writable.
        # Outputs: (n_components, n_features) dictionary.
        # Side effects: random draws from random_state.
        # KNOWN ISSUE: check_array is called with copy False, so a dict_init that
        # already has the right dtype, order and size is used as is and later modified
        # in place by fitting.
        """Initialization of the dictionary."""
        if self.dict_init is not None:
            dictionary = self.dict_init
        else:
            # Init V with SVD of X
            _, S, dictionary = _randomized_svd(
                X, self._n_components, random_state=random_state
            )
            dictionary = S[:, np.newaxis] * dictionary

        if self._n_components <= len(dictionary):
            dictionary = dictionary[: self._n_components, :]
        else:
            dictionary = np.concatenate(
                (
                    dictionary,
                    np.zeros(
                        (self._n_components - len(dictionary), dictionary.shape[1]),
                        dtype=dictionary.dtype,
                    ),
                )
            )

        dictionary = check_array(dictionary, order="F", dtype=X.dtype, copy=False)
        dictionary = np.require(dictionary, requirements="W")

        return dictionary

    def _update_inner_stats(self, X, code, batch_size, step):
        # [FUNCTION DOC]
        # HELPER: Down-weights and updates the running statistics. theta is (step + 1)
        # times batch_size for early steps and batch_size squared plus step plus 1 minus
        # batch_size afterwards; beta is (theta + 1 - batch_size) / (theta + 1).
        # Multiplies _A and _B by beta then adds code transposed times code and X
        # transposed times code, each divided by batch_size.
        # Side effects: mutates _A and _B in place.
        """Update the inner stats inplace."""
        if step < batch_size - 1:
            theta = (step + 1) * batch_size
        else:
            theta = batch_size**2 + step + 1 - batch_size
        beta = (theta + 1 - batch_size) / (theta + 1)

        self._A *= beta
        self._A += code.T @ code / batch_size
        self._B *= beta
        self._B += X.T @ code / batch_size

    def _minibatch_step(self, X, dictionary, random_state, step):
        # [FUNCTION DOC]
        # HELPER: Encodes the batch with _sparse_encode using the fit algorithm and
        # alpha, computes the mean batch cost (half squared residual plus alpha times L1
        # norm, divided by batch size), updates the inner statistics and then the
        # dictionary in place with _update_dict using _A and _B.
        # Outputs: batch cost (float).
        # Side effects: mutates dictionary, _A, _B; may resample unused atoms
        # (randomness).
        """Perform the update on the dictionary for one minibatch."""
        batch_size = X.shape[0]

        # Compute code for this batch
        code = _sparse_encode(
            X,
            dictionary,
            algorithm=self._fit_algorithm,
            alpha=self.alpha,
            n_jobs=self.n_jobs,
            positive=self.positive_code,
            max_iter=self.transform_max_iter,
            verbose=self.verbose,
        )

        batch_cost = (
            0.5 * ((X - code @ dictionary) ** 2).sum()
            + self.alpha * np.sum(np.abs(code))
        ) / batch_size

        # Update inner stats
        self._update_inner_stats(X, code, batch_size, step)

        # Update dictionary
        _update_dict(
            dictionary,
            X,
            code,
            self._A,
            self._B,
            verbose=self.verbose,
            random_state=random_state,
            positive=self.positive_dict,
        )

        return batch_cost

    def _check_convergence(
        self, X, batch_cost, new_dict, old_dict, n_samples, step, n_steps
    ):
        # [FUNCTION DOC]
        # HELPER: Early stopping. With the step counted from 1, ignores the first
        # min(100, n_samples / batch_size) steps. Then updates an exponentially weighted
        # average of the batch cost (weight batch_size / (n_samples + 1), at most 1);
        # stops if tol is positive and the Frobenius norm of the dictionary change
        # divided by n_components is at most tol; otherwise tracks the minimum EWA cost
        # and stops when it has not improved for max_no_improvement consecutive steps
        # (unless that is None).
        # Outputs: True to stop, else False.
        # Side effects: updates _ewa_cost, _ewa_cost_min, _no_improvement; prints when
        # verbose.
        """Helper function to encapsulate the early stopping logic.

        Early stopping is based on two factors:
        - A small change of the dictionary between two minibatch updates. This is
          controlled by the tol parameter.
        - No more improvement on a smoothed estimate of the objective function for a
          a certain number of consecutive minibatch updates. This is controlled by
          the max_no_improvement parameter.
        """
        batch_size = X.shape[0]

        # counts steps starting from 1 for user friendly verbose mode.
        step = step + 1

        # Ignore 100 first steps or 1 epoch to avoid initializing the ewa_cost with a
        # too bad value
        if step <= min(100, n_samples / batch_size):
            if self.verbose:
                print(f"Minibatch step {step}/{n_steps}: mean batch cost: {batch_cost}")
            return False

        # Compute an Exponentially Weighted Average of the cost function to
        # monitor the convergence while discarding minibatch-local stochastic
        # variability: https://en.wikipedia.org/wiki/Moving_average
        if self._ewa_cost is None:
            self._ewa_cost = batch_cost
        else:
            alpha = batch_size / (n_samples + 1)
            alpha = min(alpha, 1)
            self._ewa_cost = self._ewa_cost * (1 - alpha) + batch_cost * alpha

        if self.verbose:
            print(
                f"Minibatch step {step}/{n_steps}: mean batch cost: "
                f"{batch_cost}, ewa cost: {self._ewa_cost}"
            )

        # Early stopping based on change of dictionary
        dict_diff = linalg.norm(new_dict - old_dict) / self._n_components
        if self.tol > 0 and dict_diff <= self.tol:
            if self.verbose:
                print(f"Converged (small dictionary change) at step {step}/{n_steps}")
            return True

        # Early stopping heuristic due to lack of improvement on smoothed
        # cost function
        if self._ewa_cost_min is None or self._ewa_cost < self._ewa_cost_min:
            self._no_improvement = 0
            self._ewa_cost_min = self._ewa_cost
        else:
            self._no_improvement += 1

        if (
            self.max_no_improvement is not None
            and self._no_improvement >= self.max_no_improvement
        ):
            if self.verbose:
                print(
                    "Converged (lack of improvement in objective function) "
                    f"at step {step}/{n_steps}"
                )
            return True

        return False

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Validates X (float, C order), sets private params and random state,
        # initializes the dictionary, optionally shuffles a copy of X, zero-initializes
        # _A and _B and the convergence state, then cycles over mini-batches for at most
        # max_iter epochs: performs a minibatch step, checks convergence, calls callback
        # with locals, and stores a copy of the dictionary for the next comparison.
        # Outputs: self.
        # Side effects: sets components_, n_steps_, n_iter_ (float, ceiling of steps per
        # epoch), and the private state used by partial_fit; prints when verbose;
        # randomness.
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = validate_data(
            self, X, dtype=[np.float64, np.float32], order="C", copy=False
        )

        self._check_params(X)
        self._random_state = check_random_state(self.random_state)

        dictionary = self._initialize_dict(X, self._random_state)
        old_dict = dictionary.copy()

        if self.shuffle:
            X_train = X.copy()
            self._random_state.shuffle(X_train)
        else:
            X_train = X

        n_samples, n_features = X_train.shape

        if self.verbose:
            print("[dict_learning]")

        # Inner stats
        self._A = np.zeros(
            (self._n_components, self._n_components), dtype=X_train.dtype
        )
        self._B = np.zeros((n_features, self._n_components), dtype=X_train.dtype)

        # Attributes to monitor the convergence
        self._ewa_cost = None
        self._ewa_cost_min = None
        self._no_improvement = 0

        batches = gen_batches(n_samples, self._batch_size)
        batches = itertools.cycle(batches)
        n_steps_per_iter = int(np.ceil(n_samples / self._batch_size))
        n_steps = self.max_iter * n_steps_per_iter

        i = -1  # to allow max_iter = 0

        for i, batch in zip(range(n_steps), batches):
            X_batch = X_train[batch]

            batch_cost = self._minibatch_step(
                X_batch, dictionary, self._random_state, i
            )

            if self._check_convergence(
                X_batch, batch_cost, dictionary, old_dict, n_samples, i, n_steps
            ):
                break

            # XXX callback param added for backward compat in #18975 but a common
            # unified callback API should be preferred
            if self.callback is not None:
                self.callback(locals())

            old_dict[:] = dictionary

        self.n_steps_ = i + 1
        self.n_iter_ = np.ceil(self.n_steps_ / n_steps_per_iter)
        self.components_ = dictionary

        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None):
        # [FUNCTION DOC]
        # Performs a single minibatch step on X. On first call (no components_)
        # validates X with reset, initializes params, random state, dictionary, n_steps_
        # and zeroed statistics; otherwise continues from components_ and the stored
        # state.
        # Outputs: self.
        # Side effects: updates components_, _A, _B and increments n_steps_.
        """Update the model using the data in X as a mini-batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        has_components = hasattr(self, "components_")

        X = validate_data(
            self, X, dtype=[np.float64, np.float32], order="C", reset=not has_components
        )

        if not has_components:
            # This instance has not been fitted yet (fit or partial_fit)
            self._check_params(X)
            self._random_state = check_random_state(self.random_state)

            dictionary = self._initialize_dict(X, self._random_state)

            self.n_steps_ = 0

            self._A = np.zeros((self._n_components, self._n_components), dtype=X.dtype)
            self._B = np.zeros((X.shape[1], self._n_components), dtype=X.dtype)
        else:
            dictionary = self.components_

        self._minibatch_step(X, dictionary, self._random_state, self.n_steps_)

        self.components_ = dictionary
        self.n_steps_ += 1

        return self

    @property
    def _n_features_out(self):
        # [FUNCTION DOC]
        # HELPER: Number of output features (rows of components_). Ignores split_sign.
        """Number of transformed output features."""
        return self.components_.shape[0]

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Declares float64/float32 dtype preservation.
        tags = super().__sklearn_tags__()
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags
