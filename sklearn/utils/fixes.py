# [MODULE DOC]
# Compatibility layer for NumPy, SciPy, pandas, matplotlib and Python versions.
# At import time it computes version flags (np_version, sp_version and their base
# versions, SCIPY_VERSION_BELOW_1_12/_1_15, SPARRAY_PRESENT, _IS_32BIT, _IS_WASM),
# lists of sparse container classes per format (CSR_CONTAINERS, etc.), and binds
# version-dependent implementations (_sparse_linalg_cg, _sparse_min_max,
# _sparse_nan_min_max, ComplexWarning, VisibleDeprecationWarning, laplacian,
# _sparse_eye_array, _sparse_diags_array, _sparse_random_array) chosen by
# module-level if/else on the installed versions. Also exposes PETROFF_COLORS.
# Side effects: imports pandas if available (else pd is None); no other state.
"""Compatibility fixes for older version of the dependencies

If you add content to this file, please give the version of the package
at which the fix is no longer needed.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import platform
import struct

import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.stats

try:
    import pandas as pd
except ImportError:
    pd = None

from sklearn.externals._packaging.version import parse as parse_version
from sklearn.utils.parallel import _get_threadpool_controller

_IS_32BIT = 8 * struct.calcsize("P") == 32
_IS_WASM = platform.machine() in ["wasm32", "wasm64"]

np_version = parse_version(np.__version__)
np_base_version = parse_version(np_version.base_version)
sp_version = parse_version(scipy.__version__)
sp_base_version = parse_version(sp_version.base_version)

# TODO: We can consider removing the containers and importing
# directly from SciPy when sparse matrices will be deprecated.
CSR_CONTAINERS = [scipy.sparse.csr_matrix, scipy.sparse.csr_array]
CSC_CONTAINERS = [scipy.sparse.csc_matrix, scipy.sparse.csc_array]
COO_CONTAINERS = [scipy.sparse.coo_matrix, scipy.sparse.coo_array]
LIL_CONTAINERS = [scipy.sparse.lil_matrix, scipy.sparse.lil_array]
DOK_CONTAINERS = [scipy.sparse.dok_matrix, scipy.sparse.dok_array]
BSR_CONTAINERS = [scipy.sparse.bsr_matrix, scipy.sparse.bsr_array]
DIA_CONTAINERS = [scipy.sparse.dia_matrix, scipy.sparse.dia_array]

# Remove when minimum scipy version is 1.11.0
try:
    from scipy.sparse import sparray  # noqa: F401

    SPARRAY_PRESENT = True
except ImportError:
    SPARRAY_PRESENT = False


def _object_dtype_isnan(X):
    # [FUNCTION DOC]
    # HELPER: NaN mask for object-dtype arrays, using the fact that NaN != NaN.
    # Inputs: X - array (typically object dtype). Outputs: boolean array, same shape.
    return X != X


# TODO: Remove when SciPy 1.11 is the minimum supported version
def _mode(a, axis=0):
    # [FUNCTION DOC]
    # HELPER: Wrapper around scipy.stats.mode with keepdims=True that, for SciPy
    # >= 1.11 and axis=None, ravels the result to restore the older output shape.
    # Inputs: a - array; axis - int or None (default 0).
    # Outputs: scipy ModeResult (mode, count), or a raveled array when axis is None
    # on new SciPy.
    # KNOWN ISSUE: with axis=None on new SciPy, np.ravel is applied to the whole
    # ModeResult tuple, giving a flat array of [mode, count] rather than a ModeResult.
    mode = scipy.stats.mode(a, axis=axis, keepdims=True)
    if sp_version >= parse_version("1.10.999"):
        # scipy.stats.mode has changed returned array shape with axis=None
        # and keepdims=True, see https://github.com/scipy/scipy/pull/17561
        if axis is None:
            mode = np.ravel(mode)
    return mode


# TODO: Remove when Scipy 1.12 is the minimum supported version
#       Use git grep to see where this is used and update them too.
SCIPY_VERSION_BELOW_1_12 = sp_base_version < parse_version("1.12.0")


# TODO: Remove when Scipy 1.15 is the minimum supported version
#       Use git grep to see where this is used and update them too.
SCIPY_VERSION_BELOW_1_15 = sp_base_version < parse_version("1.15.0")


# TODO: Remove when Scipy 1.12 is the minimum supported version
if sp_base_version >= parse_version("1.12.0"):
    _sparse_linalg_cg = scipy.sparse.linalg.cg
else:

    def _sparse_linalg_cg(A, b, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Backport of scipy.sparse.linalg.cg for SciPy < 1.12: renames the rtol
        # keyword to tol and defaults atol to "legacy" when not given, then calls cg.
        # Inputs: A - linear operator/matrix; b - right-hand side; kwargs - cg options.
        # Outputs: (solution, info) as returned by scipy cg.
        # Side effects: mutates the local kwargs dict only.
        if "rtol" in kwargs:
            kwargs["tol"] = kwargs.pop("rtol")
        if "atol" not in kwargs:
            kwargs["atol"] = "legacy"
        return scipy.sparse.linalg.cg(A, b, **kwargs)


# TODO: Fuse the modern implementations of _sparse_min_max and _sparse_nan_min_max
# into the public min_max_axis function when SciPy 1.11 is the minimum supported
# version and delete the backport in the else branch below.
if sp_base_version >= parse_version("1.11.0"):

    def _sparse_min_max(X, axis):
        # [FUNCTION DOC]
        # HELPER: Min and max of a sparse matrix (SciPy >= 1.11) along axis using the
        # native min/max methods; when axis is not None densifies results to 1D arrays.
        # Inputs: X - sparse matrix/array; axis - None, 0 or 1.
        # Outputs: tuple (mins, maxs) - scalars if axis is None, else 1D ndarrays.
        the_min = X.min(axis=axis)
        the_max = X.max(axis=axis)

        if axis is not None:
            the_min = the_min.toarray().ravel()
            the_max = the_max.toarray().ravel()

        return the_min, the_max

    def _sparse_nan_min_max(X, axis):
        # [FUNCTION DOC]
        # HELPER: Same as _sparse_min_max but ignoring NaNs (nanmin/nanmax methods).
        # Inputs/Outputs: as _sparse_min_max.
        the_min = X.nanmin(axis=axis)
        the_max = X.nanmax(axis=axis)

        if axis is not None:
            the_min = the_min.toarray().ravel()
            the_max = the_max.toarray().ravel()

        return the_min, the_max

else:
    # This code is mostly taken from scipy 0.14 and extended to handle nans, see
    # https://github.com/scikit-learn/scikit-learn/pull/11196
    def _minor_reduce(X, ufunc):
        # [FUNCTION DOC]
        # HELPER: Applies a ufunc reduction over each non-empty major slice (row of CSR /
        # column of CSC) of X using ufunc.reduceat on X.data at the indptr starts.
        # Inputs: X - CSR or CSC matrix with sorted, deduplicated structure; ufunc -
        # binary numpy ufunc (e.g. np.minimum).
        # Outputs: (major_index, value) - indices of non-empty major slices and the
        # reduced value of each.
        # Side effects: rebuilds X locally to reset index dtypes; input unchanged.
        major_index = np.flatnonzero(np.diff(X.indptr))

        # reduceat tries casts X.indptr to intp, which errors
        # if it is int64 on a 32 bit system.
        # Reinitializing prevents this where possible, see #13737
        X = type(X)((X.data, X.indices, X.indptr), shape=X.shape)
        value = ufunc.reduceat(X.data, X.indptr[major_index])
        return major_index, value

    def _min_or_max_axis(X, axis, min_or_max):
        # [FUNCTION DOC]
        # HELPER: Backport min or max along axis 0 or 1 for sparse matrices. Converts to
        # CSC (axis 0) or CSR (axis 1), sums duplicates, reduces explicit values per
        # slice, then combines with implicit zeros for slices not completely filled.
        # Inputs: X - sparse matrix; axis - 0 or 1; min_or_max - np.minimum/np.maximum
        # (or np.fmin/np.fmax).
        # Outputs: dense 1D array of length X.shape[1 - axis].
        # Exceptions: ValueError when the reduced dimension has size 0.
        # Side effects: sum_duplicates is applied to the converted copy; if X is already
        # in the target format, tocsc/tocsr may return X itself so X can be mutated.
        N = X.shape[axis]
        if N == 0:
            raise ValueError("zero-size array to reduction operation")
        M = X.shape[1 - axis]
        mat = X.tocsc() if axis == 0 else X.tocsr()
        mat.sum_duplicates()
        major_index, value = _minor_reduce(mat, min_or_max)
        not_full = np.diff(mat.indptr)[major_index] < N
        value[not_full] = min_or_max(value[not_full], 0)
        mask = value != 0
        major_index = np.compress(mask, major_index)
        value = np.compress(mask, value)

        if axis == 0:
            res = scipy.sparse.coo_array(
                (value, (np.zeros(len(value)), major_index)),
                dtype=X.dtype,
                shape=(1, M),
            )
        else:
            res = scipy.sparse.coo_array(
                (value, (major_index, np.zeros(len(value)))),
                dtype=X.dtype,
                shape=(M, 1),
            )
        return res.toarray().ravel()

    def _sparse_min_or_max(X, axis, min_or_max):
        # [FUNCTION DOC]
        # HELPER: Backport min or max of a sparse matrix over all entries (axis None) or
        # along an axis. For axis None, reduces stored data and includes zero if any
        # implicit zeros exist. Negative axis values are wrapped by adding 2.
        # Inputs: X - sparse matrix; axis - None, 0, 1, -1, -2; min_or_max - ufunc.
        # Outputs: scalar (axis None) or 1D dense array.
        # Exceptions: ValueError for empty matrices with axis None or for invalid axis.
        if axis is None:
            if 0 in X.shape:
                raise ValueError("zero-size array to reduction operation")
            zero = X.dtype.type(0)
            if X.nnz == 0:
                return zero
            m = min_or_max.reduce(X.data.ravel())
            if X.nnz != np.prod(X.shape):
                m = min_or_max(zero, m)
            return m
        if axis < 0:
            axis += 2
        if (axis == 0) or (axis == 1):
            return _min_or_max_axis(X, axis, min_or_max)
        else:
            raise ValueError("invalid axis, use 0 for rows, or 1 for columns")

    def _sparse_min_max(X, axis):
        # [FUNCTION DOC]
        # HELPER: Backport returning (min, max) via _sparse_min_or_max with np.minimum
        # and np.maximum (NaN propagating).
        return (
            _sparse_min_or_max(X, axis, np.minimum),
            _sparse_min_or_max(X, axis, np.maximum),
        )

    def _sparse_nan_min_max(X, axis):
        # [FUNCTION DOC]
        # HELPER: Backport returning (nanmin, nanmax) via _sparse_min_or_max with
        # np.fmin and np.fmax, which ignore NaNs when another value is available.
        return (
            _sparse_min_or_max(X, axis, np.fmin),
            _sparse_min_or_max(X, axis, np.fmax),
        )


# For +1.25 NumPy versions exceptions and warnings are being moved
# to a dedicated submodule.
if np_version >= parse_version("1.25.0"):
    from numpy.exceptions import ComplexWarning, VisibleDeprecationWarning
else:
    from numpy import (  # noqa: F401
        ComplexWarning,
        VisibleDeprecationWarning,
    )


# TODO: Adapt when Pandas > 2.2 is the minimum supported version
def pd_fillna(pd, frame):
    # [FUNCTION DOC]
    # Replaces missing values of a pandas DataFrame/Series with np.nan in a way that
    # avoids pandas downcasting FutureWarnings: plain fillna for pandas < 2.2;
    # otherwise fillna then infer_objects, under the future.no_silent_downcasting
    # option for pandas < 3.0, and passing copy=False to infer_objects only for
    # pandas < 3.
    # Inputs: pd - the pandas module; frame - DataFrame or Series.
    # Outputs: new frame with missing values set to np.nan.
    pd_version = parse_version(pd.__version__).base_version
    if parse_version(pd_version) < parse_version("2.2"):
        frame = frame.fillna(value=np.nan)
    else:
        infer_objects_kwargs = (
            {} if parse_version(pd_version) >= parse_version("3") else {"copy": False}
        )
        if parse_version(pd_version) < parse_version("3.0"):
            with pd.option_context("future.no_silent_downcasting", True):
                frame = frame.fillna(value=np.nan).infer_objects(**infer_objects_kwargs)
        else:
            frame = frame.fillna(value=np.nan).infer_objects(**infer_objects_kwargs)
    return frame


# TODO: remove when SciPy 1.12 is the minimum supported version
def _preserve_dia_indices_dtype(
    sparse_container, original_container_format, requested_sparse_format
):
    # [FUNCTION DOC]
    # HELPER: When a DIA array was converted to CSR or COO, downcasts its index arrays
    # to the smallest admissible dtype (int32 where possible) for consistency with
    # DIA matrices on SciPy < 1.12. For CSR checks indptr/indices contents with
    # maxval = max(nnz, n_cols); for COO uses maxval = max(shape) on row/col.
    # Inputs: sparse_container - converted sparse object; original_container_format -
    # type name string of the original container (e.g. "dia_array");
    # requested_sparse_format - target format string.
    # Outputs: None.
    # Side effects: reassigns index attributes of sparse_container in place.
    # KNOWN ISSUE: docstring omits original_container_format and only CSR/COO (not
    # CSC as the summary says) are handled.
    """Preserve indices dtype for SciPy < 1.12 when converting from DIA to CSR/CSC.

    For SciPy < 1.12, DIA arrays indices are upcasted to `np.int64` that is
    inconsistent with DIA matrices. We downcast the indices dtype to `np.int32` to
    be consistent with DIA matrices.

    The converted indices arrays are affected back inplace to the sparse container.

    Parameters
    ----------
    sparse_container : sparse container
        Sparse container to be checked.
    requested_sparse_format : str or bool
        The type of format of `sparse_container`.

    Notes
    -----
    See https://github.com/scipy/scipy/issues/19245 for more details.
    """
    if original_container_format == "dia_array" and requested_sparse_format in (
        "csr",
        "coo",
    ):
        if requested_sparse_format == "csr":
            index_dtype = _smallest_admissible_index_dtype(
                arrays=(sparse_container.indptr, sparse_container.indices),
                maxval=max(sparse_container.nnz, sparse_container.shape[1]),
                check_contents=True,
            )
            sparse_container.indices = sparse_container.indices.astype(
                index_dtype, copy=False
            )
            sparse_container.indptr = sparse_container.indptr.astype(
                index_dtype, copy=False
            )
        else:  # requested_sparse_format == "coo"
            index_dtype = _smallest_admissible_index_dtype(
                maxval=max(sparse_container.shape)
            )
            sparse_container.row = sparse_container.row.astype(index_dtype, copy=False)
            sparse_container.col = sparse_container.col.astype(index_dtype, copy=False)


# TODO: remove when SciPy 1.12 is the minimum supported version
def _smallest_admissible_index_dtype(arrays=(), maxval=None, check_contents=False):
    # [FUNCTION DOC]
    # HELPER: Chooses np.int32 or np.int64 as index dtype. Returns int64 if maxval
    # exceeds the int32 max, or if any array's dtype cannot be cast to int32 (when
    # check_contents is False) or its non-empty contents fall outside the int32 range
    # (when check_contents is True); otherwise int32.
    # Inputs: arrays - ndarray or tuple of integer ndarrays; maxval - number or None;
    # check_contents - bool.
    # Outputs: np.int32 or np.int64 type.
    # Exceptions: ValueError if maxval exceeds int64 max or an array is not integer;
    # TypeError if an element of arrays is not an ndarray.
    """Based on input (integer) arrays `a`, determine a suitable index data
    type that can hold the data in the arrays.

    This function returns `np.int64` if it either required by `maxval` or based on the
    largest precision of the dtype of the arrays passed as argument, or by their
    contents (when `check_contents is True`). If none of the condition requires
    `np.int64` then this function returns `np.int32`.

    Parameters
    ----------
    arrays : ndarray or tuple of ndarrays, default=()
        Input arrays whose types/contents to check.

    maxval : float, default=None
        Maximum value needed.

    check_contents : bool, default=False
        Whether to check the values in the arrays and not just their types.
        By default, check only the types.

    Returns
    -------
    dtype : {np.int32, np.int64}
        Suitable index data type (int32 or int64).
    """

    int32min = np.int32(np.iinfo(np.int32).min)
    int32max = np.int32(np.iinfo(np.int32).max)

    if maxval is not None:
        if maxval > np.iinfo(np.int64).max:
            raise ValueError(
                f"maxval={maxval} is to large to be represented as np.int64."
            )
        if maxval > int32max:
            return np.int64

    if isinstance(arrays, np.ndarray):
        arrays = (arrays,)

    for arr in arrays:
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"Arrays should be of type np.ndarray, got {type(arr)} instead."
            )
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(
                f"Array dtype {arr.dtype} is not supported for index dtype. We expect "
                "integral values."
            )
        if not np.can_cast(arr.dtype, np.int32):
            if not check_contents:
                # when `check_contents` is False, we stay on the safe side and return
                # np.int64.
                return np.int64
            if arr.size == 0:
                # a bigger type not needed yet, let's look at the next array
                continue
            else:
                maxval = arr.max()
                minval = arr.min()
                if minval < int32min or maxval > int32max:
                    # a big index type is actually needed
                    return np.int64

    return np.int32


# TODO: Remove when SciPy 1.12 is the minimum supported version
if sp_version < parse_version("1.12"):
    from sklearn.externals._scipy.sparse.csgraph import laplacian
else:
    from scipy.sparse.csgraph import (
        laplacian,  # noqa: F401  # pragma: no cover
    )


# TODO: Remove when Python min version >= 3.12.
def tarfile_extractall(tarfile, path):
    # [FUNCTION DOC]
    # Extracts all members of an open tarfile to path using filter="data" for safety;
    # falls back to an unfiltered extractall when the Python version does not
    # support the filter argument (TypeError).
    # Inputs: tarfile - tarfile.TarFile object; path - destination directory.
    # Side effects: writes files to disk.
    try:
        # Use filter="data" to prevent the most dangerous security issues.
        # For more details, see
        # https://docs.python.org/3/library/tarfile.html#tarfile.TarFile.extractall
        tarfile.extractall(path, filter="data")
    except TypeError:
        tarfile.extractall(path)


def _in_unstable_openblas_configuration():
    # [FUNCTION DOC]
    # HELPER: Detects whether OpenBLAS is loaded in a known-unstable configuration.
    # Imports numpy/scipy, inspects threadpoolctl info; returns False if OpenBLAS is
    # not used; True if any OpenBLAS lib has unknown version/architecture, or is a
    # neoversen1 build older than 0.3.16; False otherwise.
    # Outputs: bool.
    """Return True if in an unstable configuration for OpenBLAS"""

    # Import libraries which might load OpenBLAS.
    import numpy  # noqa: F401
    import scipy  # noqa: F401

    modules_info = _get_threadpool_controller().info()

    open_blas_used = any(info["internal_api"] == "openblas" for info in modules_info)
    if not open_blas_used:
        return False

    # OpenBLAS 0.3.16 fixed instability for arm64, see:
    # https://github.com/xianyi/OpenBLAS/blob/1b6db3dbba672b4f8af935bd43a1ff6cff4d20b7/Changelog.txt#L56-L58
    openblas_arm64_stable_version = parse_version("0.3.16")
    for info in modules_info:
        if info["internal_api"] != "openblas":
            continue
        openblas_version = info.get("version")
        openblas_architecture = info.get("architecture")
        if openblas_version is None or openblas_architecture is None:
            # Cannot be sure that OpenBLAS is good enough. Assume unstable:
            return True  # pragma: no cover
        if (
            openblas_architecture == "neoversen1"
            and parse_version(openblas_version) < openblas_arm64_stable_version
        ):
            # See discussions in https://github.com/numpy/numpy/issues/19411
            return True  # pragma: no cover
    return False


# TODO: Remove when Scipy 1.15 is the minimum supported version. In scipy 1.15,
# the internal info details (via 'iprint' and 'disp' options) were dropped,
# following the LBFGS rewrite from Fortran to C, see
# https://github.com/scipy/scipy/issues/23186#issuecomment-2987801035. For
# scipy 1.15, 'iprint' and 'disp' have no effect and for scipy >= 1.16 a
# DeprecationWarning is emitted.
def _get_additional_lbfgs_options_dict(key, value):
    # [FUNCTION DOC]
    # HELPER: Returns {key: value} for SciPy < 1.15 and an empty dict otherwise, so
    # L-BFGS options like iprint/disp are only passed where they still have effect.
    return {} if sp_version >= parse_version("1.15") else {key: value}


# TODO: Replace when Scipy 1.12 is the minimum supported version
#       fixes for transitioning scipy.sparse function names
if not SCIPY_VERSION_BELOW_1_12:
    _sparse_eye_array = scipy.sparse.eye_array
    _sparse_diags_array = scipy.sparse.diags_array

    def _sparse_random_array(
        shape,
        *,
        density=0.01,
        format="coo",
        dtype=None,
        random_state=None,
        rng=None,
        data_sampler=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Random sparse array (SciPy >= 1.12) via scipy.sparse.random_array,
        # then forces index arrays to int32 when safe.
        # Inputs: shape - tuple; density - float in [0, 1]; format - sparse format str;
        # dtype; random_state / rng - seed or generator (rng takes precedence if truthy);
        # data_sampler - callable producing values.
        # Outputs: sparse array in the requested format.
        # Side effects: consumes randomness from the given generator.
        X = scipy.sparse.random_array(
            shape,
            density=density,
            format=format,
            dtype=dtype,
            random_state=rng or random_state,
            data_sampler=data_sampler,
        )
        _ensure_sparse_index_int32(X)
        return X

else:

    def _sparse_eye_array(m, n=None, *, k=0, dtype=float, format=None):
        # [FUNCTION DOC]
        # HELPER: Backport of scipy.sparse.eye_array for SciPy < 1.12: builds a sparse
        # identity-like matrix with diagonal offset k and converts to a DIA array then to
        # format. Inputs: m, n - shape (n defaults to m); k - diagonal; dtype; format.
        # Outputs: sparse array.
        A = scipy.sparse.eye(m, n, k=k, dtype=dtype)
        return scipy.sparse.dia_array(A).asformat(format)

    def _sparse_diags_array(
        diagonals, /, *, offsets=0, shape=None, format=None, dtype=None
    ):
        # [FUNCTION DOC]
        # HELPER: Backport of scipy.sparse.diags_array for SciPy < 1.12 via
        # scipy.sparse.diags then conversion to DIA array and to format.
        # Inputs: diagonals, offsets, shape, format, dtype as in scipy.sparse.diags.
        # Outputs: sparse array.
        A = scipy.sparse.diags(diagonals, offsets=offsets, shape=shape, dtype=dtype)
        return scipy.sparse.dia_array(A).asformat(format)

    def _sparse_random_array(
        shape,
        *,
        density=0.01,
        format="coo",
        dtype=None,
        random_state=None,
        rng=None,
        data_sampler=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Backport of _sparse_random_array for SciPy < 1.12 using
        # scipy.sparse.random (data_sampler passed as data_rvs), converted to a COO array
        # then to format. Index dtypes are not forced to int32 here.
        # Inputs/Outputs: as the modern variant above; shape must have 2 elements.
        A = scipy.sparse.random(
            *shape,
            density=density,
            dtype=dtype,
            random_state=rng or random_state,
            data_rvs=data_sampler,
        )
        return scipy.sparse.coo_array(A).asformat(format)


# TODO: remove when SciPy 1.15 is minimal supported version
# fix for casting index arrays
def _ensure_sparse_index_int32(A):
    # [FUNCTION DOC]
    # HELPER: Casts index arrays of a sparse array to int32 in place when safe,
    # dispatching on format (csr/csc/bsr: indices and indptr; coo: coords, indices or
    # row/col depending on SciPy version; dia: offsets). Other formats untouched.
    # Inputs: A - sparse array. Outputs: None.
    # Exceptions: ValueError propagated from _safely_cast_index_arrays.
    # Side effects: mutates A's index attributes.
    """Safely ensure that index arrays are int32."""
    if A.format in ("csc", "csr", "bsr"):
        A.indices, A.indptr = _safely_cast_index_arrays(A)
    elif A.format == "coo":
        if hasattr(A, "coords"):
            A.coords = _safely_cast_index_arrays(A)
        elif hasattr(A, "indices"):
            A.indices = _safely_cast_index_arrays(A)
        else:
            A.row, A.col = _safely_cast_index_arrays(A)
    elif A.format == "dia":
        A.offsets = _safely_cast_index_arrays(A)


# TODO: remove when SciPy 1.15 is minimal supported version
#       (based on scipy.sparse._sputils.py function with same name)
def _safely_cast_index_arrays(A, idx_dtype=np.int32, msg=""):
    # [FUNCTION DOC]
    # HELPER: Returns index arrays of A cast to idx_dtype (copy only when the dtype
    # changes), after checking values fit when a shape dimension exceeds the dtype
    # max. Inputs: A - sparse array; idx_dtype - integer dtype (default int32);
    # msg - text inserted into error messages.
    # Outputs: (indices, indptr) for csr/csc/bsr; tuple of coordinate arrays for coo;
    # offsets array for dia; None for other formats (dok, lil).
    # Exceptions: ValueError when values are too large for idx_dtype.
    # KNOWN ISSUE: the bsr indptr error message lacks the f prefix, so "{msg}" is
    # printed literally.
    """Safely cast sparse array indices to `idx_dtype`.

    Check the shape of `A` to determine if it is safe to cast its index
    arrays to dtype `idx_dtype`. If any dimension in shape is larger than
    fits in the dtype, casting is unsafe so raise ``ValueError``.
    If safe, cast the index arrays to `idx_dtype` and return the result
    without changing the input `A`. The caller can assign results to `A`
    attributes if desired or use the recast index arrays directly.

    Unless downcasting is needed, the original index arrays are returned.
    You can test e.g. ``A.indptr is new_indptr`` to see if downcasting occurred.

    See SciPy: scipy.sparse._sputils.py for more info on safely_cast_index_arrays()
    """
    max_value = np.iinfo(idx_dtype).max

    if A.format in ("csc", "csr"):
        if A.indptr[-1] > max_value:
            raise ValueError(f"indptr values too large for {msg}")
        # check shape vs dtype
        if max(*A.shape) > max_value:
            if (A.indices > max_value).any():
                raise ValueError(f"indices values too large for {msg}")

        indices = A.indices.astype(idx_dtype, copy=False)
        indptr = A.indptr.astype(idx_dtype, copy=False)
        return indices, indptr

    elif A.format == "coo":
        coords = getattr(A, "coords", None)
        if coords is None:
            coords = getattr(A, "indices", None)
            if coords is None:
                coords = (A.row, A.col)
        if max(*A.shape) > max_value:
            if any((co > max_value).any() for co in coords):
                raise ValueError(f"coords values too large for {msg}")
        return tuple(co.astype(idx_dtype, copy=False) for co in coords)

    elif A.format == "dia":
        if max(*A.shape) > max_value:
            if (A.offsets > max_value).any():
                raise ValueError(f"offsets values too large for {msg}")
        offsets = A.offsets.astype(idx_dtype, copy=False)
        return offsets

    elif A.format == "bsr":
        R, C = A.blocksize
        if A.indptr[-1] * R > max_value:
            raise ValueError("indptr values too large for {msg}")
        if max(*A.shape) > max_value:
            if (A.indices * C > max_value).any():
                raise ValueError(f"indices values too large for {msg}")
        indices = A.indices.astype(idx_dtype, copy=False)
        indptr = A.indptr.astype(idx_dtype, copy=False)
        return indices, indptr
    # DOK and LIL formats are not associated with index arrays.


# TODO remove when matplotlib 3.10 is the minimal supported version
# and replace usage with `mpl.color_sequences['petroff10']`
PETROFF_COLORS = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
]
