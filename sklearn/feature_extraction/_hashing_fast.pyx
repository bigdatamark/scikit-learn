# [MODULE DOC]
# Compiled core of sklearn.feature_extraction.FeatureHasher: hashes (feature,
# value) pairs with signed 32-bit MurmurHash3 into a fixed number of columns and
# builds the CSR components (indices, indptr, data) of the output matrix.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from libc.stdlib cimport abs
from libcpp.vector cimport vector

cimport numpy as cnp
import numpy as np
from sklearn.utils._typedefs cimport int32_t, int64_t
from sklearn.utils.murmurhash cimport murmurhash3_bytes_s32
from sklearn.utils._vector_sentinel cimport vector_to_nd_array

cnp.import_array()


def transform(raw_X, Py_ssize_t n_features, dtype,
              bint alternate_sign=1, unsigned int seed=0):
    # [FUNCTION DOC]
    # HELPER: implementation of FeatureHasher.transform.
    # Behavior: for each sample (an iterable of (name, value) pairs) and each pair:
    # string values turn into the feature name "name=value" with value 1; zero values
    # are skipped; names are UTF-8 encoded; the bytes are hashed with MurmurHash3
    # (seed) to a signed 32-bit h. Column index is abs(h) mod n_features (with the
    # -2**31 case handled explicitly to avoid undefined abs). If alternate_sign, the
    # value is negated when h < 0. Indices/indptr are accumulated in growable C++
    # vectors; values in a numpy array of the requested dtype that doubles in size
    # when full. indptr gets the running count after each sample. Finally, if the
    # total nnz exceeds int32 max, indices are widened to int64, otherwise indptr is
    # narrowed to int32, so both share a dtype.
    # Inputs: raw_X, iterable of iterables of (str|bytes name, str or number value);
    # n_features, number of output columns (> 0); dtype, numpy dtype for values;
    # alternate_sign, flag; seed, hash seed.
    # Outputs: tuple (indices, indptr, values) of numpy arrays for a CSR matrix with
    # len(indptr)-1 rows.
    # Exceptions: TypeError if a feature name is not str/bytes; errors from
    # converting non-numeric values to double.
    # Side effects: allocates arrays; none else.
    # KNOWN ISSUE: docstring says it returns n_samples plus indices, indptr, values as
    # lists; it actually returns three numpy arrays and no n_samples.
    """Guts of FeatureHasher.transform.

    Returns
    -------
    n_samples : integer
    indices, indptr, values : lists
        For constructing a scipy.sparse.csr_matrix.

    """
    cdef int32_t h
    cdef double value

    cdef vector[int32_t] indices
    cdef vector[int64_t] indptr
    indptr.push_back(0)

    # Since Python array does not understand Numpy dtypes, we grow the indices
    # and values arrays ourselves. Use a Py_ssize_t capacity for safety.
    cdef Py_ssize_t capacity = 8192     # arbitrary
    cdef int64_t size = 0
    cdef cnp.ndarray values = np.empty(capacity, dtype=dtype)

    for x in raw_X:
        for f, v in x:
            if isinstance(v, (str, unicode)):
                f = "%s%s%s" % (f, '=', v)
                value = 1
            else:
                value = v

            if value == 0:
                continue

            if isinstance(f, unicode):
                f = (<unicode>f).encode("utf-8")
            # Need explicit type check because Murmurhash does not propagate
            # all exceptions. Add "except *" there?
            elif not isinstance(f, bytes):
                raise TypeError("feature names must be strings")

            h = murmurhash3_bytes_s32(<bytes>f, seed)

            if h == - 2147483648:
                # abs(-2**31) is undefined behavior because h is a `np.int32`
                # The following is defined such that it is equal to: abs(-2**31) % n_features
                indices.push_back((2147483647 - (n_features - 1)) % n_features)
            else:
                indices.push_back(abs(h) % n_features)
            # improve inner product preservation in the hashed space
            if alternate_sign:
                value *= (h >= 0) * 2 - 1
            values[size] = value
            size += 1

            if size == capacity:
                capacity *= 2
                # can't use resize member because there might be multiple
                # references to the arrays due to Cython's error checking
                values = np.resize(values, capacity)

        indptr.push_back(size)

    indices_array = vector_to_nd_array(&indices)
    indptr_array = vector_to_nd_array(&indptr)

    if indptr_array[indptr_array.shape[0]-1] > np.iinfo(np.int32).max:  # = 2**31 - 1
        # both indices and indptr have the same dtype in CSR arrays
        indices_array = indices_array.astype(np.int64, copy=False)
    else:
        indptr_array = indptr_array.astype(np.int32, copy=False)

    return (indices_array, indptr_array, values[:size])
