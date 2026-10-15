# [MODULE DOC]
# Wrappers around the bundled C++ MurmurHash3_x86_32 (src/MurmurHash3.cpp):
# cpdef functions hashing a C int or a bytes object to a signed or unsigned
# 32-bit value, private helpers vectorised over int32 arrays, and the public
# murmurhash3_32 dispatcher (exposed as sklearn.utils.murmurhash3_32) used for
# feature hashing and random projections.
"""Cython wrapper for MurmurHash3 non-cryptographic hash function.

MurmurHash is an extensively tested and very fast hash function that has
good distribution properties suitable for machine learning use cases
such as feature hashing and random projections.

The original C++ code by Austin Appleby is released the public domain
and can be found here:

  https://code.google.com/p/smhasher/

"""
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.utils._typedefs cimport int32_t, uint32_t

import numpy as np

cdef extern from "src/MurmurHash3.h":
    void MurmurHash3_x86_32(void *key, int len, uint32_t seed, void *out)
    # [FUNCTION DOC]
    # External C: 32-bit MurmurHash3 of len bytes at key with seed, written to out.
    void MurmurHash3_x86_128(void *key, int len, uint32_t seed, void *out)
    # [FUNCTION DOC]
    # External C: 128-bit x86 MurmurHash3 written to out (16 bytes). Unused here.
    void MurmurHash3_x64_128 (void *key, int len, uint32_t seed, void *out)
    # [FUNCTION DOC]
    # External C: 128-bit x64 MurmurHash3 written to out (16 bytes). Unused here.


cpdef uint32_t murmurhash3_int_u32(int key, unsigned int seed):
    # [FUNCTION DOC]
    # Behavior: hash the sizeof(int) bytes of key (native byte order) with seed.
    # Inputs: key (C int), seed (unsigned int). Outputs: uint32 hash.
    # Exceptions: OverflowError on Python-side conversion of out-of-range args.
    """Compute the 32bit murmurhash3 of an int key at seed."""
    cdef uint32_t out
    MurmurHash3_x86_32(&key, sizeof(int), seed, &out)
    return out


cpdef int32_t murmurhash3_int_s32(int key, unsigned int seed):
    # [FUNCTION DOC]
    # Behavior: same as murmurhash3_int_u32 but the 32-bit result is stored into a
    # signed int32 (two's complement reinterpretation). Outputs: int32 hash.
    """Compute the 32bit murmurhash3 of an int key at seed."""
    cdef int32_t out
    MurmurHash3_x86_32(&key, sizeof(int), seed, &out)
    return out


cpdef uint32_t murmurhash3_bytes_u32(bytes key, unsigned int seed):
    # [FUNCTION DOC]
    # Behavior: hash the len(key) bytes of key with seed. Inputs: key (bytes, not
    # None), seed. Outputs: uint32 hash. Exceptions: TypeError if key is not bytes.
    """Compute the 32bit murmurhash3 of a bytes key at seed."""
    cdef uint32_t out
    MurmurHash3_x86_32(<char*> key, len(key), seed, &out)
    return out


cpdef int32_t murmurhash3_bytes_s32(bytes key, unsigned int seed):
    # [FUNCTION DOC]
    # Behavior: same as murmurhash3_bytes_u32 but returns the signed int32
    # reinterpretation. Exceptions: TypeError if key is not bytes.
    """Compute the 32bit murmurhash3 of a bytes key at seed."""
    cdef int32_t out
    MurmurHash3_x86_32(<char*> key, len(key), seed, &out)
    return out


def _murmurhash3_bytes_array_u32(
    const int32_t[:] key,
    unsigned int seed,
):
    # [FUNCTION DOC]
    # HELPER: element-wise unsigned hash of an int32 array for murmurhash3_32
    # (despite the name, it hashes integers, not bytes).
    # Inputs: key (1d int32 memoryview), seed. Outputs: new uint32 ndarray of the
    # same length. Exceptions: buffer dtype errors. Side effects: allocates output.
    """Compute 32bit murmurhash3 hashes of a key int array at seed."""
    # TODO make it possible to pass preallocated output array
    cdef:
        uint32_t[:] out = np.zeros(key.size, np.uint32)
        Py_ssize_t i
    for i in range(key.shape[0]):
        out[i] = murmurhash3_int_u32(key[i], seed)
    return np.asarray(out)


def _murmurhash3_bytes_array_s32(
    const int32_t[:] key,
    unsigned int seed,
):
    # [FUNCTION DOC]
    # HELPER: element-wise signed hash of an int32 array for murmurhash3_32
    # (despite the name, it hashes integers, not bytes).
    # Inputs: key (1d int32 memoryview), seed. Outputs: new int32 ndarray.
    """Compute 32bit murmurhash3 hashes of a key int array at seed."""
    # TODO make it possible to pass preallocated output array
    cdef:
        int32_t[:] out = np.zeros(key.size, np.int32)
        Py_ssize_t i
    for i in range(key.shape[0]):
        out[i] = murmurhash3_int_s32(key[i], seed)
    return np.asarray(out)


def murmurhash3_32(key, seed=0, positive=False):
    # [FUNCTION DOC]
    # Behavior: dispatch on the type of key: bytes are hashed directly; str is
    # UTF-8 encoded then hashed; Python int or np.int32 is converted to int32 and
    # hashed as an int; an int32 ndarray of any shape is raveled, hashed
    # element-wise and reshaped. positive selects the unsigned result, otherwise
    # the signed reinterpretation.
    # Inputs: key, seed (int convertible to unsigned int), positive (bool).
    # Outputs: int, or ndarray of the same shape as key.
    # Exceptions: TypeError for unsupported types or non-int32 arrays;
    # OverflowError if an int key does not fit in int32 or seed does not fit in
    # unsigned int.
    """Compute the 32bit murmurhash3 of key at seed.

    The underlying implementation is MurmurHash3_x86_32 generating low
    latency 32bits hash suitable for implementing lookup tables, Bloom
    filters, count min sketch or feature hashing.

    Parameters
    ----------
    key : np.int32, bytes, unicode or ndarray of dtype=np.int32
        The physical object to hash.

    seed : int, default=0
        Integer seed for the hashing algorithm.

    positive : bool, default=False
        True: the results is casted to an unsigned int
          from 0 to 2 ** 32 - 1
        False: the results is casted to a signed int
          from -(2 ** 31) to 2 ** 31 - 1

    Examples
    --------
    >>> from sklearn.utils import murmurhash3_32
    >>> murmurhash3_32(b"Hello World!", seed=42)
    3565178
    """
    if isinstance(key, bytes):
        if positive:
            return murmurhash3_bytes_u32(key, seed)
        else:
            return murmurhash3_bytes_s32(key, seed)
    elif isinstance(key, unicode):
        if positive:
            return murmurhash3_bytes_u32(key.encode('utf-8'), seed)
        else:
            return murmurhash3_bytes_s32(key.encode('utf-8'), seed)
    elif isinstance(key, int) or isinstance(key, np.int32):
        if positive:
            return murmurhash3_int_u32(<int32_t>key, seed)
        else:
            return murmurhash3_int_s32(<int32_t>key, seed)
    elif isinstance(key, np.ndarray):
        if key.dtype != np.int32:
            raise TypeError(
                "key.dtype should be int32, got %s" % key.dtype)
        if positive:
            return _murmurhash3_bytes_array_u32(key.ravel(), seed).reshape(key.shape)
        else:
            return _murmurhash3_bytes_array_s32(key.ravel(), seed).reshape(key.shape)
    else:
        raise TypeError(
            "key %r with type %s is not supported. "
            "Explicit conversion to bytes is required" % (key, type(key)))
