# [MODULE DOC]
# Declarations for the low-level helpers shared by the tree package
# (_tree, _splitter, _partitioner, _criterion) and by
# sklearn.neighbors._quad_tree. Declares: the RAND_R_MAX constant (2^31 - 1,
# upper bound of the portable random generator our_rand_r), the fused pointer
# type realloc_ptr accepted by safe_realloc, the random helpers rand_int and
# rand_uniform, a base-2 log, a pointer-to-ndarray copy helper, and the
# WeightedFenwickTree class (prefix sums of weights and weighted targets,
# used by criteria that need weighted quantiles / medians).
# Implementations live in _utils.pyx.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# See _utils.pyx for details.

cimport numpy as cnp
from sklearn.tree._tree cimport Node
from sklearn.neighbors._quad_tree cimport Cell
from sklearn.utils._typedefs cimport float32_t, float64_t, intp_t, uint8_t, uint32_t


cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    # It corresponds to the maximum representable value for
    # 32-bit signed integers (i.e. 2^31 - 1).
    RAND_R_MAX = 2147483647


# safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
# raises a MemoryError. It never calls free, since that's __dealloc__'s job.
#   cdef float32_t *p = NULL
#   safe_realloc(&p, n)
# is equivalent to p = malloc(n * sizeof(*p)) with error checking.
ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (float32_t*)
    (intp_t*)
    (uint8_t*)
    (float64_t*)
    (float64_t**)
    (Node*)
    (Cell*)
    (Node**)

cdef int safe_realloc(realloc_ptr* p, size_t nelems) except -1 nogil
# [FUNCTION DOC]
# HELPER: checked reallocation used for every growable C buffer in the
# tree code (node arrays, value arrays, Fenwick buffers, quad tree cells).
# Behavior: resizes the block pointed to by p[0] to nelems elements of the
# pointee type; checks the byte count multiplication for overflow first.
# Inputs: p, address of a pointer (may point to NULL for a fresh allocation);
# nelems, number of elements wanted.
# Outputs: returns 0 on success and stores the new block in p[0].
# Exceptions: MemoryError on overflow or allocation failure (re-acquires the
# GIL to raise); returns -1 in that case and p[0] is left unchanged.
# Side effects: heap (re)allocation; never frees, the owner frees later.


cdef cnp.ndarray sizet_ptr_to_ndarray(intp_t* data, intp_t size)
# [FUNCTION DOC]
# HELPER: wraps a C array of intp values into a new numpy array.
# Inputs: data, pointer to size intp values; size, element count.
# Outputs: a freshly owned 1-D intp ndarray holding a copy of the data.
# Exceptions: MemoryError from numpy if the copy cannot be allocated.
# Side effects: allocates a Python object; requires the GIL.


cdef intp_t rand_int(intp_t low, intp_t high,
                     uint32_t* random_state) noexcept nogil
# [FUNCTION DOC]
# HELPER: uniform random integer for feature sampling in the splitters.
# Inputs: low, high bounds (low < high required); random_state, pointer to a
# 32-bit generator state.
# Outputs: an integer in [low, high), via modulo of our_rand_r (slightly
# biased when the range does not divide 2^31).
# Exceptions: none (nogil, noexcept); high == low divides by zero.
# Side effects: advances random_state[0].


cdef float64_t rand_uniform(float64_t low, float64_t high,
                            uint32_t* random_state) noexcept nogil
# [FUNCTION DOC]
# HELPER: uniform random float, used to draw thresholds in random splitters.
# Inputs: low, high bounds; random_state, pointer to generator state.
# Outputs: low + (high - low) * r / RAND_R_MAX with r from our_rand_r; the
# value lies in [low, high] (high reachable when r equals RAND_R_MAX).
# Exceptions: none (nogil, noexcept).
# Side effects: advances random_state[0].


cdef float64_t log(float64_t x) noexcept nogil
# [FUNCTION DOC]
# HELPER: base-2 logarithm used by entropy computations in the criteria.
# Inputs: x, positive float. Outputs: natural log of x divided by natural
# log of 2. Exceptions: none (nogil); x <= 0 gives -inf or NaN.
# Side effects: none.


cdef class WeightedFenwickTree:
    # [CLASS DOC]
    # Fenwick tree (binary indexed tree) over ranks 0..size-1 that keeps two
    # parallel prefix-sum arrays: sum of weights and sum of weight * y. Supports
    # point insertion and a weighted search for the rank where the running
    # weight reaches a target (used to find weighted medians/quantiles
    # incrementally). Buffers are 1-based and sized capacity + 1; fields hold the
    # current logical size, the two buffers, the largest power of two not above
    # size (start bit for the search descent) and the running totals.
    cdef intp_t size         # number of leaves (ranks)
    cdef float64_t* tree_w   # BIT for weights
    cdef float64_t* tree_wy  # BIT for weighted targets
    cdef intp_t max_pow2     # highest power of two <= n
    cdef float64_t total_w   # running total weight
    cdef float64_t total_wy  # running total weighted target

    cdef void reset(self, intp_t size) noexcept nogil
    # [FUNCTION DOC]
    # Declaration of reset: set the logical size and zero all aggregates.
    # See _utils.pyx for details.
    cdef void add(self, intp_t idx, float64_t y, float64_t w) noexcept nogil
    # [FUNCTION DOC]
    # Declaration of add: insert (y, w) at a 0-based rank. See _utils.pyx.
    cdef intp_t search(
        self,
        float64_t t,
        float64_t* cw_out,
        float64_t* cwy_out,
        intp_t* prev_idx_out,
    ) noexcept nogil
    # [FUNCTION DOC]
    # Declaration of search: weighted descent to the rank where cumulative
    # weight reaches t, also returning prefix sums and previous rank.
    # See _utils.pyx for details.
