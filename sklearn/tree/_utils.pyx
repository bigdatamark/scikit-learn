# [MODULE DOC]
# Low-level helpers shared across the tree package (and the quad tree in
# sklearn.neighbors): overflow-checked realloc for C buffers, a copy helper
# from C intp arrays to numpy, portable random integer/float draws built on
# our_rand_r (independent of platform RAND_MAX), base-2 log, and the
# WeightedFenwickTree used for incremental weighted quantile queries.
# PytestWeightedFenwickTree and _realloc_test are Python-visible test hooks.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from libc.stdlib cimport free
from libc.stdlib cimport realloc
from libc.math cimport log as ln
from libc.string cimport memset

cimport numpy as cnp
cnp.import_array()

from sklearn.utils._random cimport our_rand_r

# =============================================================================
# Helper functions
# =============================================================================

cdef int safe_realloc(realloc_ptr* p, size_t nelems) except -1 nogil:
    # [FUNCTION DOC]
    # HELPER: overflow-checked realloc for the fused pointer types in _utils.pxd.
    # Behavior: computes nelems times element size, detects overflow by dividing
    # back, then calls realloc on p[0]; on success stores the new pointer.
    # Inputs: p, address of the pointer to grow (may hold NULL); nelems, count.
    # Outputs: returns 0 and updates p[0].
    # Exceptions: MemoryError on overflow or realloc failure (returns -1, the
    # GIL is acquired to raise); the old block stays valid and owned by caller.
    # Side effects: heap reallocation; never frees memory.
    # sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
    # 0.20.1 to crash.
    cdef size_t nbytes = nelems * sizeof(p[0][0])
    if nbytes / sizeof(p[0][0]) != nelems:
        # Overflow in the multiplication
        raise MemoryError(f"could not allocate ({nelems} * {sizeof(p[0][0])}) bytes")

    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)
    if tmp == NULL:
        raise MemoryError(f"could not allocate {nbytes} bytes")

    p[0] = tmp
    return 0


def _realloc_test():
    # [FUNCTION DOC]
    # HELPER: test hook checking that safe_realloc detects size overflow.
    # Behavior: requests half of the maximum size_t elements of intp (the byte
    # count overflows); if allocation somehow succeeds, frees and fails.
    # Inputs: none. Outputs: none.
    # Exceptions: MemoryError from safe_realloc (expected by tests);
    # AssertionError if no error was raised.
    # Side effects: none on success path.
    # Helper for tests. Tries to allocate <size_t>(-1) / 2 * sizeof(size_t)
    # bytes, which will always overflow.
    cdef intp_t* p = NULL
    safe_realloc(&p, <size_t>(-1) / 2)
    if p != NULL:
        free(p)
        assert False


cdef inline cnp.ndarray sizet_ptr_to_ndarray(intp_t* data, intp_t size):
    # [FUNCTION DOC]
    # HELPER: copies a C intp buffer into a new numpy array.
    # Behavior: wraps data as a 1-D NPY_INTP array without copying, then
    # returns a copy so the result owns its memory.
    # Inputs: data, pointer to size intp values; size, element count.
    # Outputs: new 1-D intp ndarray. Exceptions: MemoryError from numpy.
    # Side effects: allocation; needs the GIL.
    """Return copied data as 1D numpy array of intp's."""
    cdef cnp.npy_intp shape[1]
    shape[0] = <cnp.npy_intp> size
    return cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_INTP, data).copy()


cdef inline intp_t rand_int(intp_t low, intp_t high,
                            uint32_t* random_state) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: random integer in [low, high) for splitter feature sampling.
    # Behavior: low plus our_rand_r(state) modulo (high - low).
    # Inputs: low < high; random_state, pointer to 32-bit state.
    # Outputs: the integer. Exceptions: none (nogil); low == high would divide
    # by zero. Side effects: advances random_state[0].
    """Generate a random integer in [low; end)."""
    return low + our_rand_r(random_state) % (high - low)


cdef inline float64_t rand_uniform(float64_t low, float64_t high,
                                   uint32_t* random_state) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: random float between low and high for random thresholds.
    # Behavior: scales our_rand_r(state) / RAND_R_MAX to the interval.
    # Inputs: low, high; random_state, pointer to 32-bit state.
    # Outputs: float in [low, high]; high is reachable if the generator returns
    # RAND_R_MAX (callers such as node_split_random handle this case).
    # Exceptions: none (nogil). Side effects: advances random_state[0].
    # KNOWN ISSUE: the docstring says the interval is half-open [low; high) but
    # the result equals high when our_rand_r returns RAND_R_MAX.
    """Generate a random float64_t in [low; high)."""
    return ((high - low) * <float64_t> our_rand_r(random_state) /
            <float64_t> RAND_R_MAX) + low


cdef inline float64_t log(float64_t x) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: base-2 logarithm, ln(x) / ln(2), used by entropy criteria.
    # Inputs: x > 0. Outputs: log2(x). Exceptions: none (nogil); x <= 0 gives
    # -inf/NaN. Side effects: none.
    return ln(x) / ln(2.0)


cdef class WeightedFenwickTree:
    # [CLASS DOC]
    # Fenwick tree with two 1-based buffers (weights and weight * y) of length
    # capacity + 1, allocated in __cinit__ and freed in __dealloc__. A user calls
    # reset(size) once per use (size <= capacity), then add(rank, y, w) for each
    # inserted item, and search(t) to locate the rank where the running weight
    # reaches t together with prefix sums. Totals of weight and weighted y are
    # also maintained. All operations except allocation are O(log size), nogil.
    """
    Fenwick tree (Binary Indexed Tree) specialized for maintaining:
      - prefix sums of weights
      - prefix sums of weight * target (y)

    Notes:
      - Implementation uses 1-based indexing internally for the Fenwick tree
        arrays, hence the +1 sized buffers. 1-based indexing is customary for this
        data structure and makes the some index handling slightly more efficient and
        natural.
      - Memory ownership: this class allocates and frees the underlying C buffers.
      - Typical operations:
          add(rank, y, w) -> O(log n)
          search(t)       -> O(log n), finds the smallest rank with
                             cumulative weight > t (see search for details).
    """

    def __cinit__(self, intp_t capacity):
        # [FUNCTION DOC]
        # Allocates both Fenwick buffers with capacity + 1 float64 entries each.
        # Inputs: capacity, maximum number of ranks reset may later request.
        # Outputs: none. Exceptions: MemoryError from safe_realloc.
        # Side effects: heap allocation (contents uninitialized until reset).
        self.tree_w = NULL
        self.tree_wy = NULL

        # Allocate arrays of length (capacity + 1) because indices are 1-based.
        safe_realloc(&self.tree_w, capacity + 1)
        safe_realloc(&self.tree_wy, capacity + 1)

    cdef void reset(self, intp_t size) noexcept nogil:
        # [FUNCTION DOC]
        # Behavior: sets the logical size, zeroes entries 0..size of both buffers
        # and the totals, and computes max_pow2, the largest power of two <= size
        # (0 when size is 0) used as the starting step of search.
        # Inputs: size, number of ranks (precondition: size <= capacity; not checked,
        # a larger value writes past the buffers).
        # Outputs: none. Exceptions: none (nogil, noexcept).
        # Side effects: mutates the buffers and fields in place.
        """
        Reset the tree to hold 'size' elements and clear all aggregates.
        """
        cdef intp_t p
        cdef intp_t n_bytes = (size + 1) * sizeof(float64_t)  # +1 for 1-based storage

        # Public size and zeroed aggregates.
        self.size = size
        memset(self.tree_w, 0, n_bytes)
        memset(self.tree_wy, 0, n_bytes)
        self.total_w = 0.0
        self.total_wy = 0.0

        # highest power of two <= size
        p = 1
        while p <= size:
            p <<= 1
        self.max_pow2 = p >> 1

    def __dealloc__(self):
        # [FUNCTION DOC]
        # Frees both buffers if allocated. Inputs/outputs: none. Exceptions: none.
        # Side effects: releases heap memory.
        if self.tree_w != NULL:
            free(self.tree_w)
        if self.tree_wy != NULL:
            free(self.tree_wy)

    cdef void add(self, intp_t idx, float64_t y_value, float64_t weight) noexcept nogil:
        # [FUNCTION DOC]
        # Behavior: standard Fenwick point update: starting from 1-based position
        # idx + 1, add weight and weight * y_value to the node, then move to the next
        # responsible node by adding the lowest set bit, until beyond size. Totals
        # are updated too.
        # Inputs: idx, 0-based rank in [0, size); y_value, target; weight, weight.
        # Outputs: none. Exceptions: none (nogil, noexcept).
        # Side effects: mutates tree_w, tree_wy, total_w, total_wy.
        """
        Add a weighted observation to the Fenwick tree.

        Parameters
        ----------
        idx : intp_t
            The 0-based index where to add the observation
        y_value : float64_t
            The target value (y) of the observation
        weight : float64_t
            The sample weight

        Notes
        -----
        Updates both weight sums and weighted target sums in O(log n) time.
        """
        cdef float64_t weighted_y = weight * y_value
        cdef intp_t fenwick_idx = idx + 1  # Convert to 1-based indexing

        # Update Fenwick tree nodes by traversing up the tree
        while fenwick_idx <= self.size:
            self.tree_w[fenwick_idx] += weight
            self.tree_wy[fenwick_idx] += weighted_y
            # Move to next node using bit manipulation: add lowest set bit
            fenwick_idx += fenwick_idx & -fenwick_idx

        # Update global totals
        self.total_w += weight
        self.total_wy += weighted_y

    cdef intp_t search(
        self,
        float64_t target_weight,
        float64_t* cumul_weight_out,
        float64_t* cumul_weighted_y_out,
        intp_t* prev_idx_out,
    ) noexcept nogil:
        # [FUNCTION DOC]
        # Behavior: top-down Fenwick descent from bit max_pow2. Phase 1 consumes a
        # node whenever the remaining target is strictly greater than its weight,
        # accumulating weight and weighted-y prefixes; if no node weight equals the
        # remaining target exactly, the reached position is both the answer and
        # prev_idx. If an exact equality is met at some step, the state is saved:
        # phase 2 continues the strict descent from there to find prev_idx (largest
        # prefix length whose weight is strictly below the target); phase 3 restarts
        # from the saved step with a non-strict comparison to find the largest
        # prefix length whose weight is <= target, which gives the returned index and
        # the accumulated prefix sums. Ties caused by zero-weight ranks are thus
        # skipped over.
        # Inputs: target_weight t, expected in [0, total_w]; three output pointers.
        # Outputs: returns idx, a 0-based rank such that the weight of ranks
        # [0, idx) is <= t (strictly < t outside the exact-match case) and rank idx
        # pushes it to at least t. cumul_weight_out / cumul_weighted_y_out receive
        # the weight and weighted-y sums over ranks [0, idx); prev_idx_out receives
        # the largest prefix length with weight strictly below t (equal to idx
        # unless an exact match occurred).
        # Exceptions: none (nogil, noexcept).
        # Side effects: writes the three output pointers only.
        """
        Binary search to find the position where cumulative weight reaches target.

        This method performs a binary search on the Fenwick tree to find indices
        such that the cumulative weight at 'prev_idx' is < target_weight and
        the cumulative weight at the returned index is >= target_weight.

        Parameters
        ----------
        target_weight : float64_t
            The target cumulative weight to search for
        cumul_weight_out : float64_t*
            Output pointer for cumulative weight up to returned index (exclusive)
        cumul_weighted_y_out : float64_t*
            Output pointer for cumulative weighted y-sum up to returned index (exclusive)
        prev_idx_out : intp_t*
            Output pointer for the previous index (largest index with cumul_weight < target)

        Returns
        -------
        intp_t
            The index where cumulative weight first reaches or exceeds target_weight

        Notes
        -----
        - O(log n) complexity
        - Ignores nodes with zero weights (corresponding to uninserted y-values)
        - Assumes at least one active (positive-weight) item exists
        - Assumes 0 <= target_weight <= total_weight
        """
        cdef:
            intp_t current_idx = 0
            intp_t next_idx, prev_idx, equal_bit
            float64_t cumul_weight = 0.0
            float64_t cumul_weighted_y = 0.0
            intp_t search_bit = self.max_pow2  # Start from highest power of 2
            float64_t node_weight, equal_target

        # Phase 1: Standard Fenwick binary search with prefix accumulation
        # Traverse down the tree, moving right when we can consume more weight
        while search_bit != 0:
            next_idx = current_idx + search_bit
            if next_idx <= self.size:
                node_weight = self.tree_w[next_idx]
                if target_weight == node_weight:
                    # Exact match found - store state for later processing
                    equal_target = target_weight
                    equal_bit = search_bit
                    break
                elif target_weight > node_weight:
                    # We can consume this node's weight - move right and accumulate
                    target_weight -= node_weight
                    current_idx = next_idx
                    cumul_weight += node_weight
                    cumul_weighted_y += self.tree_wy[next_idx]
            search_bit >>= 1

        # If no exact match, we're done with standard search
        if search_bit == 0:
            cumul_weight_out[0] = cumul_weight
            cumul_weighted_y_out[0] = cumul_weighted_y
            prev_idx_out[0] = current_idx
            return current_idx

        # Phase 2: Handle exact match case - find prev_idx
        # Search for the largest index with cumulative weight < original target
        prev_idx = current_idx
        while search_bit != 0:
            next_idx = prev_idx + search_bit
            if next_idx <= self.size:
                node_weight = self.tree_w[next_idx]
                if target_weight > node_weight:
                    target_weight -= node_weight
                    prev_idx = next_idx
            search_bit >>= 1

        # Phase 3: Complete the exact match search
        # Restore state and search for the largest index with
        # cumulative weight <= original target (and this is case, we know we have ==)
        search_bit = equal_bit
        target_weight = equal_target
        while search_bit != 0:
            next_idx = current_idx + search_bit
            if next_idx <= self.size:
                node_weight = self.tree_w[next_idx]
                if target_weight >= node_weight:
                    target_weight -= node_weight
                    current_idx = next_idx
                    cumul_weight += node_weight
                    cumul_weighted_y += self.tree_wy[next_idx]
            search_bit >>= 1

        # Output results
        cumul_weight_out[0] = cumul_weight
        cumul_weighted_y_out[0] = cumul_weighted_y
        prev_idx_out[0] = prev_idx
        return current_idx


cdef class PytestWeightedFenwickTree(WeightedFenwickTree):
    # [CLASS DOC]
    # Test-only subclass exposing reset, add and search to Python.
    """Used for testing only"""

    def py_reset(self, intp_t n):
        # [FUNCTION DOC]
        # Test wrapper: calls reset(n). No return, no exceptions beyond reset's.
        self.reset(n)

    def py_add(self, intp_t idx, float64_t y, float64_t w):
        # [FUNCTION DOC]
        # Test wrapper: calls add(idx, y, w). No return value.
        self.add(idx, y, w)

    def py_search(self, float64_t t):
        # [FUNCTION DOC]
        # Test wrapper: runs search(t) and returns a tuple
        # (prev_idx, idx, cumulative weight, cumulative weighted y).
        cdef float64_t w, wy
        cdef intp_t prev_idx
        idx = self.search(t, &w, &wy, &prev_idx)
        return prev_idx, idx, w, wy
