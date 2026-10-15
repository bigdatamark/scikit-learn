# [MODULE DOC]
# Declarations for the fixed-size 256-bit bitset used by the histogram gradient
# boosting trees (categorical splits) and other tree code. A bitset is an array
# of 8 unsigned 32-bit words; value v (0..255) is stored at bit (v mod 32) of
# word (v div 32). Implementations live in _bitset.pyx; all routines are nogil
# and cannot raise.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.utils._typedefs cimport float64_t, uint8_t, uint32_t

ctypedef uint32_t BITSET_INNER_DTYPE_C
ctypedef BITSET_INNER_DTYPE_C[8] BITSET_DTYPE_C

cdef void init_bitset(BITSET_DTYPE_C bitset) noexcept nogil
# [FUNCTION DOC]
# Clear all 8 words of the bitset to zero (empties the set). Mutates bitset.

cdef void set_bitset(BITSET_DTYPE_C bitset, uint8_t val) noexcept nogil
# [FUNCTION DOC]
# Add value val (0..255) to the bitset by setting its bit. Mutates bitset.

cdef uint8_t in_bitset(BITSET_DTYPE_C bitset, uint8_t val) noexcept nogil
# [FUNCTION DOC]
# Return 1 if val is in the bitset, else 0. No side effects.

cpdef uint8_t in_bitset_memoryview(
    const BITSET_INNER_DTYPE_C[:] bitset, uint8_t val
) noexcept nogil
# [FUNCTION DOC]
# Membership test (returns 1 or 0) on a bitset passed as a 1d memoryview of
# words; also callable from Python. Does not check the memoryview length.

cdef uint8_t in_bitset_2d_memoryview(
    const BITSET_INNER_DTYPE_C[:, :] bitset, uint8_t val, unsigned int row
) noexcept nogil
# [FUNCTION DOC]
# Membership test on row `row` of a 2d memoryview of bitsets (one bitset per
# row); avoids creating a 1d memoryview slice. Returns 1 or 0.
