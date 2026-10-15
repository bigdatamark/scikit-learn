# [MODULE DOC]
# Implementation of 256-value bitsets (8 x uint32 words) used by tree code,
# notably histogram gradient boosting to store categorical split sets. Bit for
# value v is bit (v mod 32) of word (v div 32). Also provides a Python-level
# helper that maps a bitset over binned category codes to raw category values.
"""
A bitset is a data structure used to represent sets of integers in [0, n]. For decision
trees, we use them to represent sets of features indices (e.g. features that go to the
left child, or features that are categorical). For familiarity with bitsets and bitwise
operations see:
https://en.wikipedia.org/wiki/Bit_array
https://en.wikipedia.org/wiki/Bitwise_operation
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


cdef inline void init_bitset(BITSET_DTYPE_C bitset) noexcept nogil:
    # [FUNCTION DOC]
    # Behavior: set all 8 words of bitset to 0.
    # Inputs: bitset, a fixed array of 8 uint32 words.
    # Outputs: none (bitset written in place). Exceptions: none (nogil).
    cdef:
        unsigned int i

    for i in range(8):
        bitset[i] = 0


cdef inline void set_bitset(BITSET_DTYPE_C bitset, uint8_t val) noexcept nogil:
    # [FUNCTION DOC]
    # Behavior: OR the bit (val mod 32) into word (val div 32), adding val to the set.
    # Inputs: bitset (8 words), val in 0..255. Outputs: bitset mutated in place.
    # Exceptions: none (nogil, noexcept).
    bitset[val // 32] |= (1 << (val % 32))


cdef inline uint8_t in_bitset(BITSET_DTYPE_C bitset, uint8_t val) noexcept nogil:
    # [FUNCTION DOC]
    # Behavior: return bit (val mod 32) of word (val div 32), i.e. 1 if val is in
    # the set, 0 otherwise. Inputs: bitset, val in 0..255. Exceptions: none.
    return (bitset[val // 32] >> (val % 32)) & 1


cpdef inline uint8_t in_bitset_memoryview(
    const BITSET_INNER_DTYPE_C[:] bitset, uint8_t val
) noexcept nogil:
    # [FUNCTION DOC]
    # Behavior: same membership test as in_bitset but on a 1d memoryview of words
    # (cpdef, so also usable from Python). Inputs: bitset memoryview of at least
    # val div 32 + 1 words, val. Outputs: 1 or 0. Exceptions: none; no bounds
    # checking is relied upon.
    return (bitset[val // 32] >> (val % 32)) & 1


cdef inline uint8_t in_bitset_2d_memoryview(
    const BITSET_INNER_DTYPE_C[:, :] bitset, uint8_t val, unsigned int row
) noexcept nogil:
    # [FUNCTION DOC]
    # Behavior: membership test of val in the bitset stored in row `row` of a 2d
    # memoryview (rows are bitsets). Outputs: 1 or 0. Exceptions: none. Side
    # effects: none.
    # Same as above but works on 2d memory views to avoid the creation of 1d
    # memory views. See https://github.com/scikit-learn/scikit-learn/issues/17299
    return (bitset[row, val // 32] >> (val % 32)) & 1


cpdef inline void set_bitset_memoryview(BITSET_INNER_DTYPE_C[:] bitset, uint8_t val):
    # [FUNCTION DOC]
    # Behavior: add val to a bitset given as a 1d memoryview of words (sets the bit).
    # Inputs: writable memoryview bitset, val in 0..255.
    # Outputs: bitset mutated in place. Exceptions: none expected.
    bitset[val // 32] |= (1 << (val % 32))


def set_raw_bitset_from_binned_bitset(
    BITSET_INNER_DTYPE_C[:] raw_bitset,
    BITSET_INNER_DTYPE_C[:] binned_bitset,
    float64_t[:] categories,
):
    # [FUNCTION DOC]
    # Behavior: for each binned category index i (position in categories), if i is
    # in binned_bitset, add the raw value categories[i] (cast to uint8) to
    # raw_bitset. Translates a split expressed on bin codes into raw category codes.
    # Inputs: raw_bitset (writable words, typically zero-initialised by the caller),
    # binned_bitset (words), categories (float64 raw category value per bin).
    # Outputs: raw_bitset mutated in place; returns None.
    # Exceptions: none in practice; raw values above 255 would be truncated by the
    # uint8 cast (callers guarantee categories are < 256).
    """Set the raw_bitset from the values of the binned bitset

    categories is a mapping from binned category value to raw category value.
    """
    cdef:
        int binned_cat_value
        float64_t raw_cat_value

    for binned_cat_value, raw_cat_value in enumerate(categories):
        if in_bitset_memoryview(binned_bitset, binned_cat_value):
            set_bitset_memoryview(raw_bitset, <uint8_t>raw_cat_value)
