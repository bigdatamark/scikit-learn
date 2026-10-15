# [MODULE DOC]
# Cython-level declarations shared by all compiled modules of the histogram
# gradient boosting package: C typedefs mirroring the numpy dtypes defined in
# common.pyx (X_DTYPE_C float64 raw features, X_BINNED_DTYPE_C uint8 bin codes,
# Y_DTYPE_C float64 targets/sums/values, G_H_DTYPE_C float32 gradients and
# hessians), the packed structs hist_struct and node_struct that match
# HISTOGRAM_DTYPE and PREDICTOR_RECORD_DTYPE byte-for-byte so numpy structured
# arrays can be viewed as typed memoryviews, and the MonotonicConstraint enum.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.utils._bitset cimport BITSET_INNER_DTYPE_C, BITSET_INNER_DTYPE_C
from sklearn.utils._typedefs cimport float32_t, float64_t, intp_t, uint8_t


ctypedef float64_t X_DTYPE_C
ctypedef uint8_t X_BINNED_DTYPE_C
ctypedef float64_t Y_DTYPE_C
ctypedef float32_t G_H_DTYPE_C


cdef packed struct hist_struct:
    # [CLASS DOC]
    # Packed C struct identical in layout to HISTOGRAM_DTYPE: one histogram bin.
    # Fields: sum_gradients and sum_hessians (float64 sums over samples in the bin),
    # count (number of samples in the bin). Packed (no padding) because numpy
    # structured dtypes are unaligned by default.
    # Same as histogram dtype but we need a struct to declare views. It needs
    # to be packed since by default numpy dtypes aren't aligned
    Y_DTYPE_C sum_gradients
    Y_DTYPE_C sum_hessians
    unsigned int count


cdef packed struct node_struct:
    # [CLASS DOC]
    # Packed C struct identical in layout to PREDICTOR_RECORD_DTYPE: one node of a
    # fitted predictor tree. value is the leaf value (or node value), count the number
    # of training samples, feature_idx the split feature, num_threshold the raw-scale
    # threshold (go left if x <= threshold), missing_go_to_left the direction for NaN,
    # left/right child indices into the node array, gain the split gain, depth, is_leaf
    # flag, bin_threshold the binned-scale threshold, is_categorical flag, and
    # bitset_idx the row of the predictor's categorical bitset arrays (only meaningful
    # for categorical splits).
    # Equivalent struct to PREDICTOR_RECORD_DTYPE to use in memory views. It
    # needs to be packed since by default numpy dtypes aren't aligned
    Y_DTYPE_C value
    unsigned int count
    intp_t feature_idx
    X_DTYPE_C num_threshold
    uint8_t missing_go_to_left
    unsigned int left
    unsigned int right
    Y_DTYPE_C gain
    unsigned int depth
    uint8_t is_leaf
    X_BINNED_DTYPE_C bin_threshold
    uint8_t is_categorical
    # The index of the corresponding bitsets in the Predictor's bitset arrays.
    # Only used if is_categorical is True
    unsigned int bitset_idx


cpdef enum MonotonicConstraint:
    # [CLASS DOC]
    # Enumeration of monotonic constraints, also exposed to Python (cpdef enum):
    # NO_CST = 0 no constraint, POS = 1 increasing, NEG = -1 decreasing.
    NO_CST = 0
    POS = 1
    NEG = -1
