# [MODULE DOC]
# Thin Python wrappers around the C++ random number generator in newrand.h,
# which is shared by the bundled libsvm and liblinear sources. The generator is a
# single process-global Mersenne Twister (mt19937) plus a Lemire-style bounded
# integer post-processor that replaces the C library rand() so results are
# identical across platforms. Exposed mainly so tests can seed and sample it.
"""Wrapper for newrand.h"""

cdef extern from "newrand.h":
    void set_seed(unsigned int)
    unsigned int bounded_rand_int(unsigned int)


def set_seed_wrap(unsigned int custom_seed):
    # [FUNCTION DOC]
    # Behavior: reseeds the global mt19937 generator used by libsvm/liblinear.
    # Inputs: custom_seed, unsigned 32-bit seed (Python ints outside range raise
    # OverflowError during argument conversion).
    # Outputs: none.
    # Exceptions: only argument conversion errors.
    # Side effects: mutates process-global RNG state shared with libsvm/liblinear.
    set_seed(custom_seed)


def bounded_rand_int_wrap(unsigned int range_):
    # [FUNCTION DOC]
    # Behavior: draws one uniform integer in [0, range_) from the global generator,
    # using Lemire's multiply-shift method with rejection to avoid modulo bias.
    # Inputs: range_, unsigned 32-bit upper bound (exclusive), must be > 0.
    # Outputs: the drawn integer as a Python int.
    # Exceptions: only argument conversion errors.
    # Side effects: advances the process-global RNG state.
    # KNOWN ISSUE: range_ equal to 0 is not rejected; the C code then always returns 0
    # instead of signalling an invalid range.
    return bounded_rand_int(range_)
