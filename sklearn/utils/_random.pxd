# [MODULE DOC]
# Declarations for a tiny, platform-independent pseudo-random generator used
# in nogil Cython code (trees, SGD, k-means init, etc.): DEFAULT_SEED,
# RAND_R_MAX (2^31 - 1) and the inline xorshift generator our_rand_r. The
# sampling-without-replacement functions live in _random.pyx.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.utils._typedefs cimport uint32_t


cdef const uint32_t DEFAULT_SEED = 1

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    # It corresponds to the maximum representable value for
    # 32-bit signed integers (i.e. 2^31 - 1).
    RAND_R_MAX = 2147483647


# rand_r replacement using a 32bit XorShift generator
# See http://www.jstatsoft.org/v08/i14/paper for details
cdef inline uint32_t our_rand_r(uint32_t* seed) nogil:
    # [FUNCTION DOC]
    # Behavior: 32-bit xorshift step (shifts 13 left, 17 right, 5 left, each XORed
    # into the state). A zero state is first replaced by DEFAULT_SEED since zero
    # is a fixed point. Returns the new state reduced modulo 2^31, i.e. a value in
    # [0, RAND_R_MAX].
    # Inputs: seed, pointer to the uint32 generator state.
    # Outputs: random uint32 in [0, 2^31 - 1]. Side effects: updates seed in place.
    # Exceptions: none (declared nogil without noexcept, so callers may perform an
    # exception check after each call, but nothing is ever raised).
    """Generate a pseudo-random np.uint32 from an np.uint32 seed"""
    # seed shouldn't ever be 0.
    if (seed[0] == 0):
        seed[0] = DEFAULT_SEED

    seed[0] ^= <uint32_t>(seed[0] << 13)
    seed[0] ^= <uint32_t>(seed[0] >> 17)
    seed[0] ^= <uint32_t>(seed[0] << 5)

    # Use the modulo to make sure that we don't return a values greater than the
    # maximum representable value for signed 32bit integers (i.e. 2^31 - 1).
    # Note that the parenthesis are needed to avoid overflow: here
    # RAND_R_MAX is cast to uint32_t before 1 is added.
    return seed[0] % ((<uint32_t>RAND_R_MAX) + 1)
