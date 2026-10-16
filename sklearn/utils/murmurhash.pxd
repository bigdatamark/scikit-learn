# [MODULE DOC]
# Cython-level declarations of the MurmurHash3 (x86, 32-bit) wrappers so other
# Cython modules (e.g. feature_extraction._hashing_fast) can hash ints and byte
# strings without Python overhead. Implementations in murmurhash.pyx.
"""Export fast murmurhash C/C++ routines + cython wrappers"""

from sklearn.utils._typedefs cimport int32_t, uint32_t

# The C API is disabled for now, since it requires -I flags to get
# compilation to work even when these functions are not used.
# cdef extern from "MurmurHash3.h":
#     void MurmurHash3_x86_32(void* key, int len, unsigned int seed,
#                             void* out)
#
#     void MurmurHash3_x86_128(void* key, int len, unsigned int seed,
#                              void* out)
#
#     void MurmurHash3_x64_128(void* key, int len, unsigned int seed,
#                              void* out)


cpdef uint32_t murmurhash3_int_u32(int key, unsigned int seed)
# [FUNCTION DOC]
# Unsigned 32-bit MurmurHash3 of the native int key with the given seed.
cpdef int32_t murmurhash3_int_s32(int key, unsigned int seed)
# [FUNCTION DOC]
# Same hash as murmurhash3_int_u32, reinterpreted as signed 32-bit.
cpdef uint32_t murmurhash3_bytes_u32(bytes key, unsigned int seed)
# [FUNCTION DOC]
# Unsigned 32-bit MurmurHash3 of the bytes key with the given seed.
cpdef int32_t murmurhash3_bytes_s32(bytes key, unsigned int seed)
# [FUNCTION DOC]
# Same hash as murmurhash3_bytes_u32, reinterpreted as signed 32-bit.
