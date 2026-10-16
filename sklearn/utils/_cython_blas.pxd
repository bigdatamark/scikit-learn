# [MODULE DOC]
# Declarations of thin typed wrappers around SciPy's Cython BLAS bindings,
# fused over float and double (the `floating` type). Defines the BLAS_Order
# (row/column major) and BLAS_Trans (no transpose / transpose, encoded as the
# ASCII codes of 'n' and 't') enums. Implementations are in _cython_blas.pyx.
# All wrappers are nogil and noexcept.
from cython cimport floating


cpdef enum BLAS_Order:
    RowMajor  # C contiguous
    ColMajor  # Fortran contiguous


cpdef enum BLAS_Trans:
    NoTrans = 110  # correspond to 'n'
    Trans = 116    # correspond to 't'


# BLAS Level 1 ################################################################
cdef floating _dot(int, const floating*, int, const floating*, int) noexcept nogil
# [FUNCTION DOC]
# Dot product of two strided vectors (BLAS dot): n, x, incx, y, incy. Returns scalar.

cdef floating _asum(int, const floating*, int) noexcept nogil
# [FUNCTION DOC]
# Sum of absolute values of a strided vector (BLAS asum): n, x, incx.

cdef void _axpy(int, floating, const floating*, int, floating*, int) noexcept nogil
# [FUNCTION DOC]
# y := alpha*x + y (BLAS axpy): n, alpha, x, incx, y, incy. Mutates y.

cdef floating _nrm2(int, const floating*, int) noexcept nogil
# [FUNCTION DOC]
# Euclidean norm of a strided vector (BLAS nrm2): n, x, incx.

cdef void _copy(int, const floating*, int, const floating*, int) noexcept nogil
# [FUNCTION DOC]
# Copy x into y (BLAS copy): n, x, incx, y, incy. Mutates y even though y is
# declared const.

cdef void _scal(int, floating, const floating*, int) noexcept nogil
# [FUNCTION DOC]
# x := alpha*x (BLAS scal): n, alpha, x, incx. Mutates x even though declared const.

cdef void _rotg(floating*, floating*, floating*, floating*) noexcept nogil
# [FUNCTION DOC]
# Construct a Givens plane rotation (BLAS rotg) from a, b; writes a, b, c, s.

cdef void _rot(int, floating*, int, floating*, int, floating, floating) noexcept nogil
# [FUNCTION DOC]
# Apply a plane rotation with cosine c and sine s to strided vectors x and y
# (BLAS rot). Mutates x and y.

# BLAS Level 2 ################################################################
cdef void _gemv(BLAS_Order, BLAS_Trans, int, int, floating, const floating*, int,
                const floating*, int, floating, floating*, int) noexcept nogil
# [FUNCTION DOC]
# y := alpha*op(A)*x + beta*y (BLAS gemv) with order, trans, m, n, alpha, A,
# lda, x, incx, beta, y, incy. Mutates y.

cdef void _ger(BLAS_Order, int, int, floating, const floating*, int, const floating*,
               int, floating*, int) noexcept nogil
# [FUNCTION DOC]
# A := alpha*x*y^T + A (BLAS ger, rank-1 update) with order, m, n, alpha, x,
# incx, y, incy, A, lda. Mutates A.

# BLASLevel 3 ################################################################
cdef void _gemm(BLAS_Order, BLAS_Trans, BLAS_Trans, int, int, int, floating,
                const floating*, int, const floating*, int, floating, floating*,
                int) noexcept nogil
# [FUNCTION DOC]
# C := alpha*op(A)*op(B) + beta*C (BLAS gemm) with order, transA, transB, m, n,
# k, alpha, A, lda, B, ldb, beta, C, ldc. Mutates C.
