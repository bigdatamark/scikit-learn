# [MODULE DOC]
# Declares the OpenMP API subset used by scikit-learn through a C preprocessor
# shim: when compiled with OpenMP, omp.h is included and
# SKLEARN_OPENMP_PARALLELISM_ENABLED is 1; otherwise it is 0 and the lock and
# thread functions are replaced by no-op macros (thread num 0, max threads 1).
# Cython modules must cimport OpenMP routines from here.
# Helpers to safely access OpenMP routines
#
# no-op implementations are provided for the case where OpenMP is not available.
#
# All calls to OpenMP routines should be cimported from this module.

cdef extern from *:
    """
    #ifdef _OPENMP
        #include <omp.h>
        #define SKLEARN_OPENMP_PARALLELISM_ENABLED 1
    #else
        #define SKLEARN_OPENMP_PARALLELISM_ENABLED 0
        #define omp_lock_t int
        #define omp_init_lock(l) (void)0
        #define omp_destroy_lock(l) (void)0
        #define omp_set_lock(l) (void)0
        #define omp_unset_lock(l) (void)0
        #define omp_get_thread_num() 0
        #define omp_get_max_threads() 1
    #endif
    """
    bint SKLEARN_OPENMP_PARALLELISM_ENABLED

    ctypedef struct omp_lock_t:
        # [CLASS DOC]
        # Opaque OpenMP lock type (int when OpenMP is unavailable).
        pass

    void omp_init_lock(omp_lock_t*) noexcept nogil
    # [FUNCTION DOC]
    # Initialise an OpenMP lock (no-op without OpenMP).
    void omp_destroy_lock(omp_lock_t*) noexcept nogil
    # [FUNCTION DOC]
    # Destroy an OpenMP lock (no-op without OpenMP).
    void omp_set_lock(omp_lock_t*) noexcept nogil
    # [FUNCTION DOC]
    # Acquire an OpenMP lock, blocking (no-op without OpenMP).
    void omp_unset_lock(omp_lock_t*) noexcept nogil
    # [FUNCTION DOC]
    # Release an OpenMP lock (no-op without OpenMP).
    int omp_get_thread_num() noexcept nogil
    # [FUNCTION DOC]
    # Return the calling thread's index in the current team (0 without OpenMP).
    int omp_get_max_threads() noexcept nogil
    # [FUNCTION DOC]
    # Return the max number of threads a parallel region would use (1 without OpenMP).
