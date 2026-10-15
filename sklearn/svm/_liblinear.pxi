# [MODULE DOC]
# Include file (textually included by _liblinear.pyx) declaring the C/C++ API of
# the bundled liblinear library (linear.h), the sklearn helper layer
# (liblinear_helper.c, which converts numpy buffers into liblinear structures)
# and the BLAS function-pointer table (_cython_blas_helpers.h) through which
# liblinear calls scipy's BLAS. Contains declarations only; no executable code.
from sklearn.utils._typedefs cimport intp_t

cdef extern from "_cython_blas_helpers.h":
    ctypedef double (*dot_func)(int, const double*, int, const double*, int)
    ctypedef void (*axpy_func)(int, double, const double*, int, double*, int)
    ctypedef void (*scal_func)(int, double, const double*, int)
    ctypedef double (*nrm2_func)(int, const double*, int)
    cdef struct BlasFunctions:
        # [CLASS DOC]
        # Table of BLAS level-1 function pointers (dot, axpy, scal, nrm2, double
        # precision) filled by the caller and passed to liblinear's train so the C++
        # code uses the same BLAS that scipy links against.
        dot_func dot
        axpy_func axpy
        scal_func scal
        nrm2_func nrm2


cdef extern from "linear.h":
    cdef struct feature_node
    # [CLASS DOC]
    # Opaque liblinear sparse element: one (1-based feature index, value) pair; rows
    # are arrays of these terminated by an element with index -1.
    cdef struct problem
    # [CLASS DOC]
    # Opaque liblinear training problem: number of samples and features, label
    # array, sample weights, row pointers to feature_node arrays and the bias value.
    cdef struct model
    # [CLASS DOC]
    # Opaque trained liblinear model: parameters, class count, feature count, weight
    # vector w (column-major, features by classes), labels, bias and iteration counts.
    cdef struct parameter
    # [CLASS DOC]
    # Opaque liblinear solver parameters: solver type, tolerance, C, class weights,
    # epsilon for SVR and maximum iterations.
    ctypedef problem* problem_const_ptr "problem const *"
    ctypedef parameter* parameter_const_ptr "parameter const *"
    ctypedef char* char_const_ptr "char const *"
    char_const_ptr check_parameter(problem_const_ptr prob, parameter_const_ptr param)
    # [FUNCTION DOC]
    # Behavior: validates a problem/parameter pair (solver type, positive eps/C/p,
    # compatibility of options).
    # Inputs: prob and param pointers, must be non-NULL.
    # Outputs: NULL if valid, else a pointer to a static error message string.
    # Exceptions: none (C function). Side effects: none.
    model *train(problem_const_ptr prob, parameter_const_ptr param, BlasFunctions *blas_functions) nogil
    # [FUNCTION DOC]
    # Behavior: runs the liblinear solver selected by param on prob.
    # Inputs: prob, param, and a filled BlasFunctions table.
    # Outputs: newly allocated model which the caller must free with
    # free_and_destroy_model.
    # Exceptions: none; declared nogil so it can run without the GIL.
    # Side effects: allocates memory; uses the global newrand RNG; may print progress
    # through the configured print function.
    int get_nr_feature (model *model)
    # [FUNCTION DOC]
    # Returns the number of features stored in the model (excluding the bias column).
    # Inputs: non-NULL model. Exceptions/side effects: none.
    int get_nr_class (model *model)
    # [FUNCTION DOC]
    # Returns the number of classes stored in the model (nr_class; 2 for binary
    # problems). Inputs: non-NULL model. No side effects.
    void get_n_iter (model *model, int *n_iter)
    # [FUNCTION DOC]
    # Behavior: copies per-class (or single, for binary) iteration counts from the
    # model into n_iter.
    # Inputs: model; n_iter, caller-allocated int buffer long enough for the counts.
    # Outputs: written into n_iter. Exceptions: none.
    void free_and_destroy_model (model **)
    # [FUNCTION DOC]
    # Frees a model and its arrays and sets the pointed-to pointer to NULL.
    # Inputs: address of a model pointer. Side effects: frees memory.
    void destroy_param (parameter *)
    # [FUNCTION DOC]
    # Frees the weight_label and weight arrays owned by a parameter struct. Not used
    # by sklearn because those arrays belong to numpy buffers.
    # Side effects: frees memory (would corrupt numpy-owned memory if called here).


cdef extern from "liblinear_helper.c":
    void copy_w(void *, model *, int)
    # [FUNCTION DOC]
    # Behavior: copies the first len doubles of the model's weight vector into data.
    # Inputs: data, destination buffer of at least len doubles; model; len count.
    # Outputs: written into data. Exceptions: none.
    parameter *set_parameter(int, double, double, int, char *, char *, int, int, double)
    # [FUNCTION DOC]
    # Behavior: allocates a parameter struct and fills solver type, eps, C, number of
    # class weights, weight label and weight pointers (borrowed, not copied), maximum
    # iterations and SVR epsilon; also reseeds the global newrand RNG with the seed.
    # Inputs: solver_type, eps, C, nr_weight, weight_label (int array or NULL),
    # weight (double array or NULL), max_iter, seed, epsilon.
    # Outputs: new parameter pointer, or NULL if malloc fails.
    # Side effects: allocation; mutates global RNG state.
    problem *set_problem (char *, int, int, int, int, double, char *, char *)
    # [FUNCTION DOC]
    # Behavior: builds a liblinear problem from a dense C-contiguous matrix by
    # converting it to liblinear's sparse format in one allocated block, skipping
    # zeros, using 1-based indices, appending a bias feature when bias > 0 and an
    # index -1 sentinel per row.
    # Inputs: X data pointer, flag for float64 (else float32), n_samples, n_features,
    # number of nonzeros, bias, sample_weight pointer (float64), Y pointer (float64).
    # Outputs: new problem pointer (y and W borrowed from the caller), or NULL if any
    # allocation fails. Side effects: allocation; free with free_problem.
    problem *csr_set_problem (char *, int, char *, char *, int, int, int, double, char *, char *)
    # [FUNCTION DOC]
    # Behavior: same as set_problem but from CSR arrays (data, int32 indices and
    # indptr), shifting column indices to 1-based and adding the bias feature.
    # Inputs: data pointer, float64 flag, indices, indptr, n_samples, n_features, nnz,
    # bias, sample_weight, Y.
    # Outputs: new problem pointer or NULL on allocation failure.

    model *set_model(parameter *, char *, intp_t *, char *, double)
    # [FUNCTION DOC]
    # Declared helper to rebuild a model from arrays.
    # KNOWN ISSUE: no set_model function is defined in liblinear_helper.c; the
    # declaration is unused and would fail to link if called.

    double get_bias(model *)
    # [FUNCTION DOC]
    # Returns the bias value stored in the model. Unused by _liblinear.pyx.
    void free_problem (problem *)
    # [FUNCTION DOC]
    # Frees a problem built by set_problem/csr_set_problem: the feature_node block,
    # the row pointer array and the struct (not y or W, which are borrowed).
    void free_parameter (parameter *)
    # [FUNCTION DOC]
    # Frees only the parameter struct itself (not the borrowed weight arrays).
    void set_verbosity(int)
    # [FUNCTION DOC]
    # Behavior: installs liblinear's global print function: stdout with flushing if
    # verbosity is nonzero, otherwise a no-op printer.
    # Side effects: mutates process-global liblinear state.
