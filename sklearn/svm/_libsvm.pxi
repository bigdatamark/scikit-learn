# [MODULE DOC]
# Include file (textually included by _libsvm.pyx) declaring the dense-input API
# of the bundled libsvm fork (svm.h), the sklearn helper layer libsvm_helper.c
# (converts numpy buffers to libsvm structures and copies model fields back) and
# the BLAS function-pointer table used for kernel dot products. Declarations
# only; no executable code.
################################################################################
# Includes
from ..utils._typedefs cimport intp_t

cdef extern from "_svm_cython_blas_helpers.h":
    ctypedef double (*dot_func)(int, const double*, int, const double*, int)
    cdef struct BlasFunctions:
        # [CLASS DOC]
        # Table holding a pointer to a double-precision BLAS dot routine, filled by the
        # caller and passed into libsvm so kernel evaluations use scipy's BLAS.
        dot_func dot


cdef extern from "svm.h":
    cdef struct svm_node
    # [CLASS DOC]
    # Opaque dense libsvm node: one sample as a pointer to its feature values, the
    # number of features and the sample's row index (used for precomputed kernels).
    cdef struct svm_model
    # [CLASS DOC]
    # Opaque libsvm model: parameters, class count, support vectors, dual
    # coefficients, rho (negated intercepts), probability parameters, support
    # indices, per-class support counts and iteration counts.
    cdef struct svm_parameter:
        # [CLASS DOC]
        # libsvm hyper-parameter struct (fields visible to Cython): SVM type, kernel type
        # and its degree/gamma/coef0, cache size in MB, tolerance eps, C, class weights
        # (count, labels, values), nu, epsilon-SVR p, shrinking and probability flags,
        # iteration cap and the seed for probability-estimation cross-validation.
        int svm_type
        int kernel_type
        int degree    # for poly
        double gamma  # for poly/rbf/sigmoid
        double coef0  # for poly/sigmoid

        # these are for training only
        double cache_size  # in MB
        double eps         # stopping criteria
        double C           # for C_SVC, EPSILON_SVR and NU_SVR
        int nr_weight      # for C_SVC
        int *weight_label  # for C_SVC
        double* weight     # for C_SVC
        double nu          # for NU_SVC, ONE_CLASS, and NU_SVR
        double p           # for EPSILON_SVR
        int shrinking      # use the shrinking heuristics
        int probability    # do probability estimates
        int max_iter       # ceiling on Solver runtime
        int random_seed    # seed for random generator in probability estimation

    cdef struct svm_problem:
        # [CLASS DOC]
        # libsvm training problem: number of samples l, target array y, array of
        # svm_node rows x and per-sample weights W. y and W are borrowed buffers.
        int l
        double *y
        svm_node *x
        double *W  # instance weights

    char *svm_check_parameter(svm_problem *, svm_parameter *)
    # [FUNCTION DOC]
    # Validates problem/parameter combination. Returns NULL when valid, else a
    # pointer to a static error message. No exceptions or side effects.
    svm_model *svm_train(svm_problem *, svm_parameter *, int *, BlasFunctions *) nogil
    # [FUNCTION DOC]
    # Behavior: trains a libsvm model (SMO solver, one-vs-one for multiclass, Platt
    # scaling via internal cross-validation when probability is set).
    # Inputs: problem, parameter, address of an int status flag, BLAS table.
    # Outputs: newly allocated model (free with svm_free_and_destroy_model); the
    # status flag is set nonzero if max_iter was reached. Declared nogil; cannot
    # raise. Side effects: allocation, global RNG use, optional printing.
    void svm_free_and_destroy_model(svm_model** model_ptr_ptr)
    # [FUNCTION DOC]
    # Frees a model produced by svm_train (including its support vectors where owned)
    # and sets the pointer to NULL.
    void svm_cross_validation(svm_problem *, svm_parameter *, int nr_fold, double *target, BlasFunctions *) nogil
    # [FUNCTION DOC]
    # Behavior: nr_fold-fold cross-validation: trains on each split and writes the
    # prediction for every held-out sample into target (length l). Declared nogil.
    # Side effects: allocation, shuffles with the global RNG, optional printing.


cdef extern from "libsvm_helper.c":
    # this file contains methods for accessing libsvm 'hidden' fields
    svm_node **dense_to_sparse (char *, intp_t *)
    # [FUNCTION DOC]
    # Declared converter from dense matrix to libsvm nodes.
    # KNOWN ISSUE: libsvm_helper.c defines no dense_to_sparse (the function is named
    # dense_to_libsvm); this declaration is unused and would fail to link.
    void set_parameter (svm_parameter *, int , int , int , double, double ,
                        double , double , double , double,
                        double, int, int, int, char *, char *, int,
                        int)
    # [FUNCTION DOC]
    # Behavior: fills a caller-owned svm_parameter with, in order: svm_type,
    # kernel_type, degree, gamma, coef0, nu, cache_size, C, eps, p, shrinking,
    # probability, nr_weight, weight_label pointer, weight pointer (both borrowed),
    # max_iter, random_seed. No allocation, no errors.
    void set_problem (svm_problem *, char *, char *, char *, intp_t *, int)
    # [FUNCTION DOC]
    # Behavior: fills a caller-owned svm_problem: l from dims[0], y and W borrowed
    # from the given buffers, x set to a newly malloc'ed array of svm_node, one per
    # row of the C-contiguous float64 matrix X (pointing into X, not copied).
    # Inputs: problem, X, Y, sample_weight, dims (n_samples, n_features), kernel type
    # (unused by the helper). Outputs: problem fields; x is NULL on allocation
    # failure. Side effects: caller must free problem.x.

    svm_model *set_model (svm_parameter *, int, char *, intp_t *,
                          char *, intp_t *, intp_t *, char *,
                          char *, char *, char *, char *)
    # [FUNCTION DOC]
    # Behavior: rebuilds an svm_model from stored arrays for prediction: allocates
    # nSV, label, sv_coef row pointers and rho; for precomputed kernels creates nodes
    # holding only support indices, otherwise wraps SV rows; copies nSV and sets
    # labels 0..n_class-1 for classifiers; rho is the negated intercept; copies
    # probA/probB when probability is set; sv_coef rows point into the given buffer.
    # Inputs: param, nr_class, SV data and dims, support data and dims, sv_coef
    # strides (unused), sv_coef, intercept, nSV, probA, probB buffers.
    # Outputs: new model, or NULL if any allocation fails. Free with free_model.

    void copy_sv_coef   (char *, svm_model *)
    # [FUNCTION DOC]
    # Copies the (n_class-1) x l dual coefficient rows of the model into a
    # contiguous float64 buffer.
    void copy_n_iter  (char *, svm_model *)
    # [FUNCTION DOC]
    # Copies max(1, n_class*(n_class-1)/2) int iteration counts into the buffer.
    void copy_intercept (char *, svm_model *, intp_t *)
    # [FUNCTION DOC]
    # Writes the intercepts (negated rho, with -0.0 normalized to 0) for dims[0]
    # binary sub-problems into the float64 buffer.
    void copy_SV        (char *, svm_model *, intp_t *)
    # [FUNCTION DOC]
    # Copies the dense support vectors (l rows of SV[0].dim values) into a
    # contiguous float64 buffer; dims is unused. Must not be called for precomputed
    # kernels or when l is 0.
    int copy_support (char *data, svm_model *model)
    # [FUNCTION DOC]
    # Copies the l int support-vector indices (into the training set) into data.
    # KNOWN ISSUE: declared as returning int, but the C function returns void; the
    # return value is never used, so this is harmless in practice.
    int copy_predict (char *, svm_model *, intp_t *, char *, BlasFunctions *) nogil
    # [FUNCTION DOC]
    # Behavior: wraps the dense test matrix in svm_node rows and writes svm_predict
    # (class label or regression value) for each sample into the float64 output.
    # Outputs: 0 on success, -1 if the temporary node array cannot be allocated.
    # Declared nogil; allocates and frees a temporary array.
    int copy_predict_proba (char *, svm_model *, intp_t *, char *, BlasFunctions *) nogil
    # [FUNCTION DOC]
    # Same as copy_predict but writes n_class probability estimates per sample
    # (row-major) via svm_predict_probability. Returns 0, or -1 on allocation failure.
    int copy_predict_values(char *, svm_model *, intp_t *, char *, int, BlasFunctions *) nogil
    # [FUNCTION DOC]
    # Same as copy_predict but writes the raw decision values (one per one-vs-one
    # pair, or 1 for one-class/regression) per sample, using the given row stride
    # nr_class. Returns 0, or -1 on allocation failure.
    void copy_nSV     (char *, svm_model *)
    # [FUNCTION DOC]
    # Copies the per-class support vector counts into data; does nothing when the
    # model has no labels (regression/one-class).
    void copy_probA   (char *, svm_model *, intp_t *)
    # [FUNCTION DOC]
    # Copies dims[0] Platt-scaling A coefficients from the model into data.
    void copy_probB   (char *, svm_model *, intp_t *)
    # [FUNCTION DOC]
    # Copies dims[0] Platt-scaling B coefficients from the model into data.
    intp_t  get_l  (svm_model *)
    # [FUNCTION DOC]
    # Returns the number of support vectors l of the model.
    intp_t  get_nr (svm_model *)
    # [FUNCTION DOC]
    # Returns nr_class of the model (2 for regression and one-class models).
    int  free_problem   (svm_problem *)
    # [FUNCTION DOC]
    # Declared problem destructor.
    # KNOWN ISSUE: libsvm_helper.c defines no free_problem (only the sparse helper
    # does, for a different struct); unused here and would fail to link.
    int  free_model     (svm_model *)
    # [FUNCTION DOC]
    # Frees a model built by set_model: SV nodes, sv_coef row pointers, rho, label,
    # probA, probB, nSV and the struct; not sv_coef data, sv_ind or n_iter.
    # Returns -1 for NULL, else 0.
    void set_verbosity(int)
    # [FUNCTION DOC]
    # Installs libsvm's global print function (stdout if nonzero, else silent).
    # Side effects: process-global state.
