# [MODULE DOC]
# Cython bindings for sklearn's libsvm fork on CSR sparse input, used by the
# libsvm-based estimators (SVC, NuSVC, SVR, NuSVR, OneClassSVM) when X is sparse.
# Declares the sparse (svm_csr_*) libsvm API and the libsvm_sparse_helper.c
# layer that converts CSR arrays to per-row libsvm node arrays (1-based indices,
# -1 sentinel) and copies model fields out. Training returns the support vectors
# as a CSR matrix; prediction functions rebuild a temporary svm_csr_model from
# fitted arrays. Precomputed kernels are not supported here.
import  numpy as np
from scipy import sparse
from sklearn.utils._cython_blas cimport _dot
from sklearn.utils._sparse import _align_api_if_sparse
from sklearn.utils._typedefs cimport float64_t, int32_t, intp_t

cdef extern from *:
    ctypedef char* const_char_p "const char*"

################################################################################
# Includes

cdef extern from "_svm_cython_blas_helpers.h":
    ctypedef double (*dot_func)(int, const double*, int, const double*, int)
    cdef struct BlasFunctions:
        # [CLASS DOC]
        # Table holding a pointer to a double-precision BLAS dot routine passed into
        # libsvm for kernel evaluations.
        dot_func dot

cdef extern from "svm.h":
    cdef struct svm_csr_node
    # [CLASS DOC]
    # Opaque sparse libsvm element: (1-based feature index, value); rows end with an
    # element whose index is -1.
    cdef struct svm_csr_model
    # [CLASS DOC]
    # Opaque sparse libsvm model: parameters, class count, support vectors as node
    # rows, dual coefficients, rho, probability parameters, labels, support indices
    # and counts, iteration counts.
    cdef struct svm_parameter
    # [CLASS DOC]
    # Opaque libsvm hyper-parameter struct (allocated by set_parameter here).
    cdef struct svm_csr_problem
    # [CLASS DOC]
    # Opaque sparse libsvm training problem: sample count, targets, node rows and
    # sample weights.
    char *svm_csr_check_parameter(svm_csr_problem *, svm_parameter *)
    # [FUNCTION DOC]
    # Validates a sparse problem/parameter pair; returns NULL if valid, else a static
    # error message. No side effects.
    svm_csr_model *svm_csr_train(svm_csr_problem *, svm_parameter *, int *, BlasFunctions *) nogil
    # [FUNCTION DOC]
    # Trains a libsvm model on a sparse problem (same algorithm as dense svm_train);
    # sets *status nonzero when max_iter was reached. Declared nogil. Returns a newly
    # allocated model owning its support vector copies.
    void svm_csr_free_and_destroy_model(svm_csr_model** model_ptr_ptr)
    # [FUNCTION DOC]
    # Frees a model produced by svm_csr_train and sets the pointer to NULL.

cdef extern from "libsvm_sparse_helper.c":
    # this file contains methods for accessing libsvm 'hidden' fields
    svm_csr_problem * csr_set_problem (
        char *, intp_t *, char *, intp_t *, char *, char *, char *, int)
    # [FUNCTION DOC]
    # Behavior: allocates a problem; l = n_indptr[0] - 1; y and W borrowed; x is a
    # newly allocated array of per-row node arrays converted from CSR (values copied,
    # indices shifted to 1-based, -1 sentinel per row).
    # Inputs: values, indices shape and data (int32), indptr shape and data, Y,
    # sample_weight, kernel type (unused).
    # Outputs: new problem, or NULL on any allocation failure (partial rows freed).
    svm_csr_model *csr_set_model(svm_parameter *param, int nr_class,
                                 char *SV_data, intp_t *SV_indices_dims,
                                 char *SV_indices, intp_t *SV_intptr_dims,
                                 char *SV_intptr,
                                 char *sv_coef, char *rho, char *nSV,
                                 char *probA, char *probB)
    # [FUNCTION DOC]
    # Behavior: rebuilds an svm_csr_model for prediction: allocates nSV, label,
    # sv_coef rows (copied from the flat sv_coef buffer, one row of l per class
    # minus one) and rho (negated intercept); converts the CSR support vectors into
    # node rows; for classifiers copies nSV and sets labels 0..n_class-1; copies
    # probA/probB when probability is set.
    # Inputs: param, nr_class, SV CSR arrays with their shapes, sv_coef, intercept,
    # nSV, probA, probB buffers.
    # Outputs: new model or NULL on allocation failure. Free with free_model_SV then
    # free_model.
    # KNOWN ISSUE: a NULL result of the support vector conversion is not checked, so
    # the model may hold SV == NULL after an allocation failure.
    svm_parameter *set_parameter (int , int , int , double, double ,
                                  double , double , double , double,
                                  double, int, int, int, char *, char *, int,
                                  int)
    # [FUNCTION DOC]
    # Behavior: mallocs an svm_parameter and fills it with, in order: svm_type,
    # kernel_type, degree, gamma, coef0, nu, cache_size, C, eps, p, shrinking,
    # probability, nr_weight, weight_label and weight pointers (borrowed), max_iter,
    # random_seed. Outputs: new pointer or NULL; free with free_param.
    void copy_sv_coef   (char *, svm_csr_model *)
    # [FUNCTION DOC]
    # Copies the (n_class-1) x l dual coefficients into a contiguous float64 buffer.
    void copy_n_iter  (char *, svm_csr_model *)
    # [FUNCTION DOC]
    # Copies max(1, n_class*(n_class-1)/2) int iteration counts into the buffer.
    void copy_support   (char *, svm_csr_model *)
    # [FUNCTION DOC]
    # Copies the l int support-vector indices (into the training set) into data.
    void copy_intercept (char *, svm_csr_model *, intp_t *)
    # [FUNCTION DOC]
    # Writes intercepts (negated rho, -0.0 normalized to 0) for dims[0] pairs.
    int copy_predict (char *, svm_csr_model *, intp_t *, char *, BlasFunctions *)
    # [FUNCTION DOC]
    # Declared dense prediction helper.
    # KNOWN ISSUE: libsvm_sparse_helper.c defines no copy_predict; unused declaration
    # that would fail to link if called.
    int csr_copy_predict_values (intp_t *data_size, char *data, intp_t *index_size,
                                 char *index, intp_t *intptr_size, char *size,
                                 svm_csr_model *model, char *dec_values, int nr_class, BlasFunctions *)
    # [FUNCTION DOC]
    # Behavior: converts the CSR test matrix into node rows and writes raw
    # one-vs-one decision values (nr_class per sample, row-major) using
    # svm_csr_predict_values; each row is freed after use.
    # Inputs: sizes and pointers of data, indices and indptr; model; output buffer;
    # row stride nr_class; BLAS table. Outputs: 0, or -1 on allocation failure.
    int csr_copy_predict (intp_t *data_size, char *data, intp_t *index_size,
                          char *index, intp_t *intptr_size, char *size,
                          svm_csr_model *model, char *dec_values, BlasFunctions *) nogil
    # [FUNCTION DOC]
    # Same conversion as above but writes one svm_csr_predict result (label or
    # value) per sample. Declared nogil. Returns 0, or -1 on allocation failure.
    int csr_copy_predict_proba (intp_t *data_size, char *data, intp_t *index_size,
                                char *index, intp_t *intptr_size, char *size,
                                svm_csr_model *model, char *dec_values, BlasFunctions *) nogil
    # [FUNCTION DOC]
    # Same conversion but writes n_class probability estimates per sample via
    # svm_csr_predict_probability. Declared nogil. Returns 0 or -1.

    int  copy_predict_values(char *, svm_csr_model *, intp_t *, char *, int, BlasFunctions *)
    # [FUNCTION DOC]
    # Declared dense decision-value helper.
    # KNOWN ISSUE: not defined in libsvm_sparse_helper.c; unused declaration.
    int  csr_copy_SV (char *values, intp_t *n_indices,
                      char *indices, intp_t *n_indptr, char *indptr,
                      svm_csr_model *model, int n_features)
    # [FUNCTION DOC]
    # Behavior: writes the model's support vectors into caller-allocated CSR arrays:
    # values, 0-based column indices and indptr (indptr[0] = 0, indptr[i+1] =
    # running count). Shapes and n_features are unused. Returns 0.
    intp_t get_nonzero_SV (svm_csr_model *)
    # [FUNCTION DOC]
    # Returns the total number of stored (nonzero) entries across all support
    # vectors, used to size csr_copy_SV's output.
    void copy_nSV     (char *, svm_csr_model *)
    # [FUNCTION DOC]
    # Copies per-class support counts into data; does nothing if the model has no
    # labels (regression/one-class).
    void copy_probA   (char *, svm_csr_model *, intp_t *)
    # [FUNCTION DOC]
    # Copies dims[0] Platt A coefficients into data.
    void copy_probB   (char *, svm_csr_model *, intp_t *)
    # [FUNCTION DOC]
    # Copies dims[0] Platt B coefficients into data.
    intp_t  get_l  (svm_csr_model *)
    # [FUNCTION DOC]
    # Returns the number of support vectors l.
    intp_t  get_nr (svm_csr_model *)
    # [FUNCTION DOC]
    # Returns nr_class (2 for regression/one-class).
    int  free_problem   (svm_csr_problem *)
    # [FUNCTION DOC]
    # Frees every node row, the row array and the problem struct (not y/W).
    # Returns -1 for NULL, else 0.
    int  free_model     (svm_csr_model *)
    # [FUNCTION DOC]
    # Frees the model struct and its SV row array, sv_coef row-pointer array, rho,
    # label, probA, probB and nSV (not the individual SV rows or sv_coef rows).
    # Returns -1 for NULL, else 0.
    int  free_param     (svm_parameter *)
    # [FUNCTION DOC]
    # Frees a parameter struct allocated by set_parameter. Returns -1 for NULL.
    int free_model_SV(svm_csr_model *model)
    # [FUNCTION DOC]
    # Frees each support vector row and each sv_coef row of a model built by
    # csr_set_model; must be called before free_model. Returns 0.
    void set_verbosity(int)
    # [FUNCTION DOC]
    # Installs libsvm's global print function (stdout if nonzero, else silent).


def libsvm_sparse_train (int n_features,
                         const float64_t[::1] values,
                         const int32_t[::1] indices,
                         const int32_t[::1] indptr,
                         const float64_t[::1] Y,
                         int svm_type, int kernel_type, int degree, double gamma,
                         double coef0, double eps, double C,
                         const float64_t[::1] class_weight,
                         const float64_t[::1] sample_weight,
                         double nu, double cache_size, double p, int
                         shrinking, int probability, int max_iter,
                         int random_seed):
    # [FUNCTION DOC]
    # Behavior: fills sample_weight with ones if empty, builds a sparse problem and
    # parameter struct, validates, trains with the GIL released, then copies out
    # iteration counts, flat dual coefficients, support indices, intercepts, support
    # vectors as a CSR matrix of shape (n_SV, n_features), per-class SV counts and
    # Platt parameters (probA length 1 and empty probB for regression/one-class);
    # frees model, problem and parameters.
    # Inputs: n_features; CSR arrays values (float64), indices and indptr (int32);
    # Y float64 targets; svm_type; kernel_type index (must not be 4/precomputed);
    # degree, gamma, coef0; eps (tolerance); C; class_weight; sample_weight; nu;
    # cache_size (MB); p (epsilon-SVR width); shrinking; probability; max_iter;
    # random_seed.
    # Outputs: tuple (support, support_vectors, sv_coef_data flat of length
    # (n_class-1)*n_SV, intercept, n_class_SV, probA, probB, fit_status, n_iter).
    # Exceptions: AssertionError on sample_weight length mismatch or precomputed
    # kernel; MemoryError on allocation failure; ValueError for invalid parameters
    # (raw bytes message).
    # Side effects: allocation/free; GIL release; libsvm global RNG and printing.
    # KNOWN ISSUE: if only one of param/problem is NULL, the other is leaked when
    # MemoryError is raised.
    # KNOWN ISSUE: for regression and one-class models copy_nSV writes nothing, so
    # n_class_SV is returned uninitialized (the dense fit returns [n_SV, n_SV]).
    # KNOWN ISSUE: the docstring documents a parameter X that does not exist (the
    # matrix is passed as values/indices/indptr).
    """
    Wrap svm_train from libsvm using a scipy.sparse.csr matrix

    Work in progress.

    Parameters
    ----------
    n_features : number of features.
        XXX: can we retrieve this from any other parameter ?

    X : array-like, dtype=float, size=[N, D]

    Y : array, dtype=float, size=[N]
        target vector

    ...

    Notes
    -------------------
    See sklearn.svm.predict for a complete list of parameters.

    """

    cdef svm_parameter *param
    cdef svm_csr_problem *problem
    cdef svm_csr_model *model
    cdef const_char_p error_msg

    if len(sample_weight) == 0:
        sample_weight = np.ones(Y.shape[0], dtype=np.float64)
    else:
        assert sample_weight.shape[0] == indptr.shape[0] - 1, \
               "sample_weight and X have incompatible shapes: " + \
               "sample_weight has %s samples while X has %s" % \
               (sample_weight.shape[0], indptr.shape[0] - 1)

    # we should never end up here with a precomputed kernel matrix,
    # as this is always dense.
    assert(kernel_type != 4)

    # set libsvm problem
    problem = csr_set_problem(
        <char *> &values[0],
        <intp_t *> indices.shape,
        <char *> &indices[0],
        <intp_t *> indptr.shape,
        <char *> &indptr[0],
        <char *> &Y[0],
        <char *> &sample_weight[0],
        kernel_type,
    )

    cdef int32_t[::1] \
        class_weight_label = np.arange(class_weight.shape[0], dtype=np.int32)

    # set parameters
    param = set_parameter(
        svm_type,
        kernel_type,
        degree,
        gamma,
        coef0,
        nu,
        cache_size,
        C,
        eps,
        p,
        shrinking,
        probability,
        <int> class_weight.shape[0],
        <char *> &class_weight_label[0] if class_weight_label.size > 0 else NULL,
        <char *> &class_weight[0] if class_weight.size > 0 else NULL, max_iter,
        random_seed,
    )

    # check parameters
    if (param == NULL or problem == NULL):
        raise MemoryError("Seems we've run out of memory")
    error_msg = svm_csr_check_parameter(problem, param)
    if error_msg:
        free_problem(problem)
        free_param(param)
        raise ValueError(error_msg)
    cdef BlasFunctions blas_functions
    blas_functions.dot = _dot[double]
    # call svm_train, this does the real work
    cdef int fit_status = 0
    with nogil:
        model = svm_csr_train(problem, param, &fit_status, &blas_functions)

    cdef intp_t SV_len = get_l(model)
    cdef intp_t n_class = get_nr(model)

    cdef int[::1] n_iter
    n_iter = np.empty(max(1, n_class * (n_class - 1) // 2), dtype=np.intc)
    copy_n_iter(<char *> &n_iter[0], model)

    # copy model.sv_coef
    # we create a new array instead of resizing, otherwise
    # it would not erase previous information
    cdef float64_t[::1] sv_coef_data
    sv_coef_data = np.empty((n_class-1)*SV_len, dtype=np.float64)
    copy_sv_coef (<char *> &sv_coef_data[0] if sv_coef_data.size > 0 else NULL, model)

    cdef int32_t[::1] support
    support = np.empty(SV_len, dtype=np.int32)
    copy_support(<char *> &support[0] if support.size > 0 else NULL, model)

    # copy model.rho into the intercept
    # the intercept is just model.rho but with sign changed
    cdef float64_t[::1]intercept
    intercept = np.empty(n_class*(n_class-1)//2, dtype=np.float64)
    copy_intercept (<char *> &intercept[0], model, <intp_t *> intercept.shape)

    # copy model.SV
    # we erase any previous information in SV
    # TODO: custom kernel
    cdef intp_t nonzero_SV
    nonzero_SV = get_nonzero_SV (model)

    cdef float64_t[::1] SV_data
    cdef int32_t[::1] SV_indices, SV_indptr
    SV_data = np.empty(nonzero_SV, dtype=np.float64)
    SV_indices = np.empty(nonzero_SV, dtype=np.int32)
    SV_indptr = np.empty(<intp_t>SV_len + 1, dtype=np.int32)
    csr_copy_SV(
        <char *> &SV_data[0] if SV_data.size > 0 else NULL,
        <intp_t *> SV_indices.shape,
        <char *> &SV_indices[0] if SV_indices.size > 0 else NULL,
        <intp_t *> SV_indptr.shape,
        <char *> &SV_indptr[0] if SV_indptr.size > 0 else NULL,
        model,
        n_features,
    )
    support_vectors_ = _align_api_if_sparse(sparse.csr_array(
        (SV_data, SV_indices, SV_indptr), (SV_len, n_features)
    ))

    # copy model.nSV
    # TODO: do only in classification
    cdef int32_t[::1]n_class_SV
    n_class_SV = np.empty(n_class, dtype=np.int32)
    copy_nSV(<char *> &n_class_SV[0], model)

    # # copy probabilities
    cdef float64_t[::1] probA, probB
    if probability != 0:
        if svm_type < 2:  # SVC and NuSVC
            probA = np.empty(n_class*(n_class-1)//2, dtype=np.float64)
            probB = np.empty(n_class*(n_class-1)//2, dtype=np.float64)
            copy_probB(<char *> &probB[0], model, <intp_t *> probB.shape)
        else:
            probA = np.empty(1, dtype=np.float64)
            probB = np.empty(0, dtype=np.float64)
        copy_probA(<char *> &probA[0], model, <intp_t *> probA.shape)
    else:
        probA = np.empty(0, dtype=np.float64)
        probB = np.empty(0, dtype=np.float64)

    svm_csr_free_and_destroy_model (&model)
    free_problem(problem)
    free_param(param)

    return (
        support.base,
        support_vectors_,
        sv_coef_data.base,
        intercept.base,
        n_class_SV.base,
        probA.base,
        probB.base,
        fit_status,
        n_iter.base,
    )


def libsvm_sparse_predict (const float64_t[::1] T_data,
                           const int32_t[::1] T_indices,
                           const int32_t[::1] T_indptr,
                           const float64_t[::1] SV_data,
                           const int32_t[::1] SV_indices,
                           const int32_t[::1] SV_indptr,
                           const float64_t[::1] sv_coef,
                           const float64_t[::1]
                           intercept, int svm_type, int kernel_type, int
                           degree, double gamma, double coef0, double
                           eps, double C,
                           const float64_t[:] class_weight,
                           double nu, double p, int
                           shrinking, int probability,
                           const int32_t[::1] nSV,
                           const float64_t[::1] probA,
                           const float64_t[::1] probB):
    # [FUNCTION DOC]
    # Behavior: builds a parameter struct (cache size 100, max_iter and seed -1),
    # rebuilds the model with csr_set_model, predicts each CSR test row with the GIL
    # released via csr_copy_predict, then frees SV/coef rows, model and parameters.
    # Inputs: test CSR arrays T_data/T_indices/T_indptr; fitted SV CSR arrays,
    # flat sv_coef, intercept, svm_type, kernel_type, degree, gamma, coef0, eps, C,
    # class_weight, nu, p, shrinking, probability, nSV, probA, probB.
    # Outputs: float64 array of shape (n_test,).
    # Exceptions: MemoryError if row conversion fails; IndexError if T_data,
    # T_indices or intercept is empty.
    # Side effects: allocation/free; GIL release.
    # KNOWN ISSUE: NULL returns from set_parameter/csr_set_model are not checked, and
    # when MemoryError is raised the model and param are leaked.
    # KNOWN ISSUE: the docstring refers to copy_predict in libsvm_helper.c, but the
    # call is csr_copy_predict in libsvm_sparse_helper.c.
    """
    Predict values T given a model.

    For speed, all real work is done at the C level in function
    copy_predict (libsvm_helper.c).

    We have to reconstruct model and parameters to make sure we stay
    in sync with the python object.

    See sklearn.svm.predict for a complete list of parameters.

    Parameters
    ----------
    X : array-like, dtype=float
    Y : array
        target vector

    Returns
    -------
    dec_values : array
        predicted values.
    """
    cdef float64_t[::1] dec_values
    cdef svm_parameter *param
    cdef svm_csr_model *model
    cdef int32_t[::1] \
        class_weight_label = np.arange(class_weight.shape[0], dtype=np.int32)
    cdef int rv
    param = set_parameter(
        svm_type,
        kernel_type,
        degree,
        gamma,
        coef0,
        nu,
        100.0,  # cache size has no effect on predict
        C,
        eps,
        p,
        shrinking,
        probability,
        <int> class_weight.shape[0],
        <char *> &class_weight_label[0] if class_weight_label.size > 0 else NULL,
        <char *> &class_weight[0] if class_weight.size > 0 else NULL,
        -1,
        -1,  # random seed has no effect on predict either
    )

    model = csr_set_model(
        param, <int> nSV.shape[0],
        <char *> &SV_data[0] if SV_data.size > 0 else NULL,
        <intp_t *>SV_indices.shape,
        <char *> &SV_indices[0] if SV_indices.size > 0 else NULL,
        <intp_t *> SV_indptr.shape,
        <char *> &SV_indptr[0] if SV_indptr.size > 0 else NULL,
        <char *> &sv_coef[0] if sv_coef.size > 0 else NULL,
        <char *> &intercept[0],
        <char *> &nSV[0],
        <char *> &probA[0] if probA.size > 0 else NULL,
        <char *> &probB[0] if probB.size > 0 else NULL,
    )
    # TODO: use check_model
    dec_values = np.empty(T_indptr.shape[0]-1)
    cdef BlasFunctions blas_functions
    blas_functions.dot = _dot[double]
    with nogil:
        rv = csr_copy_predict(
            <intp_t *> T_data.shape,
            <char *> &T_data[0],
            <intp_t *> T_indices.shape,
            <char *> &T_indices[0],
            <intp_t *> T_indptr.shape,
            <char *> &T_indptr[0],
            model,
            <char *> &dec_values[0],
            &blas_functions,
        )
    if rv < 0:
        raise MemoryError("We've run out of memory")
    # free model and param
    free_model_SV(model)
    free_model(model)
    free_param(param)
    return dec_values.base


def libsvm_sparse_predict_proba(
    const float64_t[::1] T_data,
    const int32_t[::1] T_indices,
    const int32_t[::1] T_indptr,
    const float64_t[::1] SV_data,
    const int32_t[::1] SV_indices,
    const int32_t[::1] SV_indptr,
    const float64_t[::1] sv_coef,
    const float64_t[::1]
    intercept, int svm_type, int kernel_type, int
    degree, double gamma, double coef0, double
    eps, double C,
    const float64_t[:] class_weight,
    double nu, double p, int shrinking, int probability,
    const int32_t[::1] nSV,
    const float64_t[::1] probA,
    const float64_t[::1] probB,
):
    # [FUNCTION DOC]
    # Behavior: same setup as libsvm_sparse_predict, then computes n_class
    # probability estimates per test row with csr_copy_predict_proba (GIL released).
    # Inputs: same as libsvm_sparse_predict; probability should be nonzero and
    # probA/probB non-empty.
    # Outputs: float64 array of shape (n_test, n_class).
    # Exceptions/side effects: as libsvm_sparse_predict (same unchecked NULLs and
    # leak on MemoryError).
    """
    Predict values T given a model.
    """
    cdef float64_t[:, ::1] dec_values
    cdef svm_parameter *param
    cdef svm_csr_model *model
    cdef int32_t[::1] \
        class_weight_label = np.arange(class_weight.shape[0], dtype=np.int32)
    param = set_parameter(
        svm_type,
        kernel_type,
        degree,
        gamma,
        coef0,
        nu,
        100.0,  # cache size has no effect on predict
        C,
        eps,
        p,
        shrinking,
        probability,
        <int> class_weight.shape[0],
        <char *> &class_weight_label[0] if class_weight_label.size > 0 else NULL,
        <char *> &class_weight[0] if class_weight.size > 0 else NULL,
        -1,
        -1,  # random seed has no effect on predict either
    )

    model = csr_set_model(
        param,
        <int> nSV.shape[0],
        <char *> &SV_data[0] if SV_data.size > 0 else NULL,
        <intp_t *> SV_indices.shape,
        <char *> &SV_indices[0] if SV_indices.size > 0 else NULL,
        <intp_t *> SV_indptr.shape,
        <char *> &SV_indptr[0] if SV_indptr.size > 0 else NULL,
        <char *> &sv_coef[0] if sv_coef.size > 0 else NULL,
        <char *> &intercept[0],
        <char *> &nSV[0],
        <char *> &probA[0] if probA.size > 0 else NULL,
        <char *> &probB[0] if probB.size > 0 else NULL,
    )
    # TODO: use check_model
    cdef intp_t n_class = get_nr(model)
    cdef int rv
    dec_values = np.empty((T_indptr.shape[0]-1, n_class), dtype=np.float64)
    cdef BlasFunctions blas_functions
    blas_functions.dot = _dot[double]
    with nogil:
        rv = csr_copy_predict_proba(
            <intp_t *> T_data.shape,
            <char *> &T_data[0],
            <intp_t *> T_indices.shape,
            <char *> &T_indices[0],
            <intp_t *> T_indptr.shape,
            <char *> &T_indptr[0],
            model,
            <char *> &dec_values[0, 0],
            &blas_functions,
        )
    if rv < 0:
        raise MemoryError("We've run out of memory")
    # free model and param
    free_model_SV(model)
    free_model(model)
    free_param(param)
    return dec_values.base


def libsvm_sparse_decision_function(
    const float64_t[::1] T_data,
    const int32_t[::1] T_indices,
    const int32_t[::1] T_indptr,
    const float64_t[::1] SV_data,
    const int32_t[::1] SV_indices,
    const int32_t[::1] SV_indptr,
    const float64_t[::1] sv_coef,
    const float64_t[::1]
    intercept, int svm_type, int kernel_type, int
    degree, double gamma, double coef0, double
    eps, double C,
    const float64_t[:] class_weight,
    double nu, double p, int shrinking, int probability,
    const int32_t[::1] nSV,
    const float64_t[::1] probA,
    const float64_t[::1] probB,
):
    # [FUNCTION DOC]
    # Behavior: same setup as libsvm_sparse_predict, then writes raw decision
    # values with csr_copy_predict_values: n_class*(n_class-1)/2 per row for
    # classifiers, 1 for one-class/regression (svm_type > 1).
    # Inputs: same as libsvm_sparse_predict.
    # Outputs: float64 array of shape (n_test, n_pairs or 1).
    # Exceptions: MemoryError on allocation failure (model/param then leaked).
    # Side effects: allocation/free. Unlike the other predict functions the GIL is
    # held during the computation.
    """
    Predict margin (libsvm name for this is predict_values)

    We have to reconstruct model and parameters to make sure we stay
    in sync with the python object.
    """
    cdef float64_t[:, ::1] dec_values
    cdef svm_parameter *param
    cdef intp_t n_class

    cdef svm_csr_model *model
    cdef int32_t[::1] \
        class_weight_label = np.arange(class_weight.shape[0], dtype=np.int32)
    param = set_parameter(
        svm_type,
        kernel_type,
        degree,
        gamma,
        coef0,
        nu,
        100.0,  # cache size has no effect on predict
        C,
        eps,
        p,
        shrinking,
        probability,
        <int> class_weight.shape[0],
        <char *> &class_weight_label[0] if class_weight_label.size > 0 else NULL,
        <char *> &class_weight[0] if class_weight.size > 0 else NULL,
        -1,
        -1,
    )

    model = csr_set_model(
        param,
        <int> nSV.shape[0],
        <char *> &SV_data[0] if SV_data.size > 0 else NULL,
        <intp_t *> SV_indices.shape,
        <char *> &SV_indices[0] if SV_indices.size > 0 else NULL,
        <intp_t *> SV_indptr.shape,
        <char *> &SV_indptr[0] if SV_indptr.size > 0 else NULL,
        <char *> &sv_coef[0] if sv_coef.size > 0 else NULL,
        <char *> &intercept[0],
        <char *> &nSV[0],
        <char *> &probA[0] if probA.size > 0 else NULL,
        <char *> &probB[0] if probB.size > 0 else NULL,
    )

    if svm_type > 1:
        n_class = 1
    else:
        n_class = get_nr(model)
        n_class = n_class * (n_class - 1) // 2

    dec_values = np.empty((T_indptr.shape[0] - 1, n_class), dtype=np.float64)
    cdef BlasFunctions blas_functions
    blas_functions.dot = _dot[double]
    if csr_copy_predict_values(
            <intp_t *> T_data.shape,
            <char *> &T_data[0],
            <intp_t *> T_indices.shape,
            <char *> &T_indices[0],
            <intp_t *> T_indptr.shape,
            <char *> &T_indptr[0],
            model,
            <char *> &dec_values[0, 0],
            n_class,
            &blas_functions,
    ) < 0:
        raise MemoryError("We've run out of memory")
    # free model and param
    free_model_SV(model)
    free_model(model)
    free_param(param)

    return dec_values.base


def set_verbosity_wrap(int verbosity):
    # [FUNCTION DOC]
    # Turns libsvm's progress printing on (stdout) or off; mutates global state.
    """
    Control verbosity of libsvm library
    """
    set_verbosity(verbosity)
