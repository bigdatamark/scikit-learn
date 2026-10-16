# [PACKAGE DOC]
# Package: sklearn.svm
# Cython components of sklearn.svm:
# - _libsvm.pyx (+ _libsvm.pxi declarations): bindings to sklearn's fork of
#   libsvm for dense input: fit, predict, predict_proba, decision_function,
#   cross_validation, set_verbosity_wrap. Used by SVC, NuSVC, SVR, NuSVR and
#   OneClassSVM via _base.py.
# - _libsvm_sparse.pyx: the same libsvm operations on CSR sparse input
#   (libsvm_sparse_train/predict/predict_proba/decision_function).
# - _liblinear.pyx (+ _liblinear.pxi declarations): train_wrap for liblinear,
#   used by LinearSVC, LinearSVR and LogisticRegression(solver="liblinear").
# - _newrand.pyx: wrappers for the shared mt19937-based RNG (newrand.h) that
#   libsvm and liblinear use instead of rand(); seeded via the wrappers'
#   random_seed arguments.
# All wrappers convert numpy buffers to C structs through the C helpers in
# src/, release the GIL during training/prediction, and pass scipy BLAS
# function pointers to the C++ solvers.
# (This __init__.pxd holds only documentation; it declares nothing.)
