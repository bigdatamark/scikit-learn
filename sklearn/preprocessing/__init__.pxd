# [PACKAGE DOC]
# Package: sklearn.preprocessing
# Cython components:
# - _csr_polynomial_expansion: degree-2/3 polynomial/interaction expansion of
#   CSR input for PolynomialFeatures; computes output nnz then fills output CSR
#   arrays using closed-form column indices over non-zero products only.
# - _target_encoder_fast: per-category smoothed target means (fixed or "auto"
#   empirical-Bayes smoothing) for TargetEncoder.fit.
# The two modules are independent.
# (This __init__.pxd holds only documentation; it declares nothing.)
