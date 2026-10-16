# [PACKAGE DOC]
# Package: sklearn.ensemble
# Cython components of sklearn.ensemble:
# - _gradient_boosting.pyx: compiled helpers for the classic GradientBoosting*
#   estimators. predict_stages / predict_stage add learning-rate-scaled
#   predictions of fitted sklearn.tree regression trees to a raw-prediction array
#   by walking the trees' node arrays directly (dense C-ordered float32 or CSR
#   input); _random_sample_mask draws the in-bag mask for subsample < 1.
# - _hist_gradient_boosting/: the compiled core of HistGradientBoosting*
#   (binning, histograms, split finding, prediction); see the package doc in
#   _hist_gradient_boosting/__init__.py.
# The forest, bagging, voting, stacking, AdaBoost and isolation-forest
# estimators are pure Python (their trees come from sklearn.tree).
# (This __init__.pxd holds only documentation; it declares nothing.)
