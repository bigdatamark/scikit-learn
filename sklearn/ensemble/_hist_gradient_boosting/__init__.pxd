# [PACKAGE DOC]
# Package: sklearn.ensemble._hist_gradient_boosting
# Cython components of the histogram gradient boosting package. Shared types:
# common.pxd declares the C typedefs (float64 raw X and targets, uint8 bin codes,
# float32 gradients/hessians), the packed hist_struct / node_struct records and
# the MonotonicConstraint enum; common.pyx defines the matching numpy dtypes
# (HISTOGRAM_DTYPE, PREDICTOR_RECORD_DTYPE) used by all modules.
# Training pipeline, driven by the pure-Python gradient_boosting.py and
# grower.py:
# - _binning.pyx: _map_to_bins maps raw features to uint8 bins (used by
#   binning._BinMapper.transform), with a dedicated missing-values bin.
# - histogram.pyx: HistogramBuilder computes per-node, per-feature
#   gradient/hessian/count histograms from scratch or by subtraction from the
#   parent histogram.
# - splitting.pyx: Splitter finds the best split of a node from its histograms
#   (numerical and categorical features, missing values, monotonic and
#   interaction constraints, feature subsampling) and partitions sample indices
#   between children; SplitInfo carries the result to the grower.
# - _gradient_boosting.pyx: _update_raw_predictions adds the new tree's leaf
#   values to the training raw predictions using the splitter's partition.
# Prediction:
# - _predictor.pyx: per-tree prediction from raw or binned data over
#   node_struct arrays (used by predictor.TreePredictor) and the recursion-based
#   partial dependence computation.
# (This __init__.pxd holds only documentation; it declares nothing.)
