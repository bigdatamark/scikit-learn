# [PACKAGE DOC]
# Package: sklearn.tree
# Compiled (Cython) components of sklearn.tree, layered bottom-up:
# - _utils: low-level helpers shared by the others: safe_realloc (checked
#   malloc growth), random integer/uniform draws from a caller-held RNG state,
#   base-2 log, pointer-to-ndarray conversion, and WeightedFenwickTree
#   (prefix sums of weights and weight*y, used by the MAE criterion).
# - _criterion: impurity criteria (Gini, Entropy, MSE, Poisson, MAE). A
#   Criterion is bound to a node's sample range and maintains left/right child
#   statistics while the split position moves; it gives impurities, the
#   improvement of a split and node values, plus monotonic-constraint helpers.
# - _partitioner: dense and sparse (CSC) partitioners that gather and sort one
#   feature's values for the node's samples, enumerate candidate thresholds,
#   handle missing values and reorder sample indices in place for a split.
# - _splitter: BestSplitter / RandomSplitter (and sparse variants) that pick
#   candidate features (random draws, skipping known constant features), use a
#   partitioner to sweep thresholds and a criterion to score them, and return a
#   SplitRecord for a node.
# - _tree: the Tree storage (array of Node structs plus value buffer) with
#   prediction, decision paths, feature importances and pickling; the
#   depth-first and best-first TreeBuilders that drive a Splitter to grow a
#   Tree; and minimal cost-complexity pruning.
# The Python estimators in _classes.py build a Criterion, Splitter and
# TreeBuilder, call build to fill a Tree, and use Tree methods for predict.
# (This __init__.pxd holds only documentation; it declares nothing.)
