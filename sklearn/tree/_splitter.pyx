# [MODULE DOC]
# Split search for decision trees. The tree builders in _tree.pyx call a
# Splitter once per node: node_reset selects the node's samples
# samples[start:end] and initializes the Criterion, and node_split searches,
# among up to max_features randomly drawn features, for the split maximizing
# the criterion's proxy improvement, subject to min_samples_leaf,
# min_weight_leaf and optional monotonicity constraints. The search is
# written once in two generic functions over a fused Partitioner type
# (DensePartitioner or SparsePartitioner from _partitioner.pyx):
# node_split_best sorts each feature and scans every distinct threshold
# (with two passes to choose the side for missing values), and
# node_split_random draws one uniform threshold per feature. The four
# concrete classes BestSplitter, BestSparseSplitter, RandomSplitter and
# RandomSparseSplitter pair a strategy with a data layout. Features found
# constant in a node are recorded so descendants skip them.
"""Splitting algorithms in the construction of a tree.

This module contains the main splitting algorithms for constructing a tree.
Splitting is concerned with finding the optimal partition of the data into
two groups. The impurity of the groups is minimized, and the impurity is measured
by some criterion, which is typically the Gini impurity or the entropy. Criterion
are implemented in the ``_criterion`` module.

Splitting evaluates a subset of features (defined by `max_features` also
known as mtry in the literature). The module supports two primary types
of splitting strategies:

- Best Split: A greedy approach to find the optimal split. This method
  ensures that the best possible split is chosen by examining various
  thresholds for each candidate feature.
- Random Split: A stochastic approach that selects a split randomly
  from a subset of the best splits. This method is faster but does
  not guarantee the optimal split.
"""
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from libc.math cimport INFINITY
from libc.string cimport memcpy

from sklearn.tree._criterion cimport Criterion
from sklearn.tree._partitioner cimport (
    FEATURE_THRESHOLD, DensePartitioner, SparsePartitioner,
)
from sklearn.tree._utils cimport RAND_R_MAX, rand_int, rand_uniform

import numpy as np

# Introduce a fused-class to make it possible to share the split implementation
# between the dense and sparse cases in the node_split_best and node_split_random
# functions. The alternative would have been to use inheritance-based polymorphism
# but it would have resulted in a ~10% overall tree fitting performance
# degradation caused by the overhead frequent virtual method lookups.
ctypedef fused Partitioner:
    DensePartitioner
    SparsePartitioner


cdef inline void _init_split(SplitRecord* self, intp_t start_pos) noexcept nogil:
    # [FUNCTION DOC]
    # HELPER: initializes a SplitRecord before a split search.
    # Behavior: sets impurities to +inf, improvement to -inf, pos to start_pos
    # (callers pass end, meaning "no split found"), feature 0, threshold 0 and
    # missing_go_to_left false; lower_bound / upper_bound are left untouched.
    # Inputs: pointer to the record; start_pos. Outputs: the record.
    # Exceptions: none (nogil). Side effects: writes the record.
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY
    self.missing_go_to_left = False

cdef class Splitter:
    # [CLASS DOC]
    # Abstract base splitter; subclasses supply node_split. Lifecycle: construct
    # with criterion and hyper-parameters, init(X, y, ...) once per fit, then for
    # each node node_reset followed by node_split, node_value and
    # node_impurity. Pickling only preserves constructor arguments.
    """Abstract splitter class.

    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    """

    def __cinit__(
        self,
        Criterion criterion,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        object random_state,
        const int8_t[:] monotonic_cst,
    ):
        # [FUNCTION DOC]
        # Stores the criterion, max_features, min_samples_leaf, min_weight_leaf,
        # random_state (numpy RandomState-like object) and monotonic_cst (int8 per
        # feature, or None, which disables constraints); zeroes the sample and
        # feature counts. Outputs: none. Exceptions: type errors on bad arguments.
        # Side effects: none beyond field initialization.
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.

        max_features : intp_t
            The maximal number of randomly selected features which can be
            considered for a split.

        min_samples_leaf : intp_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.

        min_weight_leaf : float64_t
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.

        random_state : object
            The user inputted random state to be used for pseudo-randomness

        monotonic_cst : const int8_t[:]
            Monotonicity constraints

        """

        self.criterion = criterion

        self.n_samples = 0
        self.n_features = 0

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        self.monotonic_cst = monotonic_cst
        self.with_monotonic_cst = monotonic_cst is not None

    def __getstate__(self):
        # [FUNCTION DOC]
        # Pickle support: returns an empty state (only constructor arguments are
        # preserved). No exceptions or side effects.
        return {}

    def __setstate__(self, d):
        # [FUNCTION DOC]
        # Pickle support: ignores the state dict. No effect.
        pass

    def __reduce__(self):
        # [FUNCTION DOC]
        # Pickle support: returns the class, the constructor arguments (criterion,
        # max_features, min_samples_leaf, min_weight_leaf, random_state,
        # monotonic_cst) and the empty state. Data bound by init is not pickled.
        return (type(self), (self.criterion,
                             self.max_features,
                             self.min_samples_leaf,
                             self.min_weight_leaf,
                             self.random_state,
                             self.monotonic_cst), self.__getstate__())

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        # [FUNCTION DOC]
        # Behavior: seeds rand_r_state with random_state.randint(0, RAND_R_MAX);
        # fills samples with the indices of rows whose weight is non-zero (all rows if
        # sample_weight is None) and sets n_samples to their count; sums all weights
        # into weighted_n_samples; sets features to 0..n_features-1; allocates
        # feature_values (one per row of X) and constant_features (one per feature);
        # keeps references to y and sample_weight.
        # Inputs: X, any object with a 2-D shape (dense array or sparse matrix); y,
        # 2-D float64 C-contiguous targets (n_samples, n_outputs); sample_weight,
        # 1-D float64 or None; missing_values_in_feature_mask, unused here (used by
        # subclasses for their partitioner).
        # Outputs: returns 0.
        # Exceptions: MemoryError from numpy allocations; errors raised by
        # random_state.randint. Requires the GIL.
        # Side effects: allocates buffers, advances the random_state object.
        # KNOWN ISSUE: the docstring documents a has_missing parameter that does not
        # exist; the actual parameter is missing_values_in_feature_mask and the
        # base method does not use it.
        """Initialize the splitter.

        Take in the input data X, the target Y, and optional sample weights.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.

        y : ndarray, dtype=float64_t
            This is the vector of targets, or true labels, for the samples represented
            as a Cython memoryview.

        sample_weight : ndarray, dtype=float64_t
            The weights of the samples, where higher weighted samples are fit
            closer than lower weight samples. If not provided, all samples
            are assumed to have uniform weight. This is represented
            as a Cython memoryview.

        has_missing : bool
            At least one missing values is in X.
        """

        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef intp_t n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        self.samples = np.empty(n_samples, dtype=np.intp)
        cdef intp_t[::1] samples = self.samples

        cdef intp_t i, j
        cdef float64_t weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight is None or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight is not None:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Number of samples is number of positively weighted samples
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef intp_t n_features = X.shape[1]
        self.features = np.arange(n_features, dtype=np.intp)
        self.n_features = n_features

        self.feature_values = np.empty(n_samples, dtype=np.float32)
        self.constant_features = np.empty(n_features, dtype=np.intp)

        self.y = y

        self.sample_weight = sample_weight
        return 0

    cdef int node_reset(
        self,
        intp_t start,
        intp_t end,
        float64_t* weighted_n_node_samples
    ) except -1 nogil:
        # [FUNCTION DOC]
        # Behavior: records start and end and calls criterion.init with y, weights,
        # total weighted sample count, samples and the range, so the criterion
        # computes the node statistics.
        # Inputs: start, end (end exclusive, despite the docstring calling it the
        # last sample); weighted_n_node_samples, output pointer.
        # Outputs: returns 0; writes the node's weighted sample count.
        # Exceptions: errors propagated from criterion.init (-1 return). nogil.
        # Side effects: mutates splitter and criterion state.
        """Reset splitter on node samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : intp_t
            The index of the first sample to consider
        end : intp_t
            The index of the last sample to consider
        weighted_n_node_samples : ndarray, dtype=float64_t pointer
            The total weight of those samples
        """

        self.start = start
        self.end = end

        self.criterion.init(
            self.y,
            self.sample_weight,
            self.weighted_n_samples,
            self.samples,
            start,
            end
        )

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0

    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil:
        # [FUNCTION DOC]
        # Placeholder for subclasses: should search the best split of
        # samples[start:end] given the parent information (impurity, bounds,
        # number of known constant features, updated on return) and write it to
        # split. The base implementation does nothing and returns 0, leaving split
        # unset. Exceptions: none here; subclasses return -1 on error. nogil.

        """Find the best split on node samples[start:end].

        This is a placeholder method. The majority of computation will be done
        here.

        It should return -1 upon errors.
        """

        pass

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        # [FUNCTION DOC]
        # Writes the current node's value (e.g. class weights or mean) into dest by
        # delegating to the criterion. Exceptions: none (nogil). Side effects: writes
        # dest.
        """Copy the value of node samples[start:end] into dest."""

        self.criterion.node_value(dest)

    cdef inline void clip_node_value(self, float64_t* dest, float64_t lower_bound, float64_t upper_bound) noexcept nogil:
        # [FUNCTION DOC]
        # Delegates to the criterion to clip the node value in dest to
        # [lower_bound, upper_bound] (monotonic constraints). Exceptions: none
        # (nogil). Side effects: writes dest.
        """Clip the value in dest between lower_bound and upper_bound for monotonic constraints."""

        self.criterion.clip_node_value(dest, lower_bound, upper_bound)

    cdef float64_t node_impurity(self) noexcept nogil:
        # [FUNCTION DOC]
        # Returns the impurity of the current node as computed by the criterion.
        # Exceptions: none (nogil). Side effects: none.
        """Return the impurity of the current node."""

        return self.criterion.node_impurity()


cdef inline int node_split_best(
    Splitter splitter,
    Partitioner partitioner,
    Criterion criterion,
    SplitRecord* split,
    ParentInfo* parent_record,
) except -1 nogil:
    # [FUNCTION DOC]
    # HELPER: best-split search shared by BestSplitter and BestSparseSplitter
    # (instantiated for each Partitioner type).
    # Behavior: features are drawn without replacement by a Fisher-Yates scheme
    # on the features array, which is kept segmented into: drawn known-constant
    # features, undrawn known-constant features, newly found constant features,
    # undrawn candidates, and drawn non-constant features (from the end). Known
    # constant features (from the parent) are skipped without evaluation. The
    # loop continues while undrawn non-constant candidates remain and either
    # fewer than max_features features were visited or every visited feature so
    # far was constant. For each drawn candidate the partitioner sorts the node
    # by that feature; if all values are missing, or there are no missing values
    # and max - min <= FEATURE_THRESHOLD, the feature is marked constant.
    # Otherwise one scan (two if missing values exist: missing on the right,
    # then missing on the left after shift_missing_to_the_left) walks the
    # candidate positions from next_p; positions violating min_samples_leaf,
    # monotonicity (via criterion.check_monotonicity) or min_weight_leaf are
    # rejected, and the one with the largest proxy improvement is kept. Its
    # threshold is the midpoint of the neighbouring values (computed as a sum of
    # halves to avoid overflow) or +inf for the "only missing values go right"
    # split; if the feature had no missing values, missing_go_to_left is set to
    # send future NaNs to the larger child. Finally the node samples are
    # partitioned by the best split, children impurities and the true impurity
    # improvement are computed, the known-constant prefix of features is
    # restored from constant_features and new constant features are appended to
    # constant_features for the children.
    # Inputs: splitter, partitioner, criterion; split (output); parent_record
    # (impurity, lower/upper bounds, n_constant_features; updated).
    # Outputs: returns 0; split receives the best SplitRecord (pos == end if none
    # found); parent_record.n_constant_features is updated to the new total.
    # Exceptions: -1 only if the criterion signals an error. nogil.
    # Side effects: reorders samples, feature_values and features; writes
    # constant_features; advances the splitter's RNG state; mutates the
    # criterion.
    """Find the best split on node samples[start:end]

    Returns -1 in case of failure to allocate memory (and raise MemoryError)
    or 0 otherwise.
    """
    cdef const int8_t[:] monotonic_cst = splitter.monotonic_cst
    cdef bint with_monotonic_cst = splitter.with_monotonic_cst

    # Find the best split
    cdef intp_t start = splitter.start
    cdef intp_t end = splitter.end
    cdef intp_t end_non_missing
    cdef intp_t n_missing = 0
    cdef bint has_missing = 0
    cdef intp_t n_searches
    cdef intp_t n_left, n_right
    cdef bint missing_go_to_left

    cdef intp_t[::1] features = splitter.features
    cdef intp_t[::1] constant_features = splitter.constant_features
    cdef intp_t n_features = splitter.n_features

    cdef float32_t[::1] feature_values = splitter.feature_values
    cdef intp_t max_features = splitter.max_features
    cdef intp_t min_samples_leaf = splitter.min_samples_leaf
    cdef float64_t min_weight_leaf = splitter.min_weight_leaf
    cdef uint32_t* random_state = &splitter.rand_r_state

    cdef SplitRecord best_split, current_split
    cdef float64_t current_proxy_improvement = -INFINITY
    cdef float64_t best_proxy_improvement = -INFINITY

    cdef float64_t impurity = parent_record.impurity
    cdef float64_t lower_bound = parent_record.lower_bound
    cdef float64_t upper_bound = parent_record.upper_bound

    cdef intp_t f_i = n_features
    cdef intp_t f_j
    cdef intp_t p
    cdef intp_t p_prev

    cdef intp_t n_visited_features = 0
    # Number of features discovered to be constant during the split search
    cdef intp_t n_found_constants = 0
    # Number of features known to be constant and drawn without replacement
    cdef intp_t n_drawn_constants = 0
    cdef intp_t n_known_constants = parent_record.n_constant_features
    # n_total_constants = n_known_constants + n_found_constants
    cdef intp_t n_total_constants = n_known_constants

    _init_split(&best_split, end)

    partitioner.init_node_split(start, end)

    # Sample up to max_features without replacement using a
    # Fisher-Yates-based algorithm (using the local variables `f_i` and
    # `f_j` to compute a permutation of the `features` array).
    #
    # Skip the CPU intensive evaluation of the impurity criterion for
    # features that were already detected as constant (hence not suitable
    # for good splitting) by ancestor nodes and save the information on
    # newly discovered constant features to spare computation on descendant
    # nodes.
    while (f_i > n_total_constants and  # Stop early if remaining features
                                        # are constant
            (n_visited_features < max_features or
             # At least one drawn features must be non constant
             n_visited_features <= n_found_constants + n_drawn_constants)):

        n_visited_features += 1

        # Loop invariant: elements of features in
        # - [:n_drawn_constant[ holds drawn and known constant features;
        # - [n_drawn_constant:n_known_constant[ holds known constant
        #   features that haven't been drawn yet;
        # - [n_known_constant:n_total_constant[ holds newly found constant
        #   features;
        # - [n_total_constant:f_i[ holds features that haven't been drawn
        #   yet and aren't constant apriori.
        # - [f_i:n_features[ holds features that have been drawn
        #   and aren't constant.

        # Draw a feature at random
        f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                       random_state)

        if f_j < n_known_constants:
            # f_j in the interval [n_drawn_constants, n_known_constants[
            features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]

            n_drawn_constants += 1
            continue

        # f_j in the interval [n_known_constants, f_i - n_found_constants[
        f_j += n_found_constants
        # f_j in the interval [n_total_constants, f_i[
        current_split.feature = features[f_j]
        partitioner.sort_samples_and_feature_values(current_split.feature)
        n_missing = partitioner.n_missing
        end_non_missing = end - n_missing

        if (
            # All values for this feature are missing, or
            end_non_missing == start or
            # This feature is considered constant (max - min <= FEATURE_THRESHOLD)
            ((
                feature_values[end_non_missing - 1]
                <= feature_values[start] + FEATURE_THRESHOLD
            ) and n_missing == 0)
        ):
            # We consider this feature constant in this case.
            # Since finding a split among constant feature is not valuable,
            # we do not consider this feature for splitting.
            features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]

            n_found_constants += 1
            n_total_constants += 1
            continue

        f_i -= 1
        features[f_i], features[f_j] = features[f_j], features[f_i]
        has_missing = n_missing != 0

        # Evaluate all splits

        # If there are missing values, then we search twice for the most optimal split.
        # The first search will have all the missing values going to the right node
        # and the split with right node being only missing values is evaluated.
        # The second search will have all the missing values going to the left node.
        # This logic is governed by the partitionner and used here, so there is a strong coupling.
        # If there are no missing values, then we search only once for the most
        # optimal split.
        n_searches = 2 if has_missing else 1

        for i in range(n_searches):
            missing_go_to_left = i == 1
            if missing_go_to_left:
                partitioner.shift_missing_to_the_left()

            criterion.reset()

            p = start

            while p < end:
                partitioner.next_p(&p_prev, &p, missing_go_to_left)
                if p == end:
                    continue

                # Reject if min_samples_leaf is not guaranteed
                n_left = p - start
                n_right = end - p
                if n_left < min_samples_leaf or n_right < min_samples_leaf:
                    continue

                current_split.pos = p
                criterion.update(current_split.pos)

                # Reject if monotonicity constraints are not satisfied
                if (
                    with_monotonic_cst and
                    monotonic_cst[current_split.feature] != 0 and
                    not criterion.check_monotonicity(
                        monotonic_cst[current_split.feature],
                        lower_bound,
                        upper_bound,
                    )
                ):
                    continue

                # Reject if min_weight_leaf is not satisfied
                if ((criterion.weighted_n_left < min_weight_leaf) or
                        (criterion.weighted_n_right < min_weight_leaf)):
                    continue

                current_proxy_improvement = criterion.proxy_impurity_improvement()

                if current_proxy_improvement > best_proxy_improvement:
                    best_proxy_improvement = current_proxy_improvement
                    if p == end_non_missing and not missing_go_to_left:
                        # Split with the right node being only the missing values.
                        # Note that partioner.next_p never considers candidate
                        # splits for which the left node would move only the
                        # the missing values as this would be redundant with the
                        # split that only send missing values to the right.
                        # We use inf as a threshold because nan <= inf is false
                        # according to IEEE 754.
                        current_split.threshold = INFINITY
                    else:
                        # Split between two non-missing values: sum of halves is
                        # used to avoid infinite value.
                        current_split.threshold = (
                            feature_values[p_prev] / 2.0 + feature_values[p] / 2.0
                        )

                    # If there are no missing values in the training data, during
                    # test time, we send missing values to the branch that contains
                    # the most samples during training time.
                    if n_missing == 0:
                        current_split.missing_go_to_left = n_left > n_right
                    else:
                        current_split.missing_go_to_left = missing_go_to_left

                    best_split = current_split  # copy

    # Reorganize into samples[start:best_split.pos] + samples[best_split.pos:end]
    if best_split.pos < end:
        partitioner.partition_samples_final(
            &best_split
        )

        criterion.reset()
        criterion.update(best_split.pos)
        criterion.children_impurity(
            &best_split.impurity_left, &best_split.impurity_right
        )

        best_split.improvement = criterion.impurity_improvement(
            impurity,
            best_split.impurity_left,
            best_split.impurity_right
        )

    # Respect invariant for constant features: the original order of
    # element in features[:n_known_constants] must be preserved for sibling
    # and child nodes
    memcpy(&features[0], &constant_features[0], sizeof(intp_t) * n_known_constants)

    # Copy newly found constant features
    memcpy(&constant_features[n_known_constants],
           &features[n_known_constants],
           sizeof(intp_t) * n_found_constants)

    # Return values
    parent_record.n_constant_features = n_total_constants
    split[0] = best_split
    return 0


cdef inline int node_split_random(
    Splitter splitter,
    Partitioner partitioner,
    Criterion criterion,
    SplitRecord* split,
    ParentInfo* parent_record,
) except -1 nogil:
    # [FUNCTION DOC]
    # HELPER: random-split search shared by RandomSplitter and
    # RandomSparseSplitter (instantiated for each Partitioner type).
    # Behavior: same Fisher-Yates feature sampling and constant-feature
    # bookkeeping as node_split_best, but for each candidate the partitioner only
    # computes min and max (a feature is constant if all values are missing, or
    # there are no missing values and max - min <= FEATURE_THRESHOLD). A
    # threshold is drawn uniformly in [min, max] (replaced by min if it equals
    # max); when missing values exist, their direction is chosen by a fair coin.
    # The node is partitioned by the threshold, then the split is rejected on
    # min_samples_leaf, min_weight_leaf or monotonicity violations, and the one
    # with the largest proxy improvement is kept (missing_go_to_left set to the
    # larger child when there were no missing values). At the end the samples are
    # re-partitioned by the best split unless the last feature partitioned was
    # already the best one (features are never drawn twice, so the layout is
    # still valid), then child impurities and improvement are computed and the
    # constant-feature arrays are updated as in node_split_best.
    # Inputs/outputs: as node_split_best.
    # Exceptions: -1 only if the criterion signals an error. nogil.
    # Side effects: reorders samples, feature_values and features; writes
    # constant_features; advances the RNG state; mutates the criterion.
    """Find the best random split on node samples[start:end]

    Returns -1 in case of failure to allocate memory (and raise MemoryError)
    or 0 otherwise.
    """
    cdef const int8_t[:] monotonic_cst = splitter.monotonic_cst
    cdef bint with_monotonic_cst = splitter.with_monotonic_cst

    # Draw random splits and pick the best
    cdef intp_t start = splitter.start
    cdef intp_t end = splitter.end
    cdef intp_t n_missing = 0
    cdef bint has_missing = 0
    cdef intp_t n_left, n_right
    cdef bint missing_go_to_left

    cdef intp_t[::1] features = splitter.features
    cdef intp_t[::1] constant_features = splitter.constant_features
    cdef intp_t n_features = splitter.n_features

    cdef intp_t max_features = splitter.max_features
    cdef intp_t min_samples_leaf = splitter.min_samples_leaf
    cdef float64_t min_weight_leaf = splitter.min_weight_leaf
    cdef uint32_t* random_state = &splitter.rand_r_state

    cdef SplitRecord best_split, current_split
    cdef float64_t current_proxy_improvement = - INFINITY
    cdef float64_t best_proxy_improvement = - INFINITY

    cdef float64_t impurity = parent_record.impurity
    cdef float64_t lower_bound = parent_record.lower_bound
    cdef float64_t upper_bound = parent_record.upper_bound

    cdef intp_t f_i = n_features
    cdef intp_t f_j
    # Number of features discovered to be constant during the split search
    cdef intp_t n_found_constants = 0
    # Number of features known to be constant and drawn without replacement
    cdef intp_t n_drawn_constants = 0
    cdef intp_t n_known_constants = parent_record.n_constant_features
    # n_total_constants = n_known_constants + n_found_constants
    cdef intp_t n_total_constants = n_known_constants
    cdef intp_t n_visited_features = 0
    cdef float32_t min_feature_value
    cdef float32_t max_feature_value

    _init_split(&best_split, end)

    partitioner.init_node_split(start, end)

    # Sample up to max_features without replacement using a
    # Fisher-Yates-based algorithm (using the local variables `f_i` and
    # `f_j` to compute a permutation of the `features` array).
    #
    # Skip the CPU intensive evaluation of the impurity criterion for
    # features that were already detected as constant (hence not suitable
    # for good splitting) by ancestor nodes and save the information on
    # newly discovered constant features to spare computation on descendant
    # nodes.
    while (f_i > n_total_constants and  # Stop early if remaining features
                                        # are constant
            (n_visited_features < max_features or
             # At least one drawn features must be non constant
             n_visited_features <= n_found_constants + n_drawn_constants)):
        n_visited_features += 1

        # Loop invariant: elements of features in
        # - [:n_drawn_constant[ holds drawn and known constant features;
        # - [n_drawn_constant:n_known_constant[ holds known constant
        #   features that haven't been drawn yet;
        # - [n_known_constant:n_total_constant[ holds newly found constant
        #   features;
        # - [n_total_constant:f_i[ holds features that haven't been drawn
        #   yet and aren't constant apriori.
        # - [f_i:n_features[ holds features that have been drawn
        #   and aren't constant.

        # Draw a feature at random
        f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                       random_state)

        if f_j < n_known_constants:
            # f_j in the interval [n_drawn_constants, n_known_constants[
            features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
            n_drawn_constants += 1
            continue

        # f_j in the interval [n_known_constants, f_i - n_found_constants[
        f_j += n_found_constants
        # f_j in the interval [n_total_constants, f_i[

        current_split.feature = features[f_j]

        # Find min, max as we will randomly select a threshold between them
        partitioner.find_min_max(
            current_split.feature, &min_feature_value, &max_feature_value
        )
        n_missing = partitioner.n_missing

        if (
            # All values for this feature are missing, or
            end - start == n_missing or
            # This feature is considered constant (max - min <= FEATURE_THRESHOLD)
            (max_feature_value <= min_feature_value + FEATURE_THRESHOLD and n_missing == 0)
        ):
            # We consider this feature constant in this case.
            # Since finding a split with a constant feature is not valuable,
            # we do not consider this feature for splitting.
            features[f_j], features[n_total_constants] = features[n_total_constants], current_split.feature

            n_found_constants += 1
            n_total_constants += 1
            continue

        f_i -= 1
        features[f_i], features[f_j] = features[f_j], features[f_i]
        has_missing = n_missing != 0

        # Draw a random threshold
        current_split.threshold = rand_uniform(
            min_feature_value,
            max_feature_value,
            random_state,
        )

        if has_missing:
            # If there are missing values, then we randomly make all missing
            # values go to the right or left.
            #
            # Note: compared to the BestSplitter, we do not evaluate the
            # edge case where all the missing values go to the right node
            # and the non-missing values go to the left node. This is because
            # this would indicate a threshold outside of the observed range
            # of the feature. However, it is not clear how much probability weight should
            # be given to this edge case.
            missing_go_to_left = rand_int(0, 2, random_state)
        else:
            missing_go_to_left = 0

        if current_split.threshold == max_feature_value:
            current_split.threshold = min_feature_value

        # Partition
        current_split.pos = partitioner.partition_samples(
            current_split.threshold, missing_go_to_left
        )

        n_left = current_split.pos - start
        n_right = end - current_split.pos

        # Reject if min_samples_leaf is not guaranteed
        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            continue

        # Evaluate split
        # At this point, the criterion has a view into the samples that was partitioned
        # by the partitioner. The criterion will use the partition to evaluating the split.
        criterion.reset()
        criterion.update(current_split.pos)

        # Reject if min_weight_leaf is not satisfied
        if ((criterion.weighted_n_left < min_weight_leaf) or
                (criterion.weighted_n_right < min_weight_leaf)):
            continue

        # Reject if monotonicity constraints are not satisfied
        if (
                with_monotonic_cst and
                monotonic_cst[current_split.feature] != 0 and
                not criterion.check_monotonicity(
                    monotonic_cst[current_split.feature],
                    lower_bound,
                    upper_bound,
                )
        ):
            continue

        current_proxy_improvement = criterion.proxy_impurity_improvement()

        if current_proxy_improvement > best_proxy_improvement:
            # if there are no missing values in the training data, during
            # test time, we send missing values to the branch that contains
            # the most samples during training time.
            if has_missing:
                current_split.missing_go_to_left = missing_go_to_left
            else:
                current_split.missing_go_to_left = n_left > n_right

            best_proxy_improvement = current_proxy_improvement
            best_split = current_split  # copy

    # Reorganize into samples[start:best.pos] + samples[best.pos:end]
    if best_split.pos < end:
        if current_split.feature != best_split.feature:
            partitioner.partition_samples_final(
                &best_split
            )

        criterion.reset()
        criterion.update(best_split.pos)
        criterion.children_impurity(
            &best_split.impurity_left, &best_split.impurity_right
        )
        best_split.improvement = criterion.impurity_improvement(
            impurity,
            best_split.impurity_left,
            best_split.impurity_right
        )

    # Respect invariant for constant features: the original order of
    # element in features[:n_known_constants] must be preserved for sibling
    # and child nodes
    memcpy(&features[0], &constant_features[0], sizeof(intp_t) * n_known_constants)

    # Copy newly found constant features
    memcpy(&constant_features[n_known_constants],
           &features[n_known_constants],
           sizeof(intp_t) * n_found_constants)

    # Return values
    parent_record.n_constant_features = n_total_constants
    split[0] = best_split
    return 0


cdef class BestSplitter(Splitter):
    # [CLASS DOC]
    # Best-split strategy on dense X, using a DensePartitioner (supports NaN
    # missing values).
    """Splitter for finding the best split on dense data."""
    cdef DensePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        # [FUNCTION DOC]
        # Behavior: runs the base Splitter.init (sample list, weights, buffers) and
        # then builds the dense partitioner over X sharing the splitter's samples
        # and feature_values buffers.
        # Inputs: X (2-D float32 array), y, sample_weight (or None), missing-value mask per
        # feature (or None).
        # Outputs: returns 0 on success (implicit).
        # Exceptions: those of Splitter.init and of the partitioner constructor.
        # Side effects: allocates buffers; consumes one draw of random_state.
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = DensePartitioner(
            X, self.samples, self.feature_values, missing_values_in_feature_mask
        )

    cdef int node_split(
            self,
            ParentInfo* parent_record,
            SplitRecord* split,
    ) except -1 nogil:
        # [FUNCTION DOC]
        # Delegates to node_split_best with this splitter, its dense partitioner and its
        # criterion. Inputs/outputs as Splitter.node_split. Exceptions: none in
        # practice (error return -1 propagated). Runs without the GIL.
        return node_split_best(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )

cdef class BestSparseSplitter(Splitter):
    # [CLASS DOC]
    # Best-split strategy on CSC sparse X, using a SparsePartitioner (no missing
    # value support).
    """Splitter for finding the best split, using the sparse data."""
    cdef SparsePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        # [FUNCTION DOC]
        # Behavior: runs the base Splitter.init (sample list, weights, buffers) and
        # then builds the sparse partitioner over X sharing the splitter's samples
        # and feature_values buffers.
        # Inputs: X (CSC sparse matrix), y, sample_weight (or None), missing-value mask per
        # feature (or None).
        # Outputs: returns 0 on success (implicit).
        # Exceptions: those of Splitter.init and of the partitioner constructor (ValueError if X is not CSC).
        # Side effects: allocates buffers; consumes one draw of random_state.
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = SparsePartitioner(
            X, self.samples, self.n_samples, self.feature_values, missing_values_in_feature_mask
        )

    cdef int node_split(
            self,
            ParentInfo* parent_record,
            SplitRecord* split,
    ) except -1 nogil:
        # [FUNCTION DOC]
        # Delegates to node_split_best with this splitter, its sparse partitioner and its
        # criterion. Inputs/outputs as Splitter.node_split. Exceptions: none in
        # practice (error return -1 propagated). Runs without the GIL.
        return node_split_best(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )

cdef class RandomSplitter(Splitter):
    # [CLASS DOC]
    # Random-threshold strategy (extremely randomized trees) on dense X, using a
    # DensePartitioner.
    """Splitter for finding the best random split on dense data."""
    cdef DensePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        # [FUNCTION DOC]
        # Behavior: runs the base Splitter.init (sample list, weights, buffers) and
        # then builds the dense partitioner over X sharing the splitter's samples
        # and feature_values buffers.
        # Inputs: X (2-D float32 array), y, sample_weight (or None), missing-value mask per
        # feature (or None).
        # Outputs: returns 0 on success (implicit).
        # Exceptions: those of Splitter.init and of the partitioner constructor.
        # Side effects: allocates buffers; consumes one draw of random_state.
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = DensePartitioner(
            X, self.samples, self.feature_values, missing_values_in_feature_mask
        )

    cdef int node_split(
            self,
            ParentInfo* parent_record,
            SplitRecord* split,
    ) except -1 nogil:
        # [FUNCTION DOC]
        # Delegates to node_split_random with this splitter, its dense partitioner and its
        # criterion. Inputs/outputs as Splitter.node_split. Exceptions: none in
        # practice (error return -1 propagated). Runs without the GIL.
        return node_split_random(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )

cdef class RandomSparseSplitter(Splitter):
    # [CLASS DOC]
    # Random-threshold strategy on CSC sparse X, using a SparsePartitioner.
    """Splitter for finding the best random split, using the sparse data."""
    cdef SparsePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        # [FUNCTION DOC]
        # Behavior: runs the base Splitter.init (sample list, weights, buffers) and
        # then builds the sparse partitioner over X sharing the splitter's samples
        # and feature_values buffers.
        # Inputs: X (CSC sparse matrix), y, sample_weight (or None), missing-value mask per
        # feature (or None).
        # Outputs: returns 0 on success (implicit).
        # Exceptions: those of Splitter.init and of the partitioner constructor (ValueError if X is not CSC).
        # Side effects: allocates buffers; consumes one draw of random_state.
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = SparsePartitioner(
            X, self.samples, self.n_samples, self.feature_values, missing_values_in_feature_mask
        )
    cdef int node_split(
            self,
            ParentInfo* parent_record,
            SplitRecord* split,
    ) except -1 nogil:
        # [FUNCTION DOC]
        # Delegates to node_split_random with this splitter, its sparse partitioner and its
        # criterion. Inputs/outputs as Splitter.node_split. Exceptions: none in
        # practice (error return -1 propagated). Runs without the GIL.
        return node_split_random(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )
