# [MODULE DOC]
# HELPER: Common private helpers shared by sklearn.metrics to extend binary metrics to
#   multilabel and multiclass problems: per-label averaging of a binary metric
#   (_average_binary_score) and Hand and Till one-vs-one averaging
#   (_average_multiclass_ovo_score).
"""
Common code for all metrics.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from itertools import combinations

import numpy as np

import sklearn.externals.array_api_extra as xpx
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils._array_api import (
    _average,
    _ravel,
    get_namespace_and_device,
)
from sklearn.utils.multiclass import type_of_target


def _average_binary_score(binary_metric, y_true, y_score, average, sample_weight=None):
    # [FUNCTION DOC]
    # HELPER: Applies binary_metric per label (or per sample) and aggregates according
    #   to average.
    # Steps: validate average and the target type; if y_true is binary, return
    #   binary_metric on the inputs directly (average ignored). Otherwise validate
    #   lengths and arrays, then: for "micro" flatten y_true and y_score (repeating
    #   sample_weight once per label) and treat the result as a single label; for
    #   "weighted" compute per-label weights as the (sample-weighted) count of positives
    #   and return 0 if they sum to 0; for "samples" iterate over rows instead of
    #   columns and use sample_weight as averaging weights rather than metric weights.
    #   Compute binary_metric for each column (or row), set scores with zero averaging
    #   weight to 0 (so NaNs do not propagate) and return their weighted mean.
    # Inputs: binary_metric - callable(y_true, y_score, sample_weight=...) returning a
    #   float; y_true - binary vector or (n_samples, n_labels) indicator matrix; y_score
    #   - scores of the same shape; average - None, "micro", "macro", "weighted" or
    #   "samples" (no default); sample_weight - optional (n_samples,) weights.
    # Outputs: float for an averaged score, otherwise an array of per-label scores (in
    #   the input's array namespace/device). Returns the integer 0 for "weighted" with
    #   no positives.
    # Exceptions: ValueError for an unknown average or a target type other than binary
    #   or multilabel-indicator; errors from check_consistent_length/check_array.
    # Side effects: none.
    """Average a binary metric for multilabel classification.

    Parameters
    ----------
    binary_metric : callable, returns shape [n_classes]
        The binary metric function to use.

    y_true : array, shape = [n_samples] or [n_samples, n_classes]
        True binary labels in binary label indicators.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class or non-thresholded decision values (as returned by
        :term:`decision_function` on some classifiers).

    average : {None, 'micro', 'macro', 'samples', 'weighted'}, default='macro'
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        ``'samples'``:
            Calculate metrics for each instance, and find their average.

        Will be ignored when ``y_true`` is binary.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    score : float or array of shape [n_classes]
        If not ``None``, average the score, else return the score for each
        classes.

    """
    xp, _, _device = get_namespace_and_device(y_score, sample_weight)
    average_options = (None, "micro", "macro", "weighted", "samples")
    if average not in average_options:
        raise ValueError("average has to be one of {0}".format(average_options))

    y_type = type_of_target(y_true)
    if y_type not in ("binary", "multilabel-indicator"):
        raise ValueError("{0} format is not supported".format(y_type))

    if y_type == "binary":
        return binary_metric(y_true, y_score, sample_weight=sample_weight)

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = check_array(y_true)
    y_score = check_array(y_score)

    not_average_axis = 1
    score_weight = sample_weight
    average_weight = None

    if average == "micro":
        if score_weight is not None:
            score_weight = xp.repeat(score_weight, y_true.shape[1])
        y_true = _ravel(y_true)
        y_score = _ravel(y_score)

    elif average == "weighted":
        if score_weight is not None:
            #  Mixed integer and float type promotion not defined in array standard
            y_true = xp.asarray(y_true, dtype=score_weight.dtype)
            average_weight = xp.sum(
                xp.multiply(y_true, xp.reshape(score_weight, (-1, 1))), axis=0
            )
        else:
            average_weight = xp.sum(y_true, axis=0)
        if xpx.isclose(
            xp.sum(average_weight),
            xp.asarray(0, dtype=average_weight.dtype, device=_device),
        ):
            return 0

    elif average == "samples":
        # swap average_weight <-> score_weight
        average_weight = score_weight
        score_weight = None
        not_average_axis = 0

    if y_true.ndim == 1:
        y_true = xp.reshape(y_true, (-1, 1))

    if y_score.ndim == 1:
        y_score = xp.reshape(y_score, (-1, 1))

    n_classes = y_score.shape[not_average_axis]
    score = xp.zeros((n_classes,), device=_device)
    for c in range(n_classes):
        y_true_c = _ravel(
            xp.take(y_true, xp.asarray([c], device=_device), axis=not_average_axis)
        )
        y_score_c = _ravel(
            xp.take(y_score, xp.asarray([c], device=_device), axis=not_average_axis)
        )
        score[c] = binary_metric(y_true_c, y_score_c, sample_weight=score_weight)

    # Average the results
    if average is not None:
        if average_weight is not None:
            # Scores with 0 weights are forced to be 0, preventing the average
            # score from being affected by 0-weighted NaN elements.
            score[average_weight == 0] = 0
        return float(_average(score, weights=average_weight, xp=xp))
    else:
        return score


def _average_multiclass_ovo_score(binary_metric, y_true, y_score, average="macro"):
    # [FUNCTION DOC]
    # HELPER: Hand and Till (2001) one-vs-one averaging. For every unordered pair of
    #   classes (a, b) among the sorted unique labels of y_true, restricts to samples
    #   whose true label is a or b, computes binary_metric with a as positive class
    #   using score column a, and with b as positive using column b, and stores the mean
    #   of the two. Returns the mean of the pair scores, weighted by the fraction of
    #   samples belonging to each pair when average="weighted" (any other value behaves
    #   as "macro").
    # Inputs: binary_metric - callable(y_true_bool, y_score_1d) returning a float;
    #   y_true - (n_samples,) labels; y_score - (n_samples, n_classes) array; average -
    #   "macro" (default) or "weighted".
    # Outputs: float (numpy scalar) average score.
    # Exceptions: ValueError from check_consistent_length on length mismatch; IndexError
    #   if labels are not valid column indices.
    # Side effects: none.
    # KNOWN ISSUE: class labels are used directly as column indices into y_score, so the
    #   labels must already be encoded as 0..n_classes-1 by the caller; also unknown
    #   average values are not rejected and silently fall back to macro.
    """Average one-versus-one scores for multiclass classification.

    Uses the binary metric for one-vs-one multiclass classification,
    where the score is computed according to the Hand & Till (2001) algorithm.

    Parameters
    ----------
    binary_metric : callable
        The binary metric function to use that accepts the following as input:
            y_true_target : array, shape = [n_samples_target]
                Some sub-array of y_true for a pair of classes designated
                positive and negative in the one-vs-one scheme.
            y_score_target : array, shape = [n_samples_target]
                Scores corresponding to the probability estimates
                of a sample belonging to the designated positive class label

    y_true : array-like of shape (n_samples,)
        True multiclass labels.

    y_score : array-like of shape (n_samples, n_classes)
        Target scores corresponding to probability estimates of a sample
        belonging to a particular class.

    average : {'macro', 'weighted'}, default='macro'
        Determines the type of averaging performed on the pairwise binary
        metric scores:
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean. This does not take label imbalance into account. Classes
            are assumed to be uniformly distributed.
        ``'weighted'``:
            Calculate metrics for each label, taking into account the
            prevalence of the classes.

    Returns
    -------
    score : float
        Average of the pairwise binary metric scores.
    """
    check_consistent_length(y_true, y_score)

    y_true_unique = np.unique(y_true)
    n_classes = y_true_unique.shape[0]
    n_pairs = n_classes * (n_classes - 1) // 2
    pair_scores = np.empty(n_pairs)

    is_weighted = average == "weighted"
    prevalence = np.empty(n_pairs) if is_weighted else None

    # Compute scores treating a as positive class and b as negative class,
    # then b as positive class and a as negative class
    for ix, (a, b) in enumerate(combinations(y_true_unique, 2)):
        a_mask = y_true == a
        b_mask = y_true == b
        ab_mask = np.logical_or(a_mask, b_mask)

        if is_weighted:
            prevalence[ix] = np.average(ab_mask)

        a_true = a_mask[ab_mask]
        b_true = b_mask[ab_mask]

        a_true_score = binary_metric(a_true, y_score[ab_mask, a])
        b_true_score = binary_metric(b_true, y_score[ab_mask, b])
        pair_scores[ix] = (a_true_score + b_true_score) / 2

    return np.average(pair_scores, weights=prevalence)
