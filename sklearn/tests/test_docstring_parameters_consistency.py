# [MODULE DOC]
# Tests that shared parameter/attribute descriptions are written identically
# across related public objects (bagging/isolation forest, stacking estimators,
# and precision/recall/F-score metrics). Module-level lists
# CLASS_DOCSTRING_CONSISTENCY_CASES and FUNCTION_DOCSTRING_CONSISTENCY_CASES hold
# keyword-argument dicts for assert_docstring_consistency.
# KNOWN ISSUE: the first class case passes ignore_types as ("max_samples"), which
# is a plain string rather than a one-element tuple (missing trailing comma).

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from sklearn import metrics
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    IsolationForest,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.utils._testing import assert_docstring_consistency, skip_if_no_numpydoc

CLASS_DOCSTRING_CONSISTENCY_CASES = [
    {
        "objects": [BaggingClassifier, BaggingRegressor, IsolationForest],
        "include_params": ["max_samples"],
        "exclude_params": None,
        "include_attrs": False,
        "exclude_attrs": None,
        "include_returns": False,
        "exclude_returns": None,
        "descr_regex_pattern": r"The number of samples to draw from X to train each.*",
        "ignore_types": ("max_samples"),
    },
    {
        "objects": [StackingClassifier, StackingRegressor],
        "include_params": ["cv", "n_jobs", "passthrough", "verbose"],
        "exclude_params": None,
        "include_attrs": True,
        "exclude_attrs": ["final_estimator_"],
        "include_returns": False,
        "exclude_returns": None,
        "descr_regex_pattern": None,
    },
]

FUNCTION_DOCSTRING_CONSISTENCY_CASES = [
    {
        "objects": [
            metrics.precision_recall_fscore_support,
            metrics.f1_score,
            metrics.fbeta_score,
            metrics.precision_score,
            metrics.recall_score,
        ],
        "include_params": True,
        "exclude_params": ["average", "zero_division"],
        "include_attrs": False,
        "exclude_attrs": None,
        "include_returns": False,
        "exclude_returns": None,
        "descr_regex_pattern": None,
    },
    {
        "objects": [
            metrics.precision_recall_fscore_support,
            metrics.f1_score,
            metrics.fbeta_score,
            metrics.precision_score,
            metrics.recall_score,
        ],
        "include_params": ["average"],
        "exclude_params": None,
        "include_attrs": False,
        "exclude_attrs": None,
        "include_returns": False,
        "exclude_returns": None,
        "descr_regex_pattern": " ".join(
            (
                r"""This parameter is required for multiclass/multilabel targets\.
            If ``None``, the metrics for each class are returned\. Otherwise, this
            determines the type of averaging performed on the data:
            ``'binary'``:
                Only report results for the class specified by ``pos_label``\.
                This is applicable only if targets \(``y_\{true,pred\}``\) are binary\.
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives\.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean\.  This does not take label imbalance into account\.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support \(the number of true instances for each label\)\. This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall\."""
                r"[\s\w]*\.*"  # optionally match additional sentence
                r"""
            ``'samples'``:
                Calculate metrics for each instance, and find their average \(only
                meaningful for multilabel classification where this differs from
                :func:`accuracy_score`\)\."""
            ).split()
        ),
    },
]


@pytest.mark.parametrize("case", CLASS_DOCSTRING_CONSISTENCY_CASES)
@skip_if_no_numpydoc
def test_class_docstring_consistency(case):
    # [FUNCTION DOC]
    # Parametrized over CLASS_DOCSTRING_CONSISTENCY_CASES; forwards each case dict to
    # assert_docstring_consistency, which raises if the selected docstring sections
    # differ. Skipped when numpydoc is not installed.
    """Check docstrings parameters consistency between related classes."""
    assert_docstring_consistency(**case)


@pytest.mark.parametrize("case", FUNCTION_DOCSTRING_CONSISTENCY_CASES)
@skip_if_no_numpydoc
def test_function_docstring_consistency(case):
    # [FUNCTION DOC]
    # Same as test_class_docstring_consistency but over
    # FUNCTION_DOCSTRING_CONSISTENCY_CASES (metric functions), including a regex check
    # on the shared "average" parameter description.
    """Check docstrings parameters consistency between related functions."""
    assert_docstring_consistency(**case)
