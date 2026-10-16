# [MODULE DOC]
# Computes parameter bounds for l1-penalized linear classifiers. Currently only
# provides l1_min_c, the smallest C above which an l1-penalized model is
# guaranteed to have at least one non-zero coefficient.
"""Determination of parameter bounds"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Real

import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_array, check_consistent_length


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "y": ["array-like"],
        "loss": [StrOptions({"squared_hinge", "log"})],
        "fit_intercept": ["boolean"],
        "intercept_scaling": [Interval(Real, 0, None, closed="neither")],
    },
    prefer_skip_nested_validation=True,
)
def l1_min_c(X, y, *, loss="squared_hinge", fit_intercept=True, intercept_scaling=1.0):
    # [FUNCTION DOC]
    # Computes the critical C below which an l1-penalized squared-hinge or logistic
    # model has all-zero coefficients. Steps: validate X (sparse converted to CSC)
    # and check X and y lengths match; binarize y into a {-1, +1} indicator matrix
    # Y (one row per class, a single row for binary targets) via LabelBinarizer;
    # den is the largest absolute entry of Y times X (per class, per feature
    # correlation). If fit_intercept, also consider the synthetic constant column
    # of value intercept_scaling: den becomes the max of den and the largest
    # absolute value of Y times that column. Return 0.5/den for squared_hinge and
    # 2.0/den for log.
    # Inputs: X - array-like or sparse (n_samples, n_features); y - labels
    # (n_samples,); loss - "squared_hinge" (default) or "log"; fit_intercept - bool
    # (default True); intercept_scaling - positive float (default 1.0).
    # Outputs: float, the minimal C.
    # Exceptions: ValueError if den is 0 (all correlations zero, ill-posed);
    # validation errors from validate_params / check_array / check_consistent_length.
    # Side effects: none.
    # Note: the bound ignores class_weight and sample weights (see docstring).
    """Return the lowest bound for `C`.

    The lower bound for `C` is computed such that for `C` in `(l1_min_C, infinity)`
    the model is guaranteed not to be empty. This applies to l1 penalized
    classifiers, such as :class:`sklearn.svm.LinearSVC` with penalty='l1' and
    :class:`sklearn.linear_model.LogisticRegression` with `l1_ratio=1`.

    This value is valid if `class_weight` parameter in `fit()` is not set.

    For an example of how to use this function, see
    :ref:`sphx_glr_auto_examples_linear_model_plot_logistic_path.py`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples,)
        Target vector relative to X.

    loss : {'squared_hinge', 'log'}, default='squared_hinge'
        Specifies the loss function.
        With 'squared_hinge' it is the squared hinge loss (a.k.a. L2 loss).
        With 'log' it is the loss of logistic regression models.

    fit_intercept : bool, default=True
        Specifies if the intercept should be fitted by the model.
        It must match the fit() method parameter.

    intercept_scaling : float, default=1.0
        When fit_intercept is True, instance vector x becomes
        [x, intercept_scaling],
        i.e. a "synthetic" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        It must match the fit() method parameter.

    Returns
    -------
    l1_min_c : float
        Minimum value for C.

    Examples
    --------
    >>> from sklearn.svm import l1_min_c
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    >>> print(f"{l1_min_c(X, y, loss='squared_hinge', fit_intercept=True):.4f}")
    0.0044
    """

    X = check_array(X, accept_sparse="csc")
    check_consistent_length(X, y)

    Y = LabelBinarizer(neg_label=-1).fit_transform(y).T
    # maximum absolute value over classes and features
    den = np.max(np.abs(safe_sparse_dot(Y, X)))
    if fit_intercept:
        bias = np.full(
            (np.size(y), 1), intercept_scaling, dtype=np.array(intercept_scaling).dtype
        )
        den = max(den, abs(np.dot(Y, bias)).max())

    if den == 0.0:
        raise ValueError(
            "Ill-posed l1_min_c calculation: l1 will always "
            "select zero coefficients for this data"
        )
    if loss == "squared_hinge":
        return 0.5 / den
    else:  # loss == 'log':
        return 2.0 / den
