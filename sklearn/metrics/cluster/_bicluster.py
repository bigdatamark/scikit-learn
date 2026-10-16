# [MODULE DOC]
# Bicluster comparison metric. Provides consensus_score, which matches two sets of
# biclusters (each given as row and column boolean indicator matrices) with an optimal
# one-to-one assignment and averages their pairwise similarity.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy.optimize import linear_sum_assignment

from sklearn.utils._param_validation import StrOptions, validate_params
from sklearn.utils.validation import check_array, check_consistent_length

__all__ = ["consensus_score"]


def _check_rows_and_columns(a, b):
    # [FUNCTION DOC]
    # HELPER: Validates and unpacks two bicluster specifications.
    # Inputs: a, b - tuples (rows, cols); rows is (n_biclusters, n_rows) and cols is
    # (n_biclusters, n_cols) indicator arrays; within each tuple the number of
    # biclusters must match.
    # Outputs: four arrays a_rows, a_cols, b_rows, b_cols, converted with check_array
    # (1d allowed).
    # Exceptions: ValueError from check_consistent_length/check_array on inconsistent
    # lengths or invalid content.
    # Side effects: none.
    """Unpacks the row and column arrays and checks their shape."""
    check_consistent_length(*a)
    check_consistent_length(*b)
    checks = lambda x: check_array(x, ensure_2d=False)
    a_rows, a_cols = map(checks, a)
    b_rows, b_cols = map(checks, b)
    return a_rows, a_cols, b_rows, b_cols


def _jaccard(a_rows, a_cols, b_rows, b_cols):
    # [FUNCTION DOC]
    # HELPER: Jaccard coefficient between two biclusters treated as sets of matrix
    # cells. The intersection size is (number of shared rows) times (number of shared
    # columns); each bicluster's size is its row count times its column count; the
    # result is the intersection divided by the union (sum of sizes minus intersection).
    # Inputs: four 1d indicator vectors (boolean or 0/1) for rows/cols of each
    # bicluster.
    # Outputs: float in [0, 1] (numpy scalar).
    # Exceptions: none raised; if both biclusters are empty the division is 0/0 and
    # yields nan with a numpy RuntimeWarning.
    """Jaccard coefficient on the elements of the two biclusters."""
    intersection = (a_rows * b_rows).sum() * (a_cols * b_cols).sum()

    a_size = a_rows.sum() * a_cols.sum()
    b_size = b_rows.sum() * b_cols.sum()

    return intersection / (a_size + b_size - intersection)


def _pairwise_similarity(a, b, similarity):
    # [FUNCTION DOC]
    # HELPER: Builds the (n_a, n_b) matrix where entry i, j is similarity applied to
    # bicluster i of a and bicluster j of b.
    # Inputs: a, b - (rows, cols) tuples; similarity - callable taking (a_rows, a_cols,
    # b_rows, b_cols) 1d vectors and returning a scalar.
    # Outputs: float ndarray of shape (n_a, n_b).
    # Exceptions: propagated from _check_rows_and_columns.
    # KNOWN ISSUE: the docstring says each entry is the Jaccard coefficient, but any
    # supplied similarity callable is used.
    """Computes pairwise similarity matrix.

    result[i, j] is the Jaccard coefficient of a's bicluster i and b's
    bicluster j.

    """
    a_rows, a_cols, b_rows, b_cols = _check_rows_and_columns(a, b)
    n_a = a_rows.shape[0]
    n_b = b_rows.shape[0]
    result = np.array(
        [
            [similarity(a_rows[i], a_cols[i], b_rows[j], b_cols[j]) for j in range(n_b)]
            for i in range(n_a)
        ]
    )
    return result


@validate_params(
    {
        "a": [tuple],
        "b": [tuple],
        "similarity": [callable, StrOptions({"jaccard"})],
    },
    prefer_skip_nested_validation=True,
)
def consensus_score(a, b, *, similarity="jaccard"):
    # [FUNCTION DOC]
    # Computes the pairwise similarity matrix (Jaccard when similarity is "jaccard"),
    # solves a linear sum assignment on one minus the matrix to maximize total
    # similarity, then divides the summed matched similarities by the larger of the two
    # bicluster counts, so unmatched biclusters count as zero.
    # Inputs/outputs: see docstring; returns a Python float.
    # Exceptions: InvalidParameterError from validate_params for non-tuple a/b or an
    # invalid similarity; ValueError from input validation.
    # Side effects: none.
    """The similarity of two sets of biclusters.

    Similarity between individual biclusters is computed. Then the best
    matching between sets is found by solving a linear sum assignment problem,
    using a modified Jonker-Volgenant algorithm.
    The final score is the sum of similarities divided by the size of
    the larger set.

    Read more in the :ref:`User Guide <biclustering>`.

    Parameters
    ----------
    a : tuple (rows, columns)
        Tuple of row and column indicators for a set of biclusters.

    b : tuple (rows, columns)
        Another set of biclusters like ``a``.

    similarity : 'jaccard' or callable, default='jaccard'
        May be the string "jaccard" to use the Jaccard coefficient, or
        any function that takes four arguments, each of which is a 1d
        indicator vector: (a_rows, a_columns, b_rows, b_columns).

    Returns
    -------
    consensus_score : float
       Consensus score, a non-negative value, sum of similarities
       divided by size of larger set.

    See Also
    --------
    scipy.optimize.linear_sum_assignment : Solve the linear sum assignment problem.

    References
    ----------
    * Hochreiter, Bodenhofer, et. al., 2010. `FABIA: factor analysis
      for bicluster acquisition
      <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2881408/>`__.

    Examples
    --------
    >>> from sklearn.metrics import consensus_score
    >>> a = ([[True, False], [False, True]], [[False, True], [True, False]])
    >>> b = ([[False, True], [True, False]], [[True, False], [False, True]])
    >>> consensus_score(a, b, similarity='jaccard')
    1.0
    """
    if similarity == "jaccard":
        similarity = _jaccard
    matrix = _pairwise_similarity(a, b, similarity)
    row_indices, col_indices = linear_sum_assignment(1.0 - matrix)
    n_a = len(a[0])
    n_b = len(b[0])
    return float(matrix[row_indices, col_indices].sum() / max(n_a, n_b))
