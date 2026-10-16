# [MODULE DOC]
# Tests for the bicluster consensus score and its private Jaccard helper.
"""Testing for bicluster metrics module"""

import numpy as np

from sklearn.metrics import consensus_score
from sklearn.metrics.cluster._bicluster import _jaccard
from sklearn.utils._testing import assert_almost_equal


def test_jaccard():
    # [FUNCTION DOC]
    # Checks _jaccard on hand-built row/column indicator vectors: identical biclusters
    # give 1, a 2x2 inside a 4x4 gives 0.25, partial overlap gives 1/7, disjoint gives
    # 0.
    a1 = np.array([True, True, False, False])
    a2 = np.array([True, True, True, True])
    a3 = np.array([False, True, True, False])
    a4 = np.array([False, False, True, True])

    assert _jaccard(a1, a1, a1, a1) == 1
    assert _jaccard(a1, a1, a2, a2) == 0.25
    assert _jaccard(a1, a1, a3, a3) == 1.0 / 7
    assert _jaccard(a1, a1, a4, a4) == 0


def test_consensus_score():
    # [FUNCTION DOC]
    # Checks consensus_score is 1 when the two sets contain the same biclusters (in any
    # order) and 0 when every matched pair has disjoint cells.
    a = [[True, True, False, False], [False, False, True, True]]
    b = a[::-1]

    assert consensus_score((a, a), (a, a)) == 1
    assert consensus_score((a, a), (b, b)) == 1
    assert consensus_score((a, b), (a, b)) == 1
    assert consensus_score((a, b), (b, a)) == 1

    assert consensus_score((a, a), (b, a)) == 0
    assert consensus_score((a, a), (a, b)) == 0
    assert consensus_score((b, b), (a, b)) == 0
    assert consensus_score((b, b), (b, a)) == 0


def test_consensus_score_issue2445():
    # [FUNCTION DOC]
    # Regression test for issue 2445: with 3 biclusters in a and 2 of them in b, the
    # score must be 2/3, i.e. normalized by the larger set size.
    """Different number of biclusters in A and B"""
    a_rows = np.array(
        [
            [True, True, False, False],
            [False, False, True, True],
            [False, False, False, True],
        ]
    )
    a_cols = np.array(
        [
            [True, True, False, False],
            [False, False, True, True],
            [False, False, False, True],
        ]
    )
    idx = [0, 2]
    s = consensus_score((a_rows, a_cols), (a_rows[idx], a_cols[idx]))
    # B contains 2 of the 3 biclusters in A, so score should be 2/3
    assert_almost_equal(s, 2.0 / 3.0)
