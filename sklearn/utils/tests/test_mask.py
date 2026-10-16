# [MODULE DOC]
# Tests for sklearn.utils._mask.safe_mask, which converts boolean masks into a form
# usable for indexing both dense arrays and sparse matrices.
import pytest

from sklearn.utils._mask import safe_mask
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import check_random_state


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_safe_mask(csr_container):
    # [FUNCTION DOC]
    # Verifies that safe_mask yields a mask selecting the 3 True rows of a 5x4 random
    # array, for both a dense ndarray and each CSR container (parametrized over
    # CSR_CONTAINERS); for the sparse case the already converted mask is reused.
    random_state = check_random_state(0)
    X = random_state.rand(5, 4)
    X_csr = csr_container(X)
    mask = [False, False, True, True, True]

    mask = safe_mask(X, mask)
    assert X[mask].shape[0] == 3

    mask = safe_mask(X_csr, mask)
    assert X_csr[mask].shape[0] == 3
