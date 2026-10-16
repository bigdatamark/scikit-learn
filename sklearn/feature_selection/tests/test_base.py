# [MODULE DOC]
# Tests for SelectorMixin using a minimal StepSelector that keeps every step-th feature,
#   checking transform, inverse_transform and get_support on dense, sparse and pandas
#   inputs. Module-level fixtures: a 2x10 integer X, its expected transformed and
#   inverse-transformed versions, the support mask and feature names.
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.validation import validate_data


class StepSelector(SelectorMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: Test selector keeping features 0, step, 2*step, ...; selects nothing when
    #   step is below 1.
    """Retain every `step` features (beginning with 0).

    If `step < 1`, then no features are selected.
    """

    def __init__(self, step=2):
        # [FUNCTION DOC]
        # Stores step (default 2).
        self.step = step

    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Validates X (CSC allowed) to set n_features_in_ and returns self; y ignored.
        X = validate_data(self, X, accept_sparse="csc")
        return self

    def _get_support_mask(self):
        # [FUNCTION DOC]
        # Boolean mask with every step-th position True starting at 0, all False if step
        #   < 1.
        mask = np.zeros(self.n_features_in_, dtype=bool)
        if self.step >= 1:
            mask[:: self.step] = True
        return mask


support = [True, False] * 5
support_inds = [0, 2, 4, 6, 8]
X = np.arange(20).reshape(2, 10)
Xt = np.arange(0, 20, 2).reshape(2, 5)
Xinv = X.copy()
Xinv[:, 1::2] = 0
y = [0, 1]
feature_names = list("ABCDEFGHIJ")
feature_names_t = feature_names[::2]
feature_names_inv = np.array(feature_names)
feature_names_inv[1::2] = ""


def test_transform_dense():
    # [FUNCTION DOC]
    # Verifies dense transform and fit_transform give the expected columns, preserve
    #   int32 / float32 dtypes, work on a list of strings, and raise ValueError for a
    #   wrong number of features.
    sel = StepSelector()
    Xt_actual = sel.fit(X, y).transform(X)
    Xt_actual2 = StepSelector().fit_transform(X, y)
    assert_array_equal(Xt, Xt_actual)
    assert_array_equal(Xt, Xt_actual2)

    # Check dtype matches
    assert np.int32 == sel.transform(X.astype(np.int32)).dtype
    assert np.float32 == sel.transform(X.astype(np.float32)).dtype

    # Check 1d list and other dtype:
    names_t_actual = sel.transform([feature_names])
    assert_array_equal(feature_names_t, names_t_actual.ravel())

    # Check wrong shape raises error
    with pytest.raises(ValueError):
        sel.transform(np.array([[1], [2]]))


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_transform_sparse(csc_container):
    # [FUNCTION DOC]
    # Same checks as the dense test for each CSC container: values, dtype preservation,
    #   and ValueError on a wrongly shaped dense input.
    X_sp = csc_container(X)
    sel = StepSelector()
    Xt_actual = sel.fit(X_sp).transform(X_sp)
    Xt_actual2 = sel.fit_transform(X_sp)
    assert_array_equal(Xt, Xt_actual.toarray())
    assert_array_equal(Xt, Xt_actual2.toarray())

    # Check dtype matches
    assert np.int32 == sel.transform(X_sp.astype(np.int32)).dtype
    assert np.float32 == sel.transform(X_sp.astype(np.float32)).dtype

    # Check wrong shape raises error
    with pytest.raises(ValueError):
        sel.transform(np.array([[1], [2]]))


def test_inverse_transform_dense():
    # [FUNCTION DOC]
    # Verifies inverse_transform re-inserts zero columns, preserves dtypes, fills string
    #   input with empty strings, and raises ValueError on a wrong column count.
    sel = StepSelector()
    Xinv_actual = sel.fit(X, y).inverse_transform(Xt)
    assert_array_equal(Xinv, Xinv_actual)

    # Check dtype matches
    assert np.int32 == sel.inverse_transform(Xt.astype(np.int32)).dtype
    assert np.float32 == sel.inverse_transform(Xt.astype(np.float32)).dtype

    # Check 1d list and other dtype:
    names_inv_actual = sel.inverse_transform([feature_names_t])
    assert_array_equal(feature_names_inv, names_inv_actual.ravel())

    # Check wrong shape raises error
    with pytest.raises(ValueError):
        sel.inverse_transform(np.array([[1], [2]]))


@pytest.mark.parametrize("csc_container", CSC_CONTAINERS)
def test_inverse_transform_sparse(csc_container):
    # [FUNCTION DOC]
    # Verifies sparse inverse_transform values and dtypes for each CSC container and the
    #   ValueError on a wrong dense shape.
    X_sp = csc_container(X)
    Xt_sp = csc_container(Xt)
    sel = StepSelector()
    Xinv_actual = sel.fit(X_sp).inverse_transform(Xt_sp)
    assert_array_equal(Xinv, Xinv_actual.toarray())

    # Check dtype matches
    assert np.int32 == sel.inverse_transform(Xt_sp.astype(np.int32)).dtype
    assert np.float32 == sel.inverse_transform(Xt_sp.astype(np.float32)).dtype

    # Check wrong shape raises error
    with pytest.raises(ValueError):
        sel.inverse_transform(np.array([[1], [2]]))


def test_get_support():
    # [FUNCTION DOC]
    # Checks get_support as a boolean mask and as indices.
    sel = StepSelector()
    sel.fit(X, y)
    assert_array_equal(support, sel.get_support())
    assert_array_equal(support_inds, sel.get_support(indices=True))


def test_output_dataframe():
    # [FUNCTION DOC]
    # With pandas output, checks that selected columns keep their original dtypes
    #   (including categoricals), and that selecting nothing warns "No features were
    #   selected" and returns an empty-column DataFrame with the original index. Skipped
    #   without pandas.
    """Check output dtypes for dataframes is consistent with the input dtypes."""
    pd = pytest.importorskip("pandas")

    X = pd.DataFrame(
        {
            "a": pd.Series([1.0, 2.4, 4.5], dtype=np.float32),
            "b": pd.Series(["a", "b", "a"], dtype="category"),
            "c": pd.Series(["j", "b", "b"], dtype="category"),
            "d": pd.Series([3.0, 2.4, 1.2], dtype=np.float64),
        }
    )

    for step in [2, 3]:
        sel = StepSelector(step=step).set_output(transform="pandas")
        sel.fit(X)

        output = sel.transform(X)
        for name, dtype in output.dtypes.items():
            assert dtype == X.dtypes[name]

    # step=0 will select nothing
    sel0 = StepSelector(step=0).set_output(transform="pandas")
    sel0.fit(X, y)

    msg = "No features were selected"
    with pytest.warns(UserWarning, match=msg):
        output0 = sel0.transform(X)

    assert_array_equal(output0.index, X.index)
    assert output0.shape == (X.shape[0], 0)
