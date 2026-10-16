# [MODULE DOC]
# Tests for the _BaseImputer missing-indicator protocol (_fit_indicator,
# _transform_indicator, _concatenate_indicator) using minimal imputer subclasses that
# misuse it, plus a test of the _assign_where helper from _iterative.
import numpy as np
import pytest

from sklearn.impute._base import _BaseImputer
from sklearn.impute._iterative import _assign_where
from sklearn.utils._mask import _get_mask
from sklearn.utils._testing import _convert_container, assert_allclose


@pytest.fixture
def data():
    # [FUNCTION DOC]
    # HELPER: Fixture giving a (10, 2) standard-normal array whose even rows are all
    # NaN.
    # Outputs: ndarray. Side effects: uses the unseeded global numpy RNG.
    X = np.random.randn(10, 2)
    X[::2] = np.nan
    return X


class NoFitIndicatorImputer(_BaseImputer):
    # [CLASS DOC]
    # HELPER: Imputer that never calls _fit_indicator, used to trigger the "call
    # _fit_indicator first" error.
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: No-op fit. Outputs: self.
        return self

    def transform(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Calls _transform_indicator without prior fit and concatenates.
        # Outputs: the concatenated array (expected to raise ValueError when
        # add_indicator is True).
        return self._concatenate_indicator(X, self._transform_indicator(X))


class NoTransformIndicatorImputer(_BaseImputer):
    # [CLASS DOC]
    # HELPER: Imputer that fits the indicator but never transforms it, used to trigger
    # the "call _fit_indicator and _transform_indicator" error.
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Fits the indicator on the NaN mask of X. Outputs: self.
        mask = _get_mask(X, value_to_mask=np.nan)
        super()._fit_indicator(mask)
        return self

    def transform(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Concatenates X with a None indicator (expected to raise ValueError).
        return self._concatenate_indicator(X, None)


class NoPrecomputedMaskFit(_BaseImputer):
    # [CLASS DOC]
    # HELPER: Imputer that passes raw data instead of a boolean mask to _fit_indicator.
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Calls _fit_indicator with X itself (expected to raise ValueError).
        # Outputs: self.
        self._fit_indicator(X)
        return self

    def transform(self, X):
        # [FUNCTION DOC]
        # HELPER: Transforms indicator from X and concatenates.
        return self._concatenate_indicator(X, self._transform_indicator(X))


class NoPrecomputedMaskTransform(_BaseImputer):
    # [CLASS DOC]
    # HELPER: Imputer that fits the indicator correctly but passes raw data to
    # _transform_indicator.
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # HELPER: Fits the indicator on the NaN mask of X. Outputs: self.
        mask = _get_mask(X, value_to_mask=np.nan)
        self._fit_indicator(mask)
        return self

    def transform(self, X):
        # [FUNCTION DOC]
        # HELPER: Calls _transform_indicator with X itself (expected to raise
        # ValueError).
        return self._concatenate_indicator(X, self._transform_indicator(X))


def test_base_imputer_not_fit(data):
    # [FUNCTION DOC]
    # Verifies that transform/fit_transform raise ValueError asking to call
    # _fit_indicator when the indicator was never fitted.
    imputer = NoFitIndicatorImputer(add_indicator=True)
    err_msg = "Make sure to call _fit_indicator before _transform_indicator"
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit(data).transform(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(data)


def test_base_imputer_not_transform(data):
    # [FUNCTION DOC]
    # Verifies that concatenating without transforming the fitted indicator raises a
    # ValueError, both via fit+transform and fit_transform.
    imputer = NoTransformIndicatorImputer(add_indicator=True)
    err_msg = (
        "Call _fit_indicator and _transform_indicator in the imputer implementation"
    )
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit(data).transform(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(data)


def test_base_no_precomputed_mask_fit(data):
    # [FUNCTION DOC]
    # Verifies that passing non-mask data to _fit_indicator raises the "precomputed is
    # True but the input data is not a mask" ValueError in fit and fit_transform.
    imputer = NoPrecomputedMaskFit(add_indicator=True)
    err_msg = "precomputed is True but the input data is not a mask"
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(data)


def test_base_no_precomputed_mask_transform(data):
    # [FUNCTION DOC]
    # Verifies that passing non-mask data to _transform_indicator raises the same
    # precomputed-mask ValueError in transform and fit_transform.
    imputer = NoPrecomputedMaskTransform(add_indicator=True)
    err_msg = "precomputed is True but the input data is not a mask"
    imputer.fit(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.transform(data)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(data)


@pytest.mark.parametrize("X1_type", ["array", "pandas"])
def test_assign_where(X1_type):
    # [FUNCTION DOC]
    # Verifies _assign_where copies X2 into X1 at masked positions for both a numpy
    # array and a pandas DataFrame destination, by comparing the masked entries after
    # the call.
    """Check the behaviour of the private helpers `_assign_where`."""
    rng = np.random.RandomState(0)

    n_samples, n_features = 10, 5
    X1 = _convert_container(rng.randn(n_samples, n_features), constructor_name=X1_type)
    X2 = rng.randn(n_samples, n_features)
    mask = rng.randint(0, 2, size=(n_samples, n_features)).astype(bool)

    _assign_where(X1, X2, mask)

    if X1_type == "pandas":
        X1 = X1.to_numpy()
    assert_allclose(X1[mask], X2[mask])
