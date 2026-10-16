# [MODULE DOC]
# Common tests run against all imputers (IterativeImputer, KNNImputer, SimpleImputer):
# missing indicator behaviour, pandas nullable integer support, feature names out and
# keep_empty_features handling.
import numpy as np
import pytest

from sklearn.base import clone
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.utils._testing import (
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_array_equal,
)
from sklearn.utils.fixes import CSR_CONTAINERS


def imputers():
    # [FUNCTION DOC]
    # HELPER: Returns fresh default-ish instances of the three dense imputers
    # (IterativeImputer with tol=0.1). Outputs: list of estimators.
    return [IterativeImputer(tol=0.1), KNNImputer(), SimpleImputer()]


def sparse_imputers():
    # [FUNCTION DOC]
    # HELPER: Returns the imputers supporting sparse input (only SimpleImputer).
    return [SimpleImputer()]


# ConvergenceWarning will be raised by the IterativeImputer
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("imputer", imputers(), ids=lambda x: x.__class__.__name__)
def test_imputation_missing_value_in_test_array(imputer):
    # [FUNCTION DOC]
    # Non-regression for #13968: fitting on data without missing values and transforming
    # data with a NaN, with add_indicator=True, must not raise.
    # KNOWN ISSUE: the comment says the result should be finite, but nothing is
    # asserted.
    # [Non Regression Test for issue #13968] Missing value in test set should
    # not throw an error and return a finite dataset
    train = [[1], [2]]
    test = [[3], [np.nan]]
    imputer = clone(imputer)
    imputer.set_params(add_indicator=True)
    imputer.fit(train).transform(test)


# ConvergenceWarning will be raised by the IterativeImputer
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("marker", [np.nan, -1, 0])
@pytest.mark.parametrize("imputer", imputers(), ids=lambda x: x.__class__.__name__)
def test_imputers_add_indicator(marker, imputer):
    # [FUNCTION DOC]
    # For markers NaN, -1 and 0, verifies that with add_indicator the last four columns
    # equal the expected missing indicator, indicator_.features_ lists columns 0..3, and
    # the remaining columns equal the output with add_indicator=False.
    X = np.array(
        [
            [marker, 1, 5, marker, 1],
            [2, marker, 1, marker, 2],
            [6, 3, marker, marker, 3],
            [1, 2, 9, marker, 4],
        ]
    )
    X_true_indicator = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    imputer = clone(imputer)
    imputer.set_params(missing_values=marker, add_indicator=True)

    X_trans = imputer.fit_transform(X)
    assert_allclose(X_trans[:, -4:], X_true_indicator)
    assert_array_equal(imputer.indicator_.features_, np.array([0, 1, 2, 3]))

    imputer.set_params(add_indicator=False)
    X_trans_no_indicator = imputer.fit_transform(X)
    assert_allclose(X_trans[:, :-4], X_trans_no_indicator)


# ConvergenceWarning will be raised by the IterativeImputer
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("marker", [np.nan, -1])
@pytest.mark.parametrize(
    "imputer", sparse_imputers(), ids=lambda x: x.__class__.__name__
)
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_imputers_add_indicator_sparse(imputer, marker, csr_container):
    # [FUNCTION DOC]
    # Same as test_imputers_add_indicator but for CSR inputs (each CSR container) and
    # sparse capable imputers, comparing with assert_allclose_dense_sparse.
    imputer = clone(imputer)  # Avoid side effects from shared instances.
    X = csr_container(
        [
            [marker, 1, 5, marker, 1],
            [2, marker, 1, marker, 2],
            [6, 3, marker, marker, 3],
            [1, 2, 9, marker, 4],
        ]
    )
    X_true_indicator = csr_container(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    imputer.set_params(missing_values=marker, add_indicator=True)

    X_trans = imputer.fit_transform(X)
    assert_allclose_dense_sparse(X_trans[:, -4:], X_true_indicator)
    assert_array_equal(imputer.indicator_.features_, np.array([0, 1, 2, 3]))

    imputer.set_params(add_indicator=False)
    X_trans_no_indicator = imputer.fit_transform(X)
    assert_allclose_dense_sparse(X_trans[:, :-4], X_trans_no_indicator)


# ConvergenceWarning will be raised by the IterativeImputer
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("imputer", imputers(), ids=lambda x: x.__class__.__name__)
@pytest.mark.parametrize("add_indicator", [True, False])
def test_imputers_pandas_na_integer_array_support(imputer, add_indicator):
    # [FUNCTION DOC]
    # Verifies that fitting on a DataFrame of nullable Int16 columns with pd.NA gives
    # the same result as fitting on the equivalent float array with NaN.
    # KNOWN ISSUE: the parametrized imputer instance is mutated via set_params without
    # clone, so state is shared across parametrizations.
    # Test pandas IntegerArray with pd.NA
    pd = pytest.importorskip("pandas")
    marker = np.nan
    imputer = imputer.set_params(add_indicator=add_indicator, missing_values=marker)

    X = np.array(
        [
            [marker, 1, 5, marker, 1],
            [2, marker, 1, marker, 2],
            [6, 3, marker, marker, 3],
            [1, 2, 9, marker, 4],
        ]
    )
    # fit on numpy array
    X_trans_expected = imputer.fit_transform(X)

    # Creates dataframe with IntegerArrays with pd.NA
    X_df = pd.DataFrame(X, dtype="Int16", columns=["a", "b", "c", "d", "e"])

    # fit on pandas dataframe with IntegerArrays
    X_trans = imputer.fit_transform(X_df)

    assert_allclose(X_trans_expected, X_trans)


@pytest.mark.parametrize("imputer", imputers(), ids=lambda x: x.__class__.__name__)
@pytest.mark.parametrize("add_indicator", [True, False])
def test_imputers_feature_names_out_pandas(imputer, add_indicator):
    # [FUNCTION DOC]
    # Verifies get_feature_names_out on a DataFrame: the all-missing column "e" is
    # dropped and, with add_indicator, indicator names are appended for columns with
    # missing values.
    # KNOWN ISSUE: the parametrized imputer instance is not cloned before set_params.
    """Check feature names out for imputers."""
    pd = pytest.importorskip("pandas")
    marker = np.nan
    imputer = imputer.set_params(add_indicator=add_indicator, missing_values=marker)

    X = np.array(
        [
            [marker, 1, 5, 3, marker, 1],
            [2, marker, 1, 4, marker, 2],
            [6, 3, 7, marker, marker, 3],
            [1, 2, 9, 8, marker, 4],
        ]
    )
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d", "e", "f"])
    imputer.fit(X_df)

    names = imputer.get_feature_names_out()

    if add_indicator:
        expected_names = [
            "a",
            "b",
            "c",
            "d",
            "f",
            "missingindicator_a",
            "missingindicator_b",
            "missingindicator_d",
            "missingindicator_e",
        ]
        assert_array_equal(expected_names, names)
    else:
        expected_names = ["a", "b", "c", "d", "f"]
        assert_array_equal(expected_names, names)


@pytest.mark.parametrize("keep_empty_features", [True, False])
@pytest.mark.parametrize("imputer", imputers(), ids=lambda x: x.__class__.__name__)
def test_keep_empty_features(imputer, keep_empty_features):
    # [FUNCTION DOC]
    # Verifies that an all-NaN column is kept (shape unchanged) when keep_empty_features
    # is True and dropped otherwise, for fit_transform and then transform.
    """Check that the imputer keeps features with only missing values."""
    X = np.array([[np.nan, 1], [np.nan, 2], [np.nan, 3]])
    imputer = clone(imputer)
    imputer = imputer.set_params(
        add_indicator=False, keep_empty_features=keep_empty_features
    )

    for method in ["fit_transform", "transform"]:
        X_imputed = getattr(imputer, method)(X)
        if keep_empty_features:
            assert X_imputed.shape == X.shape
        else:
            assert X_imputed.shape == (X.shape[0], X.shape[1] - 1)


@pytest.mark.parametrize("imputer", imputers(), ids=lambda x: x.__class__.__name__)
@pytest.mark.parametrize("missing_value_test", [np.nan, 1])
def test_imputation_adds_missing_indicator_if_add_indicator_is_true(
    imputer, missing_value_test
):
    # [FUNCTION DOC]
    # Non-regression for gh-26590: with add_indicator, transform always outputs the
    # indicator column for features missing in training, even when the test data has no
    # missing value; the imputed part equals the add_indicator=False output and the
    # indicator is [1, 0] for a NaN test value and [0, 0] otherwise.
    """Check that missing indicator always exists when add_indicator=True.

    Non-regression test for gh-26590.
    """
    X_train = np.array([[0, np.nan], [1, 2]])

    # Test data where missing_value_test variable can be set to np.nan or 1.
    X_test = np.array([[0, missing_value_test], [1, 2]])

    imputer = clone(imputer)
    imputer.set_params(add_indicator=True)
    imputer.fit(X_train)

    X_test_imputed_with_indicator = imputer.transform(X_test)
    assert X_test_imputed_with_indicator.shape == (2, 3)

    imputer.set_params(add_indicator=False)
    imputer.fit(X_train)
    X_test_imputed_without_indicator = imputer.transform(X_test)
    assert X_test_imputed_without_indicator.shape == (2, 2)

    assert_allclose(
        X_test_imputed_with_indicator[:, :-1], X_test_imputed_without_indicator
    )
    if np.isnan(missing_value_test):
        expected_missing_indicator = [1, 0]
    else:
        expected_missing_indicator = [0, 0]

    assert_allclose(X_test_imputed_with_indicator[:, -1], expected_missing_indicator)
