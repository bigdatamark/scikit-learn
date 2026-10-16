# [MODULE DOC]
# Tests for sklearn.compose.TransformedTargetRegressor: parameter validation,
# func/inverse_func and transformer consistency with 1D, 2D and 3D targets,
# input types passed to the transformer and regressor, single fit of the
# transformer, routing of fit/predict kwargs, global output config and metadata
# routing. Module-level friedman holds make_friedman1(random_state=0) data
# (100 samples, 10 features) shared by most tests.
import warnings

import numpy as np
import pytest

from sklearn import config_context, datasets
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils._testing import assert_allclose

friedman = datasets.make_friedman1(random_state=0)


def test_transform_target_regressor_error():
    # [FUNCTION DOC]
    # Verifies fit errors: ValueError when both transformer and func/inverse_func are
    # set; TypeError when sample_weight is passed to a regressor whose fit lacks it
    # (OrthogonalMatchingPursuit); ValueError when only one of func or inverse_func is
    # given.
    X, y = friedman
    # provide a transformer and functions at the same time
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(),
        transformer=StandardScaler(),
        func=np.exp,
        inverse_func=np.log,
    )
    with pytest.raises(
        ValueError,
        match="'transformer' and functions 'func'/'inverse_func' cannot both be set.",
    ):
        regr.fit(X, y)
    # fit with sample_weight with a regressor which does not support it
    sample_weight = np.ones((y.shape[0],))
    regr = TransformedTargetRegressor(
        regressor=OrthogonalMatchingPursuit(), transformer=StandardScaler()
    )
    with pytest.raises(
        TypeError,
        match=r"fit\(\) got an unexpected keyword argument 'sample_weight'",
    ):
        regr.fit(X, y, sample_weight=sample_weight)

    # one of (func, inverse_func) is given but the other one is not
    regr = TransformedTargetRegressor(func=np.exp)
    with pytest.raises(
        ValueError,
        match="When 'func' is provided, 'inverse_func' must also be provided",
    ):
        regr.fit(X, y)

    regr = TransformedTargetRegressor(inverse_func=np.log)
    with pytest.raises(
        ValueError,
        match="When 'inverse_func' is provided, 'func' must also be provided",
    ):
        regr.fit(X, y)


def test_transform_target_regressor_invertible():
    # [FUNCTION DOC]
    # Verifies check_inverse=True warns (UserWarning) when func=sqrt and
    # inverse_func=log are not inverses, and that after set_params(check_inverse=False)
    # fitting raises no UserWarning (warnings turned into errors).
    X, y = friedman
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=np.sqrt,
        inverse_func=np.log,
        check_inverse=True,
    )
    with pytest.warns(
        UserWarning,
        match=(r"The provided functions.* are not strictly inverse of each other"),
    ):
        regr.fit(X, y)
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), func=np.sqrt, inverse_func=np.log
    )
    regr.set_params(check_inverse=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        regr.fit(X, y)


def _check_standard_scaled(y, y_pred):
    # [FUNCTION DOC]
    # HELPER: asserts y_pred equals y standardized column-wise (mean 0, population
    # std 1). Inputs: y, y_pred - arrays of matching shape. Raises AssertionError.
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    assert_allclose((y - y_mean) / y_std, y_pred)


def _check_shifted_by_one(y, y_pred):
    # [FUNCTION DOC]
    # HELPER: asserts y_pred equals y + 1 elementwise. Raises AssertionError.
    assert_allclose(y + 1, y_pred)


def test_transform_target_regressor_functions():
    # [FUNCTION DOC]
    # With func=log/inverse_func=exp and a 1D target, checks transformer_ applies
    # log and inverts it, prediction shape matches y, predictions equal exp of the
    # inner regressor's predictions, and inner coefficients match a LinearRegression
    # fitted on log(y).
    X, y = friedman
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), func=np.log, inverse_func=np.exp
    )
    y_pred = regr.fit(X, y).predict(X)
    # check the transformer output
    y_tran = regr.transformer_.transform(y.reshape(-1, 1)).squeeze()
    assert_allclose(np.log(y), y_tran)
    assert_allclose(
        y, regr.transformer_.inverse_transform(y_tran.reshape(-1, 1)).squeeze()
    )
    assert y.shape == y_pred.shape
    assert_allclose(y_pred, regr.inverse_func(regr.regressor_.predict(X)))
    # check the regressor output
    lr = LinearRegression().fit(X, regr.func(y))
    assert_allclose(regr.regressor_.coef_.ravel(), lr.coef_.ravel())


def test_transform_target_regressor_functions_multioutput():
    # [FUNCTION DOC]
    # Same as the 1D functions test but with a two-column target (y and y^2 + 1).
    X = friedman[0]
    y = np.vstack((friedman[1], friedman[1] ** 2 + 1)).T
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), func=np.log, inverse_func=np.exp
    )
    y_pred = regr.fit(X, y).predict(X)
    # check the transformer output
    y_tran = regr.transformer_.transform(y)
    assert_allclose(np.log(y), y_tran)
    assert_allclose(y, regr.transformer_.inverse_transform(y_tran))
    assert y.shape == y_pred.shape
    assert_allclose(y_pred, regr.inverse_func(regr.regressor_.predict(X)))
    # check the regressor output
    lr = LinearRegression().fit(X, regr.func(y))
    assert_allclose(regr.regressor_.coef_.ravel(), lr.coef_.ravel())


@pytest.mark.parametrize(
    "X,y", [friedman, (friedman[0], np.vstack((friedman[1], friedman[1] ** 2 + 1)).T)]
)
def test_transform_target_regressor_1d_transformer(X, y):
    # [FUNCTION DOC]
    # Parametrized over 1D and 2D targets; uses a FunctionTransformer (x+1, x-1)
    # without validation. Checks shape preservation, forward shift by one, inverse
    # recovery, and that predictions and coefficients match a manual LinearRegression
    # fitted on the transformed target and inverse transformed.
    # All transformer in scikit-learn expect 2D data. FunctionTransformer with
    # validate=False lift this constraint without checking that the input is a
    # 2D vector. We check the consistency of the data shape using a 1D and 2D y
    # array.
    transformer = FunctionTransformer(
        func=lambda x: x + 1, inverse_func=lambda x: x - 1
    )
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), transformer=transformer
    )
    y_pred = regr.fit(X, y).predict(X)
    assert y.shape == y_pred.shape
    # consistency forward transform
    y_tran = regr.transformer_.transform(y)
    _check_shifted_by_one(y, y_tran)
    assert y.shape == y_pred.shape
    # consistency inverse transform
    assert_allclose(y, regr.transformer_.inverse_transform(y_tran).squeeze())
    # consistency of the regressor
    lr = LinearRegression()
    transformer2 = clone(transformer)
    lr.fit(X, transformer2.fit_transform(y))
    y_lr_pred = lr.predict(X)
    assert_allclose(y_pred, transformer2.inverse_transform(y_lr_pred))
    assert_allclose(regr.regressor_.coef_, lr.coef_)


@pytest.mark.parametrize(
    "X,y", [friedman, (friedman[0], np.vstack((friedman[1], friedman[1] ** 2 + 1)).T)]
)
def test_transform_target_regressor_2d_transformer(X, y):
    # [FUNCTION DOC]
    # Parametrized over 1D and 2D targets with StandardScaler (2D-only transformer).
    # Checks shapes, standardization of the transformed target, inverse recovery, and
    # equivalence with a manual scale-fit-predict-inverse pipeline (reshaping 1D y to
    # a column and squeezing back).
    # Check consistency with transformer accepting only 2D array and a 1D/2D y
    # array.
    transformer = StandardScaler()
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), transformer=transformer
    )
    y_pred = regr.fit(X, y).predict(X)
    assert y.shape == y_pred.shape
    # consistency forward transform
    if y.ndim == 1:  # create a 2D array and squeeze results
        y_tran = regr.transformer_.transform(y.reshape(-1, 1))
    else:
        y_tran = regr.transformer_.transform(y)
    _check_standard_scaled(y, y_tran.squeeze())
    assert y.shape == y_pred.shape
    # consistency inverse transform
    assert_allclose(y, regr.transformer_.inverse_transform(y_tran).squeeze())
    # consistency of the regressor
    lr = LinearRegression()
    transformer2 = clone(transformer)
    if y.ndim == 1:  # create a 2D array and squeeze results
        lr.fit(X, transformer2.fit_transform(y.reshape(-1, 1)).squeeze())
        y_lr_pred = lr.predict(X).reshape(-1, 1)
        y_pred2 = transformer2.inverse_transform(y_lr_pred).squeeze()
    else:
        lr.fit(X, transformer2.fit_transform(y))
        y_lr_pred = lr.predict(X)
        y_pred2 = transformer2.inverse_transform(y_lr_pred)

    assert_allclose(y_pred, y_pred2)
    assert_allclose(regr.regressor_.coef_, lr.coef_)


def test_transform_target_regressor_2d_transformer_multioutput():
    # [FUNCTION DOC]
    # StandardScaler with a two-column target: checks shapes, standardization,
    # inverse recovery, and equivalence with manual fitting on the scaled target.
    # KNOWN ISSUE: fully covered by the 2D case of the parametrized 2d_transformer
    # test; redundant.
    # Check consistency with transformer accepting only 2D array and a 2D y
    # array.
    X = friedman[0]
    y = np.vstack((friedman[1], friedman[1] ** 2 + 1)).T
    transformer = StandardScaler()
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), transformer=transformer
    )
    y_pred = regr.fit(X, y).predict(X)
    assert y.shape == y_pred.shape
    # consistency forward transform
    y_tran = regr.transformer_.transform(y)
    _check_standard_scaled(y, y_tran)
    assert y.shape == y_pred.shape
    # consistency inverse transform
    assert_allclose(y, regr.transformer_.inverse_transform(y_tran).squeeze())
    # consistency of the regressor
    lr = LinearRegression()
    transformer2 = clone(transformer)
    lr.fit(X, transformer2.fit_transform(y))
    y_lr_pred = lr.predict(X)
    assert_allclose(y_pred, transformer2.inverse_transform(y_lr_pred))
    assert_allclose(regr.regressor_.coef_, lr.coef_)


def test_transform_target_regressor_3d_target():
    # [FUNCTION DOC]
    # Non-regression (gh-18866): 3D target (n, 3, 2) with a FunctionTransformer
    # that flattens to 2D and unflattens back; checks prediction shape equals y shape.
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/18866
    # Check with a 3D target with a transformer that reshapes the target
    X = friedman[0]
    y = np.tile(friedman[1].reshape(-1, 1, 1), [1, 3, 2])

    def flatten_data(data):
        # [FUNCTION DOC]
        # HELPER: reshapes data to (n_samples, -1). Outputs: 2D array view.
        return data.reshape(data.shape[0], -1)

    def unflatten_data(data):
        # [FUNCTION DOC]
        # HELPER: reshapes data to (n_samples, -1, 2). Outputs: 3D array view.
        return data.reshape(data.shape[0], -1, 2)

    transformer = FunctionTransformer(func=flatten_data, inverse_func=unflatten_data)
    regr = TransformedTargetRegressor(
        regressor=LinearRegression(), transformer=transformer
    )
    y_pred = regr.fit(X, y).predict(X)
    assert y.shape == y_pred.shape


def test_transform_target_regressor_multi_to_single():
    # [FUNCTION DOC]
    # Two-column target mapped by func to a single column (Euclidean norm of the two
    # columns), identity inverse, check_inverse off. Checks predictions are (100, 1)
    # whether func returns a 2D column or a 1D array, and that both agree.
    X = friedman[0]
    y = np.transpose([friedman[1], (friedman[1] ** 2 + 1)])

    def func(y):
        # [FUNCTION DOC]
        # HELPER: returns the row-wise norm of the first two columns as a (n, 1) array.
        out = np.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)
        return out[:, np.newaxis]

    def inverse_func(y):
        # [FUNCTION DOC]
        # HELPER: identity inverse; returns y unchanged.
        return y

    tt = TransformedTargetRegressor(
        func=func, inverse_func=inverse_func, check_inverse=False
    )
    tt.fit(X, y)
    y_pred_2d_func = tt.predict(X)
    assert y_pred_2d_func.shape == (100, 1)

    # force that the function only return a 1D array
    def func(y):
        # [FUNCTION DOC]
        # HELPER: same norm as the first func but returned as a 1D array of shape (n,).
        return np.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)

    tt = TransformedTargetRegressor(
        func=func, inverse_func=inverse_func, check_inverse=False
    )
    tt.fit(X, y)
    y_pred_1d_func = tt.predict(X)
    assert y_pred_1d_func.shape == (100, 1)

    assert_allclose(y_pred_1d_func, y_pred_2d_func)


class DummyCheckerArrayTransformer(TransformerMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: test transformer asserting every input it receives is a numpy ndarray;
    # transform and inverse_transform are identities.
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Asserts X is an ndarray; returns self.
        assert isinstance(X, np.ndarray)
        return self

    def transform(self, X):
        # [FUNCTION DOC]
        # Asserts X is an ndarray; returns X unchanged.
        assert isinstance(X, np.ndarray)
        return X

    def inverse_transform(self, X):
        # [FUNCTION DOC]
        # Asserts X is an ndarray; returns X unchanged.
        assert isinstance(X, np.ndarray)
        return X


class DummyCheckerListRegressor(DummyRegressor):
    # [CLASS DOC]
    # HELPER: DummyRegressor asserting X is passed as a Python list to fit/predict.
    def fit(self, X, y, sample_weight=None):
        # [FUNCTION DOC]
        # Asserts X is a list then delegates to DummyRegressor.fit. Outputs: self.
        assert isinstance(X, list)
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        # [FUNCTION DOC]
        # Asserts X is a list then delegates to DummyRegressor.predict.
        assert isinstance(X, list)
        return super().predict(X)


def test_transform_target_regressor_ensure_y_array():
    # [FUNCTION DOC]
    # Verifies y is converted to an ndarray before reaching the transformer while X
    # lists are passed through to the regressor untouched; passing an ndarray X then
    # triggers AssertionError in the checker regressor for fit and predict.
    # check that the target ``y`` passed to the transformer will always be a
    # numpy array. Similarly, if ``X`` is passed as a list, we check that the
    # predictor receive as it is.
    X, y = friedman
    tt = TransformedTargetRegressor(
        transformer=DummyCheckerArrayTransformer(),
        regressor=DummyCheckerListRegressor(),
        check_inverse=False,
    )
    tt.fit(X.tolist(), y.tolist())
    tt.predict(X.tolist())
    with pytest.raises(AssertionError):
        tt.fit(X, y.tolist())
    with pytest.raises(AssertionError):
        tt.predict(X)


class DummyTransformer(TransformerMixin, BaseEstimator):
    # [CLASS DOC]
    # HELPER: identity transformer counting fit calls in fit_counter.
    """Dummy transformer which count how many time fit was called."""

    def __init__(self, fit_counter=0):
        # [FUNCTION DOC]
        # Stores fit_counter (int, default 0).
        self.fit_counter = fit_counter

    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Increments fit_counter; returns self. Side effects: mutates fit_counter.
        self.fit_counter += 1
        return self

    def transform(self, X):
        # [FUNCTION DOC]
        # Identity; returns X.
        return X

    def inverse_transform(self, X):
        # [FUNCTION DOC]
        # Identity; returns X.
        return X


@pytest.mark.parametrize("check_inverse", [False, True])
def test_transform_target_regressor_count_fit(check_inverse):
    # [FUNCTION DOC]
    # Regression (gh-11618): with check_inverse False or True, the transformer is
    # fitted exactly once (fit_counter == 1).
    # regression test for gh-issue #11618
    # check that we only call a single time fit for the transformer
    X, y = friedman
    ttr = TransformedTargetRegressor(
        transformer=DummyTransformer(), check_inverse=check_inverse
    )
    ttr.fit(X, y)
    assert ttr.transformer_.fit_counter == 1


class DummyRegressorWithExtraFitParams(DummyRegressor):
    # [CLASS DOC]
    # HELPER: DummyRegressor whose fit accepts an extra check_input kwarg and asserts
    # it was passed as False.
    def fit(self, X, y, sample_weight=None, check_input=True):
        # [FUNCTION DOC]
        # Asserts check_input is falsy, then delegates to DummyRegressor.fit.
        # Outputs: self. Raises AssertionError when check_input is truthy (the default).
        # on the test below we force this to false, we make sure this is
        # actually passed to the regressor
        assert not check_input
        return super().fit(X, y, sample_weight)


def test_transform_target_regressor_pass_fit_parameters():
    # [FUNCTION DOC]
    # Verifies extra fit kwargs (check_input=False) are forwarded to the regressor
    # and that the transformer is fitted once.
    X, y = friedman
    regr = TransformedTargetRegressor(
        regressor=DummyRegressorWithExtraFitParams(), transformer=DummyTransformer()
    )

    regr.fit(X, y, check_input=False)
    assert regr.transformer_.fit_counter == 1


def test_transform_target_regressor_route_pipeline():
    # [FUNCTION DOC]
    # Verifies a step-prefixed fit kwarg (est__check_input) routed through Pipeline
    # reaches the inner regressor; the transformer is fitted once.
    X, y = friedman

    regr = TransformedTargetRegressor(
        regressor=DummyRegressorWithExtraFitParams(), transformer=DummyTransformer()
    )
    estimators = [("normalize", StandardScaler()), ("est", regr)]

    pip = Pipeline(estimators)
    pip.fit(X, y, **{"est__check_input": False})

    assert regr.transformer_.fit_counter == 1


class DummyRegressorWithExtraPredictParams(DummyRegressor):
    # [CLASS DOC]
    # HELPER: DummyRegressor whose predict records predict_called and asserts the
    # extra check_input kwarg was passed as False.
    def predict(self, X, check_input=True):
        # [FUNCTION DOC]
        # Sets predict_called = True, asserts check_input is falsy, delegates to
        # DummyRegressor.predict. Side effects: sets predict_called on self.
        # In the test below we make sure that the check input parameter is
        # passed as false
        self.predict_called = True
        assert not check_input
        return super().predict(X)


def test_transform_target_regressor_pass_extra_predict_parameters():
    # [FUNCTION DOC]
    # Verifies predict kwargs (check_input=False) are forwarded to the regressor's
    # predict, checked via predict_called.
    # Checks that predict kwargs are passed to regressor.
    X, y = friedman
    regr = TransformedTargetRegressor(
        regressor=DummyRegressorWithExtraPredictParams(), transformer=DummyTransformer()
    )

    regr.fit(X, y)
    regr.predict(X, check_input=False)
    assert regr.regressor_.predict_called


@pytest.mark.parametrize("output_format", ["pandas", "polars"])
def test_transform_target_regressor_not_warns_with_global_output_set(output_format):
    # [FUNCTION DOC]
    # Regression (gh-29361): with global transform_output pandas or polars, fitting
    # with func/inverse_func emits no warnings (all warnings turned into errors).
    """Test that TransformedTargetRegressor will not raise warnings if
    set_config(transform_output="pandas"/"polars") is set globally; regression test for
    issue #29361."""
    X, y = datasets.make_regression()
    y = np.abs(y) + 1
    with config_context(transform_output=output_format):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TransformedTargetRegressor(
                regressor=LinearRegression(), func=np.log, inverse_func=np.exp
            ).fit(X, y)


class ValidateDimensionRegressor(BaseEstimator):
    # [CLASS DOC]
    # HELPER: regressor that asserts in fit that y has exactly ndim dimensions.
    """A regressor that expects the target to have a specific number of dimensions."""

    def __init__(self, ndim):
        # [FUNCTION DOC]
        # Stores ndim (expected target dimensionality, int).
        self.ndim = ndim

    def fit(self, X, y):
        # [FUNCTION DOC]
        # Asserts y.ndim equals self.ndim.
        # KNOWN ISSUE: returns None instead of self; harmless here as the return value
        # is not used.
        assert y.ndim == self.ndim

    def predict(self, X):
        # [FUNCTION DOC]
        # Never called (no-op returning None).
        pass  # pragma: no cover


@pytest.mark.parametrize("ndim", [1, 2])
def test_transform_target_regressor_preserves_input_shape(ndim):
    # [FUNCTION DOC]
    # Regression (gh-26530): the regressor inside TransformedTargetRegressor receives
    # a target with the same ndim (1 or 2) as the input y, checked by the asserting
    # regressor during fit.
    """Check that TransformedTargetRegressor internally preserves the shape of the input

    non-regression test for issue #26530.
    """
    X, y = datasets.make_regression(n_samples=10, n_features=5, random_state=42)
    if ndim == 2:
        y = y.reshape(-1, 1)

    regr = TransformedTargetRegressor(regressor=ValidateDimensionRegressor(ndim))
    regr.fit(X, y)


@config_context(enable_metadata_routing=True)
def test_transform_target_regressor_metadata_routing_default_estimator():
    # [FUNCTION DOC]
    # Runs with metadata routing enabled: fitting with sample_weight using the default
    # regressor (LinearRegression) must not raise, i.e. the default regressor
    # requests sample_weight. Passes if no exception; no explicit assertion.
    """Test that metadata request is set on the default regressor"""
    X, y = make_regression()
    ttr = TransformedTargetRegressor()
    ttr.fit(X, y, sample_weight=np.ones(shape=(X.shape[0],)))
