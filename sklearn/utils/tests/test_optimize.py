# [MODULE DOC]
# Tests for sklearn.utils.optimize: the private Newton-CG solver _newton_cg (accuracy vs
# scipy, Array API support, verbose output and warning paths) and
# _check_optimize_result's ConvergenceWarning messages.
import warnings

import numpy as np
import pytest
from scipy.optimize import fmin_ncg

from sklearn import config_context
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._array_api import move_to, yield_namespace_device_dtype_combinations
from sklearn.utils._bunch import Bunch
from sklearn.utils._testing import _array_api_for_tests, assert_allclose
from sklearn.utils.optimize import _check_optimize_result, _newton_cg


def test_newton_cg(global_random_seed):
    # [FUNCTION DOC]
    # On a random 10x10 matrix A (seeded by global_random_seed), minimizes the quadratic
    # one half of the squared norm of A x from x0 = ones; checks _newton_cg reaches 0
    # within 1e-7 and agrees with scipy fmin_ncg within atol 1e-5.
    # KNOWN ISSUE: the hess callback passed to fmin_ncg computes the product using
    # x.all() (a scalar boolean) instead of the vector p, so it is not the Hessian-
    # vector product A^T A p; the comparison only passes because the scipy run still
    # converges near 0.
    # Test that newton_cg gives same result as scipy's fmin_ncg

    rng = np.random.RandomState(global_random_seed)
    A = rng.normal(size=(10, 10))
    x0 = np.ones(10)

    def func(x):
        # [FUNCTION DOC]
        # HELPER: Objective, one half of the squared norm of A x. Inputs: x - 1D array
        # (10,).
        # Outputs: float.
        Ax = A.dot(x)
        return 0.5 * (Ax).dot(Ax)

    def grad(x):
        # [FUNCTION DOC]
        # HELPER: Gradient A^T A x. Inputs: x - 1D array. Outputs: 1D array (10,).
        return A.T.dot(A.dot(x))

    def hess(x, p):
        # [FUNCTION DOC]
        # HELPER: Intended Hessian-vector product for fmin_ncg.
        # Inputs: x - current point; p - direction vector.
        # Outputs: array computed from p dotted with A^T A applied to x.all().
        # KNOWN ISSUE: uses x.all() rather than p for the Hessian application; result is
        # not the true Hessian-vector product.
        return p.dot(A.T.dot(A.dot(x.all())))

    def grad_hess(x):
        # [FUNCTION DOC]
        # HELPER: Returns the gradient at x and a callable computing the Hessian-vector
        # product A^T A z, as expected by _newton_cg.
        return grad(x), lambda x: A.T.dot(A.dot(x))

    # func is a definite positive quadratic form, so the minimum is at x = 0
    # hence the use of absolute tolerance.
    assert np.all(np.abs(_newton_cg(grad_hess, func, grad, x0, tol=1e-10)[0]) <= 1e-7)
    assert_allclose(
        _newton_cg(grad_hess, func, grad, x0, tol=1e-7)[0],
        fmin_ncg(f=func, x0=x0, fprime=grad, fhess_p=hess),
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "array_namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
def test_newton_cg_array_api_compliance(array_namespace, device_name, dtype_name):
    # [FUNCTION DOC]
    # For each namespace/device/dtype combination, solves the least squares problem with
    # A = [[3, -1], [-1, 1]], y = [0, 1] under array_api_dispatch and checks the
    # solution moved to numpy equals [1/2, 3/2] within 1e-10.
    """Test that newton_cg works with Array API input."""
    xp, device = _array_api_for_tests(array_namespace, device_name)
    A = xp.asarray(np.array([[3, -1], [-1, 1]]).astype(dtype_name), device=device)
    y = xp.asarray(np.arange(2).astype(dtype_name), device=device)
    x0 = xp.asarray(np.ones(2).astype(dtype_name), device=device)

    def func(x):
        # [FUNCTION DOC]
        # HELPER: Least squares objective, one half of the squared norm of y - A x.
        return 0.5 * (y - A @ x) @ (y - A @ x)

    def grad(x):
        # [FUNCTION DOC]
        # HELPER: Gradient A^T (A x - y).
        return A.T @ (A @ x - y)

    def hess(x, p):
        # [FUNCTION DOC]
        # HELPER: Hessian-vector product A^T A p (independent of x).
        return A.T @ (A @ p)

    def grad_hess(x):
        # [FUNCTION DOC]
        # HELPER: Returns grad(x) and a closure mapping p to hess(x, p).
        return grad(x), lambda p: hess(x, p)

    with config_context(array_api_dispatch=True):
        res = _newton_cg(grad_hess, func, grad, x0, tol=1e-10)

    assert_allclose(
        move_to(res[0], xp=np, device="cpu"),
        [1 / 2, 3 / 2],
        atol=1e-10,
    )


@pytest.mark.parametrize("verbose", [0, 1, 2])
def test_newton_cg_verbosity(capsys, verbose):
    # [FUNCTION DOC]
    # For verbose 0, 1, 2 on an identity quadratic, checks captured stdout is empty at 0
    # and contains iteration/convergence messages otherwise, plus inner CG and line
    # search messages at 2. At verbose 2 it also runs pathological problems (huge
    # collinear Hessian, near-singular Hessian, wrong-sign Hessian, ill-conditioned
    # diagonal with maxinner=1) and checks the expected warnings
    # (RuntimeWarning/UserWarning or ConvergenceWarning) and diagnostic messages for
    # line search fallbacks, tiny curvature, steepest-descent fallback and inner
    # maxiter.
    # Note: the two consecutive "verbose >= 2" branches could be merged; behavior is
    # same.
    """Test the std output of verbose newton_cg solver."""
    A = np.eye(2)
    b = np.array([1, 2], dtype=float)

    _newton_cg(
        grad_hess=lambda x: (A @ x - b, lambda z: A @ z),
        func=lambda x: 0.5 * x @ A @ x - b @ x,
        grad=lambda x: A @ x - b,
        x0=np.zeros(A.shape[0]),
        verbose=verbose,
    )  # returns array([1., 2])
    captured = capsys.readouterr()

    if verbose == 0:
        assert captured.out == ""
    else:
        msg = [
            "Newton-CG iter = 1",
            "Check Convergence",
            "max |gradient|",
            "Solver did converge at loss = ",
        ]
        for m in msg:
            assert m in captured.out

    if verbose >= 2:
        msg = [
            "Inner CG solver iteration 1 stopped with",
            "sum(|residuals|) <= tol",
            "Line Search",
            "try line search wolfe1",
            "wolfe1 line search was successful",
        ]
        for m in msg:
            assert m in captured.out

    if verbose >= 2:
        # Set up a badly scaled singular Hessian with a completely wrong starting
        # position. This should trigger 2nd line search check
        A = np.array([[1.0, 2], [2, 4]]) * 1e30  # collinear columns
        b = np.array([1.0, 2.0])
        # Note that scipy.optimize._linesearch LineSearchWarning inherits from
        # RuntimeWarning, but we do not want to import from non public APIs.
        with pytest.warns((RuntimeWarning, UserWarning)):
            _newton_cg(
                grad_hess=lambda x: (A @ x - b, lambda z: A @ z),
                func=lambda x: 0.5 * x @ A @ x - b @ x,
                grad=lambda x: A @ x - b,
                x0=np.array([-2.0, 1]),  # null space of hessian
                verbose=verbose,
            )
        captured = capsys.readouterr()
        msg = [
            "wolfe1 line search was not successful",
            "check loss |improvement| <= eps * |loss_old|:",
            "check sum(|gradient|) < sum(|gradient_old|):",
            "last resort: try line search wolfe2",
        ]
        for m in msg:
            assert m in captured.out

        # Set up a badly conditioned Hessian that leads to tiny curvature.
        # X.T @ X have singular values array([1.00000400e+01, 1.00008192e-11])
        A = np.array([[1.0, 2], [1, 2 + 1e-15]])
        b = np.array([-2.0, 1])
        with pytest.warns(ConvergenceWarning):
            _newton_cg(
                grad_hess=lambda x: (A @ x - b, lambda z: A @ z),
                func=lambda x: 0.5 * x @ A @ x - b @ x,
                grad=lambda x: A @ x - b,
                x0=b,
                verbose=verbose,
                maxiter=2,
            )
        captured = capsys.readouterr()
        msg = [
            "tiny_|p| = eps * ||p||^2",
        ]
        for m in msg:
            assert m in captured.out

        # Test for a case with negative Hessian.
        # We do not trigger "Inner CG solver iteration {i} stopped with negative
        # curvature", but that is very hard to trigger.
        A = np.eye(2)
        b = np.array([-2.0, 1])
        with pytest.warns((RuntimeWarning, UserWarning)):
            _newton_cg(
                # Note the wrong sign in the hessian product.
                grad_hess=lambda x: (A @ x - b, lambda z: -A @ z),
                func=lambda x: 0.5 * x @ A @ x - b @ x,
                grad=lambda x: A @ x - b,
                x0=np.array([1.0, 1.0]),
                verbose=verbose,
                maxiter=3,
            )
        captured = capsys.readouterr()
        msg = [
            "Inner CG solver iteration 0 fell back to steepest descent",
        ]
        for m in msg:
            assert m in captured.out

        A = np.diag([1e-3, 1, 1e3])
        b = np.array([-2.0, 1, 2.0])
        with pytest.warns(ConvergenceWarning):
            _newton_cg(
                grad_hess=lambda x: (A @ x - b, lambda z: A @ z),
                func=lambda x: 0.5 * x @ A @ x - b @ x,
                grad=lambda x: A @ x - b,
                x0=np.ones_like(b),
                verbose=verbose,
                maxiter=2,
                maxinner=1,
            )
        captured = capsys.readouterr()
        msg = [
            "Inner CG solver stopped reaching maxiter=1",
        ]
        for m in msg:
            assert m in captured.out


def test_check_optimize():
    # [FUNCTION DOC]
    # Uses a Bunch mocking lbfgs output to verify _check_optimize_result: no warning for
    # status 0; for status 1 without max_iter a single ConvergenceWarning mentioning the
    # iteration count and message and data scaling but not increasing iterations; with
    # max_iter it does recommend increasing iterations; for another status (2) with
    # max_iter it does not.
    # Mock some lbfgs output using a Bunch instance:
    result = Bunch()

    # First case: no warnings
    result.nit = 1
    result.status = 0
    result.message = "OK"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_optimize_result("lbfgs", result)

    # Second case: warning about implicit `max_iter`: do not recommend the user
    # to increase `max_iter` this is not a user settable parameter.
    result.status = 1
    result.message = "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
    with pytest.warns(ConvergenceWarning) as record:
        _check_optimize_result("lbfgs", result)

    assert len(record) == 1
    warn_msg = record[0].message.args[0]
    assert "lbfgs failed to converge after 1 iteration(s)" in warn_msg
    assert result.message in warn_msg
    assert "Increase the number of iterations" not in warn_msg
    assert "scale the data" in warn_msg

    # Third case: warning about explicit `max_iter`: recommend user to increase
    # `max_iter`.
    with pytest.warns(ConvergenceWarning) as record:
        _check_optimize_result("lbfgs", result, max_iter=1)

    assert len(record) == 1
    warn_msg = record[0].message.args[0]
    assert "lbfgs failed to converge after 1 iteration(s)" in warn_msg
    assert result.message in warn_msg
    assert "Increase the number of iterations" in warn_msg
    assert "scale the data" in warn_msg

    # Fourth case: other convergence problem before reaching `max_iter`: do not
    # recommend increasing `max_iter`.
    result.nit = 2
    result.status = 2
    result.message = "ABNORMAL"
    with pytest.warns(ConvergenceWarning) as record:
        _check_optimize_result("lbfgs", result, max_iter=10)

    assert len(record) == 1
    warn_msg = record[0].message.args[0]
    assert "lbfgs failed to converge after 2 iteration(s)" in warn_msg
    assert result.message in warn_msg
    assert "Increase the number of iterations" not in warn_msg
    assert "scale the data" in warn_msg
