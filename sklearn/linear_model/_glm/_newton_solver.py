# [MODULE DOC]
# Second-order (Newton) solvers for GLMs built on LinearModelLoss. Provides the abstract
#   template NewtonSolver (setup, gradient/Hessian update, inner solve for the Newton
#   step, Armijo backtracking line search, convergence check, lbfgs fallback, finalize)
#   and the concrete NewtonCholeskySolver that solves the Newton system with a dense
#   symmetric solve. Used by _GeneralizedLinearRegressor (solver "newton-cholesky") and
#   LogisticRegression.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Newton solver for Generalized Linear Models
"""

import warnings
from abc import ABC, abstractmethod

import numpy as np
import scipy.linalg
import scipy.optimize

from sklearn._loss.loss import HalfSquaredError
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.utils.fixes import _get_additional_lbfgs_options_dict
from sklearn.utils.optimize import _check_optimize_result


class NewtonSolver(ABC):
    # [CLASS DOC]
    # Abstract template-method solver. Subclasses must implement update_gradient_hessian
    #   and inner_solve; may override setup and finalize. The driver is solve. State lives
    #   on self (coef, gradient, raw_prediction, loss_value, iteration, converged,
    #   use_fallback_lbfgs_solve, ...) and is overwritten during solve.
    """Newton solver for GLMs.

    This class implements Newton/2nd-order optimization routines for GLMs. Each Newton
    iteration aims at finding the Newton step which is done by the inner solver. With
    Hessian H, gradient g and coefficients coef, one step solves:

        H @ coef_newton = -g

    For our GLM / LinearModelLoss, we have gradient g and Hessian H:

        g = X.T @ loss.gradient + l2_reg_strength * coef
        H = X.T @ diag(loss.hessian) @ X + l2_reg_strength * identity

    Backtracking line search updates coef = coef_old + t * coef_newton for some t in
    (0, 1].

    This is a base class, actual implementations (child classes) may deviate from the
    above pattern and use structure specific tricks.

    Usage pattern:
        - initialize solver: sol = NewtonSolver(...)
        - solve the problem: sol.solve(X, y, sample_weight)

    References
    ----------
    - Jorge Nocedal, Stephen J. Wright. (2006) "Numerical Optimization"
      2nd edition
      https://doi.org/10.1007/978-0-387-40065-5

    - Stephen P. Boyd, Lieven Vandenberghe. (2004) "Convex Optimization."
      Cambridge University Press, 2004.
      https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf

    Parameters
    ----------
    coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
        Initial coefficients of a linear model.
        If shape (n_classes * n_dof,), the classes of one feature are contiguous,
        i.e. one reconstructs the 2d-array via
        coef.reshape((n_classes, -1), order="F").

    linear_loss : LinearModelLoss
        The loss to be minimized.

    l2_reg_strength : float, default=0.0
        L2 regularization strength.

    tol : float, default=1e-4
        The optimization problem is solved when each of the following condition is
        fulfilled:
        1. maximum |gradient| <= tol
        2. Newton decrement d: 1/2 * d^2 <= tol

    max_iter : int, default=100
        Maximum number of Newton steps allowed.

    n_threads : int, default=1
        Number of OpenMP threads to use for the computation of the Hessian and gradient
        of the loss function.

    Attributes
    ----------
    coef_old : ndarray of shape coef.shape
        Coefficient of previous iteration.

    coef_newton : ndarray of shape coef.shape
        Newton step.

    gradient : ndarray of shape coef.shape
        Gradient of the loss w.r.t. the coefficients.

    gradient_old : ndarray of shape coef.shape
        Gradient of previous iteration.

    loss_value : float
        Value of objective function = loss + penalty.

    loss_value_old : float
        Value of objective function of previous itertion.

    raw_prediction : ndarray of shape (n_samples,) or (n_samples, n_classes)

    converged : bool
        Indicator for convergence of the solver.

    iteration : int
        Number of Newton steps, i.e. calls to inner_solve

    use_fallback_lbfgs_solve : bool
        If set to True, the solver will resort to call LBFGS to finish the optimisation
        procedure in case of convergence issues.

    gradient_times_newton : float
        gradient @ coef_newton, set in inner_solve and used by line_search. If the
        Newton step is a descent direction, this is negative.
    """

    def __init__(
        self,
        *,
        coef,
        linear_loss=LinearModelLoss(base_loss=HalfSquaredError(), fit_intercept=True),
        l2_reg_strength=0.0,
        tol=1e-4,
        max_iter=100,
        n_threads=1,
        verbose=0,
    ):
        # [FUNCTION DOC]
        # Stores the arguments on self without validation or copying.
        # Inputs: coef - initial coefficients (may be mutated later by subclasses);
        #   linear_loss - LinearModelLoss (default squared error with intercept; the
        #   default instance is created once at definition time and shared);
        #   l2_reg_strength - float; tol - float; max_iter - int; n_threads - int for
        #   OpenMP; verbose - int, printing level.
        # Side effects: sets attributes only.
        self.coef = coef
        self.linear_loss = linear_loss
        self.l2_reg_strength = l2_reg_strength
        self.tol = tol
        self.max_iter = max_iter
        self.n_threads = n_threads
        self.verbose = verbose

    def setup(self, X, y, sample_weight):
        # [FUNCTION DOC]
        # Computes the initial raw predictions from self.coef and X and the initial
        #   objective value (loss plus L2 penalty) using those raw predictions.
        # Inputs: X, y, sample_weight as passed to solve.
        # Side effects: sets self.raw_prediction and self.loss_value.
        # KNOWN ISSUE: the docstring claims self.coef is initialized when None, but the
        #   code never handles a None coef.
        """Precomputations

        If None, initializes:
            - self.coef
        Sets:
            - self.raw_prediction
            - self.loss_value
        """
        _, _, self.raw_prediction = self.linear_loss.weight_intercept_raw(self.coef, X)
        self.loss_value = self.linear_loss.loss(
            coef=self.coef,
            X=X,
            y=y,
            sample_weight=sample_weight,
            l2_reg_strength=self.l2_reg_strength,
            n_threads=self.n_threads,
            raw_prediction=self.raw_prediction,
        )

    @abstractmethod
    def update_gradient_hessian(self, X, y, sample_weight):
        # [FUNCTION DOC]
        # Abstract. Must compute the current gradient (and Hessian or a substitute) at
        #   self.coef and store them on self for inner_solve and check_convergence.
        """Update gradient and Hessian."""

    @abstractmethod
    def inner_solve(self, X, y, sample_weight):
        # [FUNCTION DOC]
        # Abstract. Must compute the Newton direction into self.coef_newton and its
        #   inner product with the gradient into self.gradient_times_newton, or set
        #   self.use_fallback_lbfgs_solve to True to request the lbfgs fallback.
        """Compute Newton step.

        Sets:
            - self.coef_newton
            - self.gradient_times_newton
        """

    def fallback_lbfgs_solve(self, X, y, sample_weight):
        # [FUNCTION DOC]
        # Finishes the optimization with scipy L-BFGS-B starting from the current coef.
        #   Steps: flatten coef in Fortran order, run minimize on
        #   linear_loss.loss_gradient with the remaining iteration budget (max_iter minus
        #   iterations used), maxls 50, gtol tol and a tiny ftol; add the lbfgs iteration
        #   count to self.iteration; set converged when the scipy status is 0; reshape
        #   coef back to its original 2d shape if needed.
        # Inputs: X, y, sample_weight.
        # Side effects: replaces self.coef, updates self.iteration and self.converged;
        #   may emit ConvergenceWarning via _check_optimize_result; lbfgs may print when
        #   verbose > 1.
        """Fallback solver in case of emergency.

        If a solver detects convergence problems, it may fall back to this methods in
        the hope to exit with success instead of raising an error.

        Sets:
            - self.coef
            - self.converged
        """
        coef_shape = self.coef.shape
        self.coef = self.coef.ravel(order="F")  # scipy minimize expects 1d arrays
        max_iter = self.max_iter - self.iteration
        opt_res = scipy.optimize.minimize(
            self.linear_loss.loss_gradient,
            self.coef,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": max_iter,
                "maxls": 50,  # default is 20
                "gtol": self.tol,
                "ftol": 64 * np.finfo(np.float64).eps,
                **_get_additional_lbfgs_options_dict("iprint", self.verbose - 1),
            },
            args=(X, y, sample_weight, self.l2_reg_strength, self.n_threads),
        )
        self.iteration += _check_optimize_result("lbfgs", opt_res, max_iter=max_iter)
        self.coef = opt_res.x
        self.converged = opt_res.status == 0
        if len(coef_shape) > 1:
            self.coef = self.coef.reshape(coef_shape, order="F")

    def line_search(self, X, y, sample_weight):
        # [FUNCTION DOC]
        # Backtracking line search along self.coef_newton with shrink factor 1/2 and
        #   Armijo constant 2^-11. For up to 21 step sizes t (1, 1/2, ..., 2^-20): set
        #   coef to coef_old plus t times the Newton step, update raw predictions linearly
        #   (reusing the raw prediction of the step direction) and evaluate loss and
        #   gradient. Accept t if the loss decrease satisfies the Armijo condition (loss
        #   change at most t times sigma times gradient_times_newton). Otherwise, if the
        #   loss change is within 16 machine epsilons relative to the old loss, accept t
        #   when the L1 norm of the new gradient is smaller than that of the old gradient.
        #   If no t is accepted, warn and request the lbfgs fallback.
        # Inputs: X, y, sample_weight.
        # Outputs: none.
        # Side effects: sets coef_old, loss_value_old, gradient_old, coef, loss_value,
        #   gradient, and raw_prediction (only on success); emits ConvergenceWarning and
        #   sets use_fallback_lbfgs_solve on failure; prints when verbose.
        # KNOWN ISSUE: on failure coef, loss_value and gradient are left at the last
        #   (rejected, tiny) trial step while raw_prediction is not updated, so state is
        #   inconsistent; the fallback lbfgs then starts from that last trial coef.
        """Backtracking line search.

        Sets:
            - self.coef_old
            - self.coef
            - self.loss_value_old
            - self.loss_value
            - self.gradient_old
            - self.gradient
            - self.raw_prediction
        """
        # line search parameters
        beta, sigma = 0.5, 0.00048828125  # 1/2, 1/2**11
        # Remember: dtype follows X, also the one of self.loss_value. For Array API
        # support, self.loss_value might be float instead of np.floatXX.
        eps = 16 * np.finfo(X.dtype).eps
        t = 1  # step size

        # gradient_times_newton = self.gradient @ self.coef_newton
        # was computed in inner_solve.
        armijo_term = sigma * self.gradient_times_newton
        _, _, raw_prediction_newton = self.linear_loss.weight_intercept_raw(
            self.coef_newton, X
        )

        self.coef_old = self.coef
        self.loss_value_old = self.loss_value
        self.gradient_old = self.gradient

        # np.sum(np.abs(self.gradient_old))
        sum_abs_grad_old = -1

        is_verbose = self.verbose >= 2
        if is_verbose:
            print("  Backtracking Line Search")
            print(f"    eps=16 * finfo.eps={eps}")

        for i in range(21):  # until and including t = beta**20 ~ 1e-6
            self.coef = self.coef_old + t * self.coef_newton
            raw = self.raw_prediction + t * raw_prediction_newton
            self.loss_value, self.gradient = self.linear_loss.loss_gradient(
                coef=self.coef,
                X=X,
                y=y,
                sample_weight=sample_weight,
                l2_reg_strength=self.l2_reg_strength,
                n_threads=self.n_threads,
                raw_prediction=raw,
            )
            # Note: If coef_newton is too large, loss_gradient may produce inf values,
            # potentially accompanied by a RuntimeWarning.
            # This case will be captured by the Armijo condition.

            # 1. Check Armijo / sufficient decrease condition.
            # The smaller (more negative) the better.
            loss_improvement = self.loss_value - self.loss_value_old
            check = loss_improvement <= t * armijo_term
            if is_verbose:
                print(
                    f"    line search iteration={i + 1}, step size={t}\n"
                    f"      check loss improvement <= armijo term: {loss_improvement} "
                    f"<= {t * armijo_term} {check}"
                )
            if check:
                break
            # 2. Deal with relative loss differences around machine precision.
            tiny_loss = np.abs(self.loss_value_old * eps)
            check = np.abs(loss_improvement) <= tiny_loss
            if is_verbose:
                print(
                    "      check loss |improvement| <= eps * |loss_old|:"
                    f" {np.abs(loss_improvement)} <= {tiny_loss} {check}"
                )
            if check:
                if sum_abs_grad_old < 0:
                    sum_abs_grad_old = scipy.linalg.norm(self.gradient_old, ord=1)
                # 2.1 Check sum of absolute gradients as alternative condition.
                sum_abs_grad = scipy.linalg.norm(self.gradient, ord=1)
                check = sum_abs_grad < sum_abs_grad_old
                if is_verbose:
                    print(
                        "      check sum(|gradient|) < sum(|gradient_old|): "
                        f"{sum_abs_grad} < {sum_abs_grad_old} {check}"
                    )
                if check:
                    break

            t *= beta
        else:
            warnings.warn(
                (
                    f"Line search of Newton solver {self.__class__.__name__} at"
                    f" iteration #{self.iteration} did not converge after 21 line "
                    "search refinement iterations. It will now resort to lbfgs instead."
                ),
                ConvergenceWarning,
            )
            if self.verbose:
                print("  Line search did not converge and resorts to lbfgs instead.")
            self.use_fallback_lbfgs_solve = True
            return

        self.raw_prediction = raw
        if is_verbose:
            print(
                f"    line search successful after {i + 1} iterations with "
                f"loss={self.loss_value}."
            )

    def check_convergence(self, X, y, sample_weight):
        # [FUNCTION DOC]
        # Sets self.converged to True only if both hold: maximum absolute gradient entry
        #   is at most tol, and half the Newton decrement squared (coef_newton times
        #   Hessian times coef_newton) is at most tol. Otherwise returns leaving converged
        #   unchanged.
        # Inputs: X, y, sample_weight (used only to recompute the loss for verbose
        #   printing).
        # Side effects: sets self.converged; prints when verbose.
        # KNOWN ISSUE: relies on self.hessian, which the base class never defines; only
        #   subclasses that store a Hessian (NewtonCholeskySolver) can use it unchanged.
        """Check for convergence.

        Sets self.converged.
        """
        if self.verbose:
            print("  Check Convergence")
        # Note: Checking maximum relative change of coefficient <= tol is a bad
        # convergence criterion because even a large step could have brought us close
        # to the true minimum.
        # coef_step = self.coef - self.coef_old
        # change = np.max(np.abs(coef_step) / np.maximum(1, np.abs(self.coef_old)))
        # check = change <= tol

        # 1. Criterion: maximum |gradient| <= tol
        #    The gradient was already updated in line_search()
        g_max_abs = np.max(np.abs(self.gradient))
        check = g_max_abs <= self.tol
        if self.verbose:
            print(f"    1. max |gradient| {g_max_abs} <= {self.tol} {check}")
        if not check:
            return

        # 2. Criterion: For Newton decrement d, check 1/2 * d^2 <= tol
        #       d = sqrt(grad @ hessian^-1 @ grad)
        #         = sqrt(coef_newton @ hessian @ coef_newton)
        #    See Boyd, Vanderberghe (2009) "Convex Optimization" Chapter 9.5.1.
        d2 = self.coef_newton @ self.hessian @ self.coef_newton
        check = 0.5 * d2 <= self.tol
        if self.verbose:
            print(f"    2. Newton decrement {0.5 * d2} <= {self.tol} {check}")
        if not check:
            return

        if self.verbose:
            loss_value = self.linear_loss.loss(
                coef=self.coef,
                X=X,
                y=y,
                sample_weight=sample_weight,
                l2_reg_strength=self.l2_reg_strength,
                n_threads=self.n_threads,
            )
            print(f"  Solver did converge at loss = {loss_value}.")
        self.converged = True

    def finalize(self, X, y, sample_weight):
        # [FUNCTION DOC]
        # Hook for post-processing of self.coef after the loop; no-op in the base class.
        """Finalize the solvers results.

        Some solvers may need this, others not.
        """
        pass

    def solve(self, X, y, sample_weight):
        # [FUNCTION DOC]
        # Main driver. Calls setup, then iterates (iteration starting at 1, at most
        #   max_iter times, until converged): update_gradient_hessian, inner_solve,
        #   line_search, check_convergence, breaking out early if inner_solve or
        #   line_search requested the lbfgs fallback. If not converged afterwards, runs
        #   fallback_lbfgs_solve when requested, else emits a ConvergenceWarning. Finally
        #   decrements iteration so it counts completed steps, calls finalize, and returns
        #   self.coef.
        # Inputs: X - (n_samples, n_features) data; y - targets; sample_weight -
        #   (n_samples,) array.
        # Outputs: final coef array (shape depends on subclass handling, see class
        #   docstring).
        # Side effects: mutates many self attributes (iteration, converged,
        #   use_fallback_lbfgs_solve, coef, ...); warnings and prints.
        """Solve the optimization problem.

        This is the main routine.

        Order of calls:
            self.setup()
            while iteration:
                self.update_gradient_hessian()
                self.inner_solve()
                self.line_search()
                self.check_convergence()
            self.finalize()

        Returns
        -------
        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
            Solution of the optimization problem.
        """
        # setup usually:
        #   - initializes self.coef if needed
        #   - initializes and calculates self.raw_predictions, self.loss_value
        self.setup(X=X, y=y, sample_weight=sample_weight)

        self.iteration = 1
        self.converged = False
        self.use_fallback_lbfgs_solve = False

        while self.iteration <= self.max_iter and not self.converged:
            if self.verbose:
                print(f"Newton iter={self.iteration}")

            self.use_fallback_lbfgs_solve = False  # Fallback solver.

            # 1. Update Hessian and gradient
            self.update_gradient_hessian(X=X, y=y, sample_weight=sample_weight)

            # TODO:
            # if iteration == 1:
            # We might stop early, e.g. we already are close to the optimum,
            # usually detected by zero gradients at this stage.

            # 2. Inner solver
            #    Calculate Newton step/direction
            #    This usually sets self.coef_newton and self.gradient_times_newton.
            self.inner_solve(X=X, y=y, sample_weight=sample_weight)
            if self.use_fallback_lbfgs_solve:
                break

            # 3. Backtracking line search
            #    This usually sets self.coef_old, self.coef, self.loss_value_old
            #    self.loss_value, self.gradient_old, self.gradient,
            #    self.raw_prediction.
            self.line_search(X=X, y=y, sample_weight=sample_weight)
            if self.use_fallback_lbfgs_solve:
                break

            # 4. Check convergence
            #    Sets self.converged.
            self.check_convergence(X=X, y=y, sample_weight=sample_weight)

            # 5. Next iteration
            self.iteration += 1

        if not self.converged:
            if self.use_fallback_lbfgs_solve:
                # Note: The fallback solver circumvents check_convergence and relies on
                # the convergence checks of lbfgs instead. Enough warnings have been
                # raised on the way.
                self.fallback_lbfgs_solve(X=X, y=y, sample_weight=sample_weight)
            else:
                warnings.warn(
                    (
                        f"Newton solver did not converge after {self.iteration - 1} "
                        "iterations."
                    ),
                    ConvergenceWarning,
                )

        self.iteration -= 1
        self.finalize(X=X, y=y, sample_weight=sample_weight)
        return self.coef


class NewtonCholeskySolver(NewtonSolver):
    # [CLASS DOC]
    # Newton solver that forms the full dense Hessian (size coef.size squared) and
    #   solves the Newton system with scipy.linalg.solve assuming a symmetric matrix.
    #   Handles the overparametrized multinomial case by fixing the last class to zero (no
    #   penalty) or the last intercept to zero (with intercept), then converts back to the
    #   symmetric parametrization in finalize.
    """Cholesky based Newton solver.

    Inner solver for finding the Newton step H w_newton = -g uses Cholesky based linear
    solver.
    """

    def setup(self, X, y, sample_weight):
        # [FUNCTION DOC]
        # Runs the base setup, then: for multiclass losses flattens coef in Fortran
        #   order; allocates gradient (shape of coef) and square Hessian buffers; computes
        #   flags is_multinomial_with_intercept and is_multinomial_no_penalty. For the no-
        #   penalty multinomial, subtracts the last class's coefficients from every class
        #   so the last class becomes zero; for multinomial with intercept, subtracts the
        #   last intercept from all intercepts.
        # Inputs: X, y, sample_weight.
        # Side effects: sets raw_prediction, loss_value, coef, gradient, hessian and the
        #   two flags; the shift is done in place and may modify the caller's coef array
        #   when flattening returns a view.
        super().setup(X=X, y=y, sample_weight=sample_weight)
        if self.linear_loss.base_loss.is_multiclass:
            # Easier with ravelled arrays, e.g., for scipy.linalg.solve.
            # As with LinearModelLoss, we always are contiguous in n_classes.
            self.coef = self.coef.ravel(order="F")
        # Note that the computation of gradient in LinearModelLoss follows the shape of
        # coef.
        self.gradient = np.empty_like(self.coef)
        # But the hessian is always 2d.
        n = self.coef.size
        self.hessian = np.empty_like(self.coef, shape=(n, n))
        # To help case distinctions.
        self.is_multinomial_with_intercept = (
            self.linear_loss.base_loss.is_multiclass and self.linear_loss.fit_intercept
        )
        self.is_multinomial_no_penalty = (
            self.linear_loss.base_loss.is_multiclass and self.l2_reg_strength == 0
        )
        if self.is_multinomial_no_penalty:
            # See inner_solve. The provided coef might not adhere to the convention
            # that the last class is set to zero.
            # This is done by the usual freedom of a (overparametrized) multinomial to
            # add a constant to all classes which doesn't change predictions.
            n_classes = self.linear_loss.base_loss.n_classes
            coef = self.coef.reshape(n_classes, -1, order="F")  # easier as 2d
            coef -= coef[-1, :]  # coef -= coef of last class
        elif self.is_multinomial_with_intercept:
            # See inner_solve. Same as above, but only for the intercept.
            n_classes = self.linear_loss.base_loss.n_classes
            # intercept -= intercept of last class
            self.coef[-n_classes:] -= self.coef[-1]

    def update_gradient_hessian(self, X, y, sample_weight):
        # [FUNCTION DOC]
        # Computes gradient and Hessian at self.coef in place into the preallocated
        #   buffers using the raw predictions from the last line search.
        # Side effects: fills self.gradient and self.hessian; sets self.hessian_warning
        #   (True when many pointwise Hessian values were negative).
        _, _, self.hessian_warning = self.linear_loss.gradient_hessian(
            coef=self.coef,
            X=X,
            y=y,
            sample_weight=sample_weight,
            l2_reg_strength=self.l2_reg_strength,
            n_threads=self.n_threads,
            gradient_out=self.gradient,
            hessian_out=self.hessian,
            raw_prediction=self.raw_prediction,  # this was updated in line_search
        )

    def inner_solve(self, X, y, sample_weight):
        # [FUNCTION DOC]
        # Computes the Newton step. If hessian_warning is set, warn and request the
        #   lbfgs fallback. For the no-penalty multinomial, zero gradient and Hessian
        #   entries of the last class and solve the reduced system without them; for
        #   multinomial with intercept, zero and drop the last intercept entry; otherwise
        #   use full gradient and Hessian. Solve Hessian times step equals minus gradient
        #   (symmetric, LinAlgWarning promoted to error), re-insert zeros for removed
        #   entries, and compute gradient_times_newton. If that is positive (not a descent
        #   direction), request fallback.
        # Inputs: X, y, sample_weight (unused).
        # Side effects: sets coef_newton, gradient_times_newton; modifies gradient and
        #   hessian in place in multinomial cases; on singular or ill-conditioned Hessian
        #   emits LinAlgWarning and sets use_fallback_lbfgs_solve; ConvergenceWarning on
        #   hessian_warning; prints when verbose.
        if self.hessian_warning:
            warnings.warn(
                (
                    f"The inner solver of {self.__class__.__name__} detected a "
                    "pointwise hessian with many negative values at iteration "
                    f"#{self.iteration}. It will now resort to lbfgs instead."
                ),
                ConvergenceWarning,
            )
            if self.verbose:
                print(
                    "  The inner solver detected a pointwise Hessian with many "
                    "negative values and resorts to lbfgs instead."
                )
            self.use_fallback_lbfgs_solve = True
            return

        # Note: The following case distinction could also be shifted to the
        # implementation of HalfMultinomialLoss instead of here within the solver.
        if self.is_multinomial_no_penalty:
            # The multinomial loss is overparametrized for each unpenalized feature, so
            # at least the intercepts. This can be seen by noting that predicted
            # probabilities are invariant under shifting all coefficients of a single
            # feature j for all classes by the same amount c:
            #   coef[k, :] -> coef[k, :] + c    =>    proba stays the same
            # where we have assumed coef.shape = (n_classes, n_features).
            # Therefore, also the loss (-log-likelihood), gradient and hessian stay the
            # same, see
            # Noah Simon and Jerome Friedman and Trevor Hastie. (2013) "A Blockwise
            # Descent Algorithm for Group-penalized Multiresponse and Multinomial
            # Regression". https://doi.org/10.48550/arXiv.1311.6529
            #
            # We choose the standard approach and set all the coefficients of the last
            # class to zero, for all features including the intercept.
            # Note that coef was already dealt with in setup.
            n_classes = self.linear_loss.base_loss.n_classes
            n_dof = self.coef.size // n_classes  # degree of freedom per class
            n = self.coef.size - n_dof  # effective size
            self.gradient[n_classes - 1 :: n_classes] = 0
            self.hessian[n_classes - 1 :: n_classes, :] = 0
            self.hessian[:, n_classes - 1 :: n_classes] = 0
            # We also need the reduced variants of gradient and hessian where the
            # entries set to zero are removed. For 2 features and 3 classes with
            # arbitrary values, "x" means removed:
            #   gradient = [0, 1, x, 3, 4, x]
            #
            #   hessian = [0,  1, x,  3,  4, x]
            #             [1,  7, x,  9, 10, x]
            #             [x,  x, x,  x,  x, x]
            #             [3,  9, x, 21, 22, x]
            #             [4, 10, x, 22, 28, x]
            #             [x,  x, x,  x, x,  x]
            # The following slicing triggers copies of gradient and hessian.
            gradient = self.gradient.reshape(-1, n_classes)[:, :-1].flatten()
            hessian = self.hessian.reshape(n_dof, n_classes, n_dof, n_classes)[
                :, :-1, :, :-1
            ].reshape(n, n)
        elif self.is_multinomial_with_intercept:
            # Here, only intercepts are unpenalized. We again choose the last class and
            # set its intercept to zero.
            # Note that coef was already dealt with in setup.
            self.gradient[-1] = 0
            self.hessian[-1, :] = 0
            self.hessian[:, -1] = 0
            gradient, hessian = self.gradient[:-1], self.hessian[:-1, :-1]
        else:
            gradient, hessian = self.gradient, self.hessian

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", scipy.linalg.LinAlgWarning)
                self.coef_newton = scipy.linalg.solve(
                    hessian, -gradient, check_finite=False, assume_a="sym"
                )
                if self.is_multinomial_no_penalty:
                    self.coef_newton = np.c_[
                        self.coef_newton.reshape(n_dof, n_classes - 1), np.zeros(n_dof)
                    ].reshape(-1)
                    assert self.coef_newton.flags.f_contiguous
                elif self.is_multinomial_with_intercept:
                    self.coef_newton = np.r_[self.coef_newton, 0]
                self.gradient_times_newton = self.gradient @ self.coef_newton
                if self.gradient_times_newton > 0:
                    if self.verbose:
                        print(
                            "  The inner solver found a Newton step that is not a "
                            "descent direction and resorts to LBFGS steps instead."
                        )
                    self.use_fallback_lbfgs_solve = True
                    return
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgWarning) as e:
            warnings.warn(
                f"The inner solver of {self.__class__.__name__} stumbled upon a "
                "singular or very ill-conditioned Hessian matrix at iteration "
                f"{self.iteration}. It will now resort to lbfgs instead.\n"
                "Further options are to use another solver or to avoid such situation "
                "in the first place. Possible remedies are removing collinear features"
                " of X or increasing the penalization strengths.\n"
                "The original Linear Algebra message was:\n" + str(e),
                scipy.linalg.LinAlgWarning,
            )
            # Possible causes:
            # 1. hess_pointwise is negative. But this is already taken care in
            #    LinearModelLoss.gradient_hessian.
            # 2. X is singular or ill-conditioned
            #    This might be the most probable cause.
            #
            # There are many possible ways to deal with this situation. Most of them
            # add, explicitly or implicitly, a matrix to the hessian to make it
            # positive definite, confer to Chapter 3.4 of Nocedal & Wright 2nd ed.
            # Instead, we resort to lbfgs.
            if self.verbose:
                print(
                    "  The inner solver stumbled upon a singular or ill-conditioned "
                    "Hessian matrix and resorts to LBFGS instead."
                )
            self.use_fallback_lbfgs_solve = True
            return

    def finalize(self, X, y, sample_weight):
        # [FUNCTION DOC]
        # Converts coef to the symmetric multinomial parametrization: without penalty,
        #   reshape to (n_classes, n_dof) and subtract the per-feature mean over classes;
        #   with intercept only, subtract the mean of the intercepts. No change otherwise.
        # Side effects: replaces/mutates self.coef.
        if self.is_multinomial_no_penalty:
            # Our convention is usually the symmetric parametrization where
            # sum(coef[classes, features], axis=0) = 0.
            # We convert now to this convention. Note that it does not change
            # the predicted probabilities.
            n_classes = self.linear_loss.base_loss.n_classes
            self.coef = self.coef.reshape(n_classes, -1, order="F")
            self.coef -= np.mean(self.coef, axis=0)
        elif self.is_multinomial_with_intercept:
            # Only the intercept needs an update to the symmetric parametrization.
            n_classes = self.linear_loss.base_loss.n_classes
            self.coef[-n_classes:] -= np.mean(self.coef[-n_classes:])
