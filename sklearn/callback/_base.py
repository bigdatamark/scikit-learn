# [MODULE DOC]
# Defines the typing Protocols that callbacks must satisfy. They are runtime_checkable,
# so isinstance checks only verify the presence of the hook methods/attributes, not
# their signatures. No behavior lives here: all hook bodies are docstring-only.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from typing import Protocol, runtime_checkable


@runtime_checkable
class _BaseCallback(Protocol):
    # [CLASS DOC]
    # HELPER: Base protocol requiring the setup and teardown hooks shared by all callbacks.
    """Protocol for the base callbacks."""

    def setup(self, estimator, context):
        # [FUNCTION DOC]
        # Protocol hook (no implementation). Called once before fit work starts.
        # Inputs: estimator - the estimator running fit; context - its CallbackContext.
        # Outputs: None expected.
        """Method called at the beginning of the fit method of the estimator.

        For auto-propagated callbacks, this method is called only once, before running
        the fit method of the outermost estimator.

        Parameters
        ----------
        estimator : estimator instance
            The estimator calling this callback hook.

        context : `sklearn.callback.CallbackContext` instance
            Context of the corresponding task. This is usually the root context of the
            estimator but it can be an intermediate context if the estimator is a
            sub-estimator of a meta-estimator.
        """

    def teardown(self, estimator, context):
        # [FUNCTION DOC]
        # Protocol hook (no implementation). Called once after fit finishes, even on error
        # (via callback_management_context). Inputs as in setup. Outputs: None expected.
        """Method called after finishing the fit method of the estimator.

        For auto-propagated callbacks, this method is called only once, after finishing
        the fit method of the outermost estimator.

        Parameters
        ----------
        estimator : estimator instance
            The estimator calling this callback hook.

        context : `sklearn.callback.CallbackContext` instance
            Context of the corresponding task. This is usually the root context of the
            estimator but it can be an intermediate context if the estimator is a
            sub-estimator of a meta-estimator.
        """


@runtime_checkable
class FitCallback(_BaseCallback, Protocol):
    # [CLASS DOC]
    # Protocol for callbacks receiving per-task begin/end notifications during fit.
    # Satisfied by any object with setup, teardown, on_fit_task_begin, on_fit_task_end.
    """Protocol for the callbacks evaluated on tasks during the fit of an estimator."""

    def on_fit_task_begin(
        self,
        estimator,
        context,
        *,
        X=None,
        y=None,
        metadata=None,
        fitted_estimator=None,
    ):
        # [FUNCTION DOC]
        # Protocol hook (no implementation). Called when a task of the context tree starts.
        # Inputs: estimator, context; keyword-only X, y (training data), metadata (dict, e.g.
        # sample weights), fitted_estimator (a usable snapshot); all default to None.
        # Outputs: return value ignored.
        """Method called at the beginning of each fit task of the estimator.

        Parameters
        ----------
        estimator : estimator instance
            The estimator calling this callback hook.

        context : `sklearn.callback.CallbackContext` instance
            Context of the corresponding task.

        X : array-like
            The training data at this task.

        y : array-like
            The training target values at this task.

        metadata : dict
            Training metadata at this task, e.g. sample weights.

        fitted_estimator : estimator instance
            A new instance of the estimator that is ready to predict, transform, etc ...
            as if fit had stopped at the beginning of this task.
        """

    def on_fit_task_end(
        self,
        estimator,
        context,
        *,
        X=None,
        y=None,
        metadata=None,
        fitted_estimator=None,
    ):
        # [FUNCTION DOC]
        # Protocol hook (no implementation). Called when a task ends; same inputs as
        # on_fit_task_begin. Outputs: truthy value requests stopping the current iteration
        # level (see CallbackContext.call_on_fit_task_end).
        """Method called at the end of each fit task of the estimator.

        Parameters
        ----------
        estimator : estimator instance
            The estimator calling this callback hook.

        context : `sklearn.callback.CallbackContext` instance
            Context of the corresponding task.

        X : array-like
            The training data at this task.

        y : array-like
            The training target values at this task.

        metadata : dict
            Training metadata at this task, e.g. sample weights.

        fitted_estimator : estimator instance
            A new instance of the estimator that is ready to predict, transform, etc ...
            as if fit had stopped at the end of this task.

        Returns
        -------
        stop : bool
            Whether or not to stop the current level of iterations at this task.
        """


@runtime_checkable
class AutoPropagatedCallback(_BaseCallback, Protocol):
    # [CLASS DOC]
    # Protocol for callbacks that are copied down to sub-estimators of meta-estimators.
    # Requires setup, teardown and a max_propagation_depth attribute/property.
    """Protocol for the auto-propagated callbacks

    An auto-propagated callback is a callback that is meant to be set on a top-level
    estimator and that is automatically propagated to its sub-estimators (if any).
    """

    @property
    def max_propagation_depth(self):
        # [FUNCTION DOC]
        # Protocol property (no implementation): maximum nesting depth (int) to which the
        # callback is propagated, or None for unlimited.
        """The maximum number of nested estimators at which the callback should be
        propagated.

        If set to None, the callback is propagated to sub-estimators at all nesting
        levels.
        """
