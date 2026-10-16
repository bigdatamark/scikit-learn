# [MODULE DOC]
# Estimator-side support for callbacks: a mixin to register callbacks and build the
# root CallbackContext, plus a context manager and decorator guaranteeing that the
# callbacks' teardown hooks run after fit.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import functools
from contextlib import contextmanager

from sklearn.callback._base import AutoPropagatedCallback, FitCallback
from sklearn.callback._callback_context import CallbackContext


class CallbackSupportMixin:
    # [CLASS DOC]
    # Mixin giving an estimator set_callbacks and _init_callback_context. State stored on
    # the estimator: _skl_callbacks (list), _callback_fit_ctx and _skl_callbacks_to_teardown
    # (only during fit; deleted by callback_management_context).
    """Mixin class to add callback support to an estimator.

    .. document the private method
    .. automethod:: _init_callback_context
    """

    def set_callbacks(self, *callbacks):
        # [FUNCTION DOC]
        # Registers callbacks on the estimator.
        # Inputs: *callbacks - objects satisfying FitCallback (classes themselves rejected).
        # Outputs: self.
        # Exceptions: TypeError if any argument is not a FitCallback instance.
        # Side effects: sets self._skl_callbacks to a list of the callbacks; when called with
        # no arguments, removes the attribute instead (clearing callbacks).
        """Set callbacks for the estimator.

        Parameters
        ----------
        *callbacks : callback instances
            The callbacks to set.

        Returns
        -------
        self : estimator instance
            The estimator instance itself.
        """
        if not all(
            # isinstance for a Protocol returns True for classes too (not only
            # instances). Hence the additional check for classes.
            isinstance(callback, FitCallback) and not isinstance(callback, type)
            for callback in callbacks
        ):
            raise TypeError(
                "callbacks must be instances following the FitCallback protocol."
            )

        if callbacks:
            self._skl_callbacks = list(callbacks)
        else:
            self.__dict__.pop("_skl_callbacks", None)

        return self

    def _init_callback_context(
        self, task_name="fit", task_id=0, max_subtasks=0, sequential_subtasks=True
    ):
        # [FUNCTION DOC]
        # Builds the root CallbackContext of this fit and runs setup on owned callbacks.
        # Inputs: task_name (str, "fit"), task_id (int, 0), max_subtasks (int or None; 0 =
        # leaf), sequential_subtasks (bool, auto-numbered children if True).
        # Steps: create the context via CallbackContext._from_estimator; then for each
        # registered callback, skip it if it is auto-propagated AND the estimator has a
        # _parent_callback_ctx (i.e. it was propagated from a meta-estimator, whose owner
        # handles setup/teardown); otherwise record it for teardown and call its setup.
        # Outputs: the root CallbackContext.
        # Side effects: sets self._callback_fit_ctx and self._skl_callbacks_to_teardown.
        # If a setup raises, callbacks already appended (including the failing one) remain
        # recorded and will be torn down.
        """Initialize the callback context for the estimator and setup its callbacks.

        This method should be called once, at the beginning of the fit method.

        It will only setup callbacks that are not propagated from a meta-estimator.

        Parameters
        ----------
        task_name : str, default="fit"
            The name of the root task.

        task_id : int, default=0
            Identifier for the root task.

        max_subtasks : int or None, default=0
            The maximum number of subtasks that can be children of the root task. None
            means the maximum number of subtasks is not known in advance. 0 means it's a
            leaf.

        sequential_subtasks : bool, default=True
            Whether the root context has sequential subtasks. If True, children contexts
            of the root context, created via `subcontext`, will have automatically
            assigned consecutive integer task_ids starting from 0.

        Returns
        -------
        callback_fit_ctx : CallbackContext
            The root callback context for the estimator.
        """
        self._callback_fit_ctx = CallbackContext._from_estimator(
            estimator=self,
            task_name=task_name,
            task_id=task_id,
            max_subtasks=max_subtasks,
            sequential_subtasks=sequential_subtasks,
        )

        # Setup callbacks. We store callbacks for which setup has started in order to
        # only tear those down after fit.
        self._skl_callbacks_to_teardown = []
        for callback in getattr(self, "_skl_callbacks", []):
            # Only call the setup hook of callbacks that are not propagated from a
            # meta-estimator.
            if not (
                isinstance(callback, AutoPropagatedCallback)
                and hasattr(self, "_parent_callback_ctx")
            ):
                self._skl_callbacks_to_teardown.append(callback)
                callback.setup(estimator=self, context=self._callback_fit_ctx)

        return self._callback_fit_ctx


@contextmanager
def callback_management_context(estimator):
    # [FUNCTION DOC]
    # Generator-based context manager wrapping a fit call.
    # Inputs: estimator - the estimator being fitted.
    # Yields: None.
    # Behavior: on exit (success or error), if the estimator has _callback_fit_ctx (i.e.
    # _init_callback_context was called), call teardown on every recorded callback,
    # collecting exceptions instead of stopping, then delete _skl_callbacks_to_teardown
    # and _callback_fit_ctx.
    # Exceptions: re-raises a single teardown error as-is, or an ExceptionGroup for
    # several. A teardown error replaces/chains onto any exception raised by fit.
    # KNOWN ISSUE: ExceptionGroup is a builtin only on Python >= 3.11; on older
    # interpreters the multi-error branch would raise NameError.
    """Context manager to manage callback lifecycle around estimator fit.

    The context manager is responsible for calling the callbacks `teardown` hook in a
    `try finally` block, which guarantees that callbacks teardown will always be called,
    whether the estimator's fit exits successfully or not.

    Parameters
    ----------
    estimator : estimator instance
        The estimator being fitted.

    Yields
    ------
    None.
    """
    try:
        yield
    finally:
        if hasattr(estimator, "_callback_fit_ctx"):
            teardown_errors = []
            for callback in estimator._skl_callbacks_to_teardown:
                try:
                    callback.teardown(
                        estimator=estimator, context=estimator._callback_fit_ctx
                    )
                except Exception as exc:
                    teardown_errors.append(exc)

            del estimator._skl_callbacks_to_teardown
            del estimator._callback_fit_ctx
            if len(teardown_errors) == 1:
                raise teardown_errors[0]
            if teardown_errors:
                raise ExceptionGroup(
                    "The following callback teardown errors occurred",
                    teardown_errors,
                )


def with_callbacks(method):
    # [FUNCTION DOC]
    # Decorator wrapping an estimator method (typically fit) inside
    # callback_management_context. Inputs: method - unbound method taking the estimator
    # first. Outputs: wrapped function preserving metadata via functools.wraps.
    """Decorator to run the method within a callback context manager.

    This decorator is responsible for calling the callbacks `teardown` hooks of
    callbacks in a `try finally` block, which guarantees that callbacks teardown will
    always be called, whether the estimator's method exits successfully or not.

    It will only teardown callbacks that are not propagated from a meta-estimator.

    Parameters
    ----------
    method : method
        The method to decorate.

    Returns
    -------
    decorated_method : method
        The decorated method.
    """

    @functools.wraps(method)
    def callback_managed_method(estimator, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: The wrapper; calls method(estimator, *args, **kwargs) inside the
        # management context and returns its result.
        with callback_management_context(estimator):
            return method(estimator, *args, **kwargs)

    return callback_managed_method
