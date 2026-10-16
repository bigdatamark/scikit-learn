# [MODULE DOC]
# Gallery example teaching how third-party estimators and meta-estimators support the
# sklearn.callback infrastructure: shows a simplified KMeans and a list-based grid
# search first without, then with callback support (CallbackSupportMixin,
# with_callbacks, callback contexts/sub-contexts, propagation to sub-estimators), and
# runs both with a ProgressBar callback on random data (RandomState(42)).
# Side effects: progress output; classes are deliberately redefined (F811).
"""
==============================================
Supporting callbacks in third party estimators
==============================================

.. currentmodule:: sklearn.callback

This document shows how to make third party :term:`estimators` and
:term:`meta-estimators` compatible with the callback infrastructure supported by
scikit-learn.

Generally speaking, a callback is a function that is provided by the user to be called
at specific steps of a process, or to be triggered by specific events. Callbacks provide
a clean mechanism for inserting custom logic like monitoring progress or metrics,
without modifying the core algorithm of the process.

In scikit-learn, callbacks take the form of classes following a `protocol
<https://typing.python.org/en/latest/spec/protocol.html>`__. This protocol requires the
callback classes to implement specific methods (referred to as callback hooks) which
are called at specific steps of the fitting of an estimator or a meta-estimator.
These hooks are :meth:`~FitCallback.setup`, :meth:`~FitCallback.on_fit_task_begin`,
:meth:`~FitCallback.on_fit_task_end` and :meth:`~FitCallback.teardown`. The
:meth:`~FitCallback.setup` and :meth:`~FitCallback.teardown` hooks are called only once,
respectively at the start and end of the estimator's :term:`fit` method, and are
responsible for setting up and shutting down the callback. The
:meth:`~FitCallback.on_fit_task_begin` and :meth:`~FitCallback.on_fit_task_end` hooks
are respectively called at the beginning and end of each task in `fit` and are
responsible for the actual callback work. In scikit-learn estimators, a task in `fit` is
usually one step of a loop, with nested loops corresponding to nested tasks. In general,
a task can be whatever unit of work the estimator's developer wants it to be.

In order to support the callbacks, estimators need to initialize and manage
:class:`~CallbackContext` objects. As the name implies, these objects hold the
contextual information necessary to run the callback hooks. They are also responsible
for calling the callback hooks at the right time.

In the following, we show how to convert an example estimator class and an example
meta-estimator class to make them compliant with the scikit-learn callback
infrastructure.

First a few imports and some random data for the rest of the script.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

import numpy as np

from sklearn.base import BaseEstimator, clone
from sklearn.callback import CallbackSupportMixin, ProgressBar, with_callbacks
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import check_cv
from sklearn.utils import check_random_state
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted, validate_data

n_samples, n_features = 100, 4
rng = np.random.RandomState(42)
X = rng.rand(n_samples, n_features)


# %%
# Custom Estimator
# ----------------
# Here we demonstrate how to implement a custom estimator that supports callbacks. For
# the example, a simplified version of KMeans is presented. First, let's implement our
# `SimpleKMeans` estimator without the callback support.


class SimpleKMeans(BaseEstimator):
    # [CLASS DOC]
    # Plain (callback-free) simplified KMeans: random uniform centroid initialization
    # then a
    # fixed number of Lloyd iterations. Fitted attribute: cluster_centers_ of shape
    # (n_clusters, n_features).
    def __init__(self, n_clusters=6, n_iter=100, random_state=None):
        # [FUNCTION DOC]
        # Stores hyper-parameters: n_clusters (int, 6), n_iter (int, 100), random_state
        # (seed,
        # RandomState or None) controlling centroid initialization.
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.random_state = random_state  # controls the centroids' initialization

    def _compute_labels(self, X):
        # [FUNCTION DOC]
        # HELPER: index of the nearest centroid (Euclidean) for each row of X.
        # Output: int array (n_samples,).
        # Get the index of the closest centroid for each point in X.
        return np.argmin(euclidean_distances(X, self.cluster_centers_), axis=1)

    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Validates X (setting n_features_in_), initializes centroids uniformly in [0,
        # 1) per
        # feature, then for n_iter iterations assigns labels and moves each non-empty
        # cluster's
        # centroid to its members' mean (empty clusters keep their centroid). y is
        # ignored.
        # Outputs: self. Side effects: sets cluster_centers_; consumes random_state.
        # `y` is not used but we need to declare it to adhere to scikit-learn's
        # estimators fit convention.

        # Input validation is a good practice in estimators, for more information about
        # it you can refer to
        # https://scikit-learn.org/stable/developers/develop.html#input-validation.
        X = validate_data(self, X)
        random_state = check_random_state(self.random_state)
        # Randomly initialize the centroids.
        self.cluster_centers_ = random_state.rand(self.n_clusters, X.shape[1])

        for i in range(self.n_iter):
            # The fit iterations consist in getting the cluster label of each data point
            # according to their closest centroid, and then updating the centroids as
            # the center of each cluster.
            labels = self._compute_labels(X)

            for k in range(self.n_clusters):
                # For each centroid, if its cluster is not empty, its coordinates are
                # updated with the coordinates of the cluster's center.
                if (labels == k).any():
                    self.cluster_centers_[k] = X[labels == k].mean(axis=0)

        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # Checks fitted state and returns nearest-centroid labels for X.
        # KNOWN ISSUE: X is not validated (no validate_data with reset False), so
        # feature-count
        # mismatches are not reported in sklearn style.
        check_is_fitted(self)
        return self._compute_labels(X)

    def transform(self, X):
        # [FUNCTION DOC]
        # Checks fitted state and returns Euclidean distances to each centroid,
        # shape (n_samples, n_clusters). Same missing validation as predict.
        check_is_fitted(self)
        return euclidean_distances(X, self.cluster_centers_)


# %%
# Now let's add all the elements necessary to support callbacks.


# First things first, the estimator must inherit from the `CallbackSupportMixin` class.
class SimpleKMeans(CallbackSupportMixin, BaseEstimator):  # noqa: F811
    # [CLASS DOC]
    # Callback-enabled version of SimpleKMeans (inherits CallbackSupportMixin), with the
    # same hyper-parameters and fitted attributes.
    def __init__(self, n_clusters=6, n_iter=100, random_state=None):
        # [FUNCTION DOC]
        # Stores n_clusters, n_iter and random_state as in the plain version.
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.random_state = random_state

    def _compute_labels(self, X):
        # [FUNCTION DOC]
        # HELPER: index of the nearest centroid for each row of X.
        return np.argmin(euclidean_distances(X, self.cluster_centers_), axis=1)

    # Then the `fit` function must be decorated with the `with_callbacks`
    # decorator, which takes care of the proper teardown of callbacks.
    @with_callbacks
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Same algorithm as the plain fit, instrumented with callbacks: creates the root
        # context
        # ("fit", max_subtasks n_iter) and calls its begin hook; per iteration creates a
        # "fit iteration" sub-context, calls begin, updates centroids, then calls end
        # with a
        # reconstruction_attributes callable returning the current centroids; stops
        # early if
        # the end hook returns True. Finally calls the root context end hook.
        # Outputs: self. Side effects: sets cluster_centers_, invokes user callbacks;
        # teardown
        # is handled by the with_callbacks decorator even on error.
        X = validate_data(self, X)
        random_state = check_random_state(self.random_state)
        # The `CallbackContext` object must be instantiated with the
        # `_init_callback_context` method provided by the mixin, which calls the
        # `setup` hooks of the callbacks. This context corresponds to the root task of
        # the fit function.
        callback_ctx = self._init_callback_context(
            task_name="fit", max_subtasks=self.n_iter
        )
        # Then the callback context's `call_on_fit_task_begin` method must be called. It
        # will call all the callbacks' `on_fit_task_begin` hooks. The `estimator`
        # argument is mandatory and optional `kwargs` can be passed to provide extra
        # contextual information for the callbacks, for example here `X` and `y` are
        # passed. See the following note for more details on these extra `kwargs`.
        callback_ctx.call_on_fit_task_begin(estimator=self, X=X, y=y)

        self.cluster_centers_ = random_state.rand(self.n_clusters, X.shape[1])

        for i in range(self.n_iter):
            # For each sub-task of fit (here each iteration of the loop), a sub-context
            # must be created with the callback context's `subcontext` method.
            subcontext = callback_ctx.subcontext(task_name="fit iteration")
            # The sub-context corresponds to a new sub-task, so its
            # `call_on_fit_task_begin` method must also be called.
            subcontext.call_on_fit_task_begin(estimator=self, X=X, y=y)

            labels = self._compute_labels(X)

            for k in range(self.n_clusters):
                if (labels == k).any():
                    self.cluster_centers_[k] = X[labels == k].mean(axis=0)

            # After each sub-task, the `call_on_fit_task_end` method of its sub-context
            # must be called, also with `estimator` as a mandatory argument and optional
            # `kwargs`. It will call all the callbacks' `on_fit_task_end` hooks. Here
            # the extra `kwargs` contain a `reconstruction_attributes` callable,
            # which returns the necessary attributes to generate an estimator instance
            # ready to predict, as if the fit process just stopped at this step.
            if subcontext.call_on_fit_task_end(
                estimator=self,
                X=X,
                y=y,
                reconstruction_attributes=lambda: {
                    "cluster_centers_": self.cluster_centers_,
                },
            ):
                # The `call_on_fit_task_end` method returns a boolean, which is set
                # to True if any of the callbacks' `on_fit_task_end` methods return
                # True. This enables the interruption of the `fit` process by the
                # callbacks, for example to implement early stopping. Thus the
                # `call_on_fit_task_end` method can be used in an `if` / `break` block
                # to enable such interruptions.
                break

        # After the root task of the fit function is done, the `call_on_fit_task_end`
        # method of its callback context must be called.
        callback_ctx.call_on_fit_task_end(
            estimator=self,
            X=X,
            y=y,
            reconstruction_attributes=lambda: {
                "cluster_centers_": self.cluster_centers_,
            },
        )

        # The callbacks' `teardown` hooks are called automatically in the decorator,
        # after fit finishes, even if it crashed.
        return self

    def predict(self, X):
        # [FUNCTION DOC]
        # Checks fitted state and returns nearest-centroid labels (no input validation).
        check_is_fitted(self)
        return self._compute_labels(X)

    def transform(self, X):
        # [FUNCTION DOC]
        # Checks fitted state and returns distances to centroids (no input validation).
        check_is_fitted(self)
        return euclidean_distances(X, self.cluster_centers_)


# %%
# .. note::
#
#     See the documentation of the methods
#     :meth:`~CallbackContext.call_on_fit_task_begin` and
#     :meth:`~CallbackContext.call_on_fit_task_end` for the description of the `kwargs`
#     they can accept. These `kwargs` are optional, but an estimator should provide all
#     the ones it is capable of producing in each task to be compatible with a maximum
#     number of callbacks.

# %%
# Registering callbacks to the custom estimator
# ---------------------------------------------
# Now the `SimpleKMeans` estimator can be used with callbacks, for example with the
# :class:`~ProgressBar` callback to monitor progress.

estimator = SimpleKMeans(random_state=rng)
callback = ProgressBar()
estimator.set_callbacks(callback)
estimator.fit(X)

# %%
# Custom meta-estimator
# ---------------------
# Now we demonstrate how to implement a custom meta-estimator that supports callbacks.
# For the example, we implement a simplified version of a grid search, where only a list
# of parameter combinations is searched through instead of a grid, parallelizing the
# evaluation of the parameters.
# Let's start with the implementation without the callback support.


# Function to run in parallel, it fits and scores an estimator on the folds of a CV.
def _fit_and_score_cv(estimator, X, y, cv, score_func):
    # [FUNCTION DOC]
    # HELPER (callback-free version): for each CV split, fits a clone of estimator on
    # the
    # training fold (y sliced when given, else None) and scores it on the test fold with
    # score_func(estimator, X_test, y_test). Inputs: X, y - indexable arrays; cv -
    # splitter.
    # Outputs: list of per-fold scores.
    scores_per_fold = []
    # We iterate over the folds of the CV split.
    for train_idx, test_idx in cv.split(X):
        # A clone of the estimator is used for the current fold.
        cloned_estimator = clone(estimator)
        # The split of the current fold is applied to the data.
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = (y[train_idx], y[test_idx]) if y is not None else (None, None)
        # The clone of the estimator is fitted.
        cloned_estimator.fit(X_train, y_train)
        # Its score is computed.
        scores_per_fold.append(score_func(cloned_estimator, X_test, y_test))
    return scores_per_fold


class SimpleGridSearch(BaseEstimator):
    # [CLASS DOC]
    # Callback-free simplified grid search over an explicit list of parameter dicts,
    # evaluated in parallel with joblib. Fitted attribute: cv_results_, list of
    # (params, per-fold scores) tuples. No best-estimator selection or refit.
    def __init__(self, estimator, param_list, cv, score_func, n_jobs=1):
        # [FUNCTION DOC]
        # Stores estimator (template), param_list (list of dicts), cv (int or splitter),
        # score_func (callable(estimator, X, y) -> float) and n_jobs (default 1).
        # the estimator to evaluate
        self.estimator = estimator
        # the list of parameter combinations to iterate over
        self.param_list = param_list
        # the number of splits for the CV, or a CV splitter instance
        self.cv = cv
        # the scoring function
        self.score_func = score_func
        # number of jobs for parallelization
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Resolves cv with check_cv, runs _fit_and_score_cv in parallel for a clone of
        # the
        # estimator configured with each params dict, and stores cv_results_.
        # Outputs: self.
        # We use a cross-validator instance to evaluate each parameter combination on
        # multiple folds.
        cv = check_cv(self.cv)

        # We iterate over the parameter combinations in parallel, fitting an estimator
        # and computing a score value for each fold.
        scores_per_fold = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_and_score_cv)(
                estimator=clone(self.estimator).set_params(**params),
                X=X,
                y=y,
                cv=cv,
                score_func=self.score_func,
            )
            for params in self.param_list
        )

        # The `cv_results_` attribute holds the score values for each parameter
        # combination and fold, as a list of tuples, each one of the form
        # (parameter combination, list of scores per fold).
        self.cv_results_ = list(zip(self.param_list, scores_per_fold))

        return self


# %%
# Now let's update the class to support callbacks.


# The parallelized function needs to receive the callback context corresponding to its
# task and the instance calling it.
def _fit_and_score_cv(estimator, X, y, cv, score_func, outer_subcontext, caller):
    # [FUNCTION DOC]
    # HELPER (callback version): same as the plain version, but calls begin/end hooks on
    # outer_subcontext around the whole loop and, for each fold, creates a "fold i"
    # sub-context, propagates callbacks to the cloned estimator within it and calls its
    # begin/end hooks around fitting and scoring. caller (the meta-estimator) is passed
    # as
    # the estimator argument of the hooks. Outputs: list of per-fold scores.
    # The outer sub-context's `call_on_fit_task_begin` must be called.
    outer_subcontext.call_on_fit_task_begin(estimator=caller, X=X, y=y)
    scores_per_fold = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        cloned_estimator = clone(estimator)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = (y[train_idx], y[test_idx]) if y is not None else (None, None)
        # For each inner iteration a sub-context must be created.
        inner_subcontext = outer_subcontext.subcontext(task_name=f"fold {i}")
        # Since a sub-estimator is fitted in this task, the callbacks must be propagated
        # to that estimator with the `propagate_callback_context` context manager. Note
        # that only the callbacks following the `AutoPropagatedCallback` protocol can be
        # propagated.
        with inner_subcontext.propagate_callback_context(cloned_estimator):
            # After the propagation, the inner sub-context's `call_on_fit_task_begin`
            # method must be called.
            inner_subcontext.call_on_fit_task_begin(
                estimator=caller, X=X_train, y=y_train
            )
            cloned_estimator.fit(X_train, y_train)
            scores_per_fold.append(score_func(cloned_estimator, X_test, y_test))
            # The inner sub-context's `call_on_fit_task_end` method must be called.
            inner_subcontext.call_on_fit_task_end(
                estimator=caller, X=X_train, y=y_train
            )
    # The outer sub-context's `call_on_fit_task_end` method must be called.
    outer_subcontext.call_on_fit_task_end(estimator=caller, X=X, y=y)
    return scores_per_fold


# The class must inherit from `CallbackSupportMixin`.
class SimpleGridSearch(CallbackSupportMixin, BaseEstimator):  # noqa: F811
    # [CLASS DOC]
    # Callback-enabled SimpleGridSearch (inherits CallbackSupportMixin); same parameters
    # and
    # fitted attribute.
    def __init__(self, estimator, param_list, cv, score_func, n_jobs=1):
        # [FUNCTION DOC]
        # Stores estimator, param_list, cv, score_func and n_jobs as in the plain
        # version.
        self.estimator = estimator
        self.param_list = param_list
        self.cv = cv
        self.score_func = score_func
        self.n_jobs = n_jobs

    # The `fit` method must be decorated.
    @with_callbacks
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Resolves cv, creates the root context ("fit", one subtask per params dict) and
        # calls
        # its begin hook, creates all "param iteration" sub-contexts (max_subtasks =
        # number of
        # splits) before parallelizing to avoid races, runs the callback version of
        # _fit_and_score_cv in parallel, stores cv_results_ and calls the root end hook.
        # Outputs: self. Side effects: invokes callbacks, sets cv_results_.
        cv = check_cv(self.cv)
        # The callback context must be instantiated, which also calls the `setup` hooks
        # of the callbacks.
        callback_ctx = self._init_callback_context(
            task_name="fit", max_subtasks=len(self.param_list)
        )
        # The `call_on_fit_task_begin` method of this context must be called.
        callback_ctx.call_on_fit_task_begin(estimator=self, X=X, y=y)

        # The sub-tasks of the `fit` function are nested on two levels : the outer
        # iterations over parameter combinations and the inner iterations over CV folds.
        # Sub-contexts must be created for each of these levels. For the outer level,
        # the sub-contexts are instantiated outside of the parallelized function. In
        # order to prevent any racing condition during their creation, these
        # sub-contexts must be all created before the parallelization.
        outer_subcontexts = [
            callback_ctx.subcontext(
                task_name="param iteration", max_subtasks=cv.get_n_splits()
            )
            for _ in range(len(self.param_list))
        ]

        scores_per_fold = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_and_score_cv)(
                estimator=clone(self.estimator).set_params(**params),
                X=X,
                y=y,
                cv=cv,
                score_func=self.score_func,
                outer_subcontext=outer_subcontexts[i],
                caller=self,
            )
            for i, params in enumerate(self.param_list)
        )

        self.cv_results_ = list(zip(self.param_list, scores_per_fold))

        # The root context's `call_on_fit_task_end` must be called.
        callback_ctx.call_on_fit_task_end(estimator=self, X=X, y=y)

        # The callbacks' `teardown` hooks are called automatically in the decorator.
        return self


# %%
# The main difference with a simple estimator is that the callbacks must be propagated
# to the sub-estimators through the corresponding callback sub-context's
# :meth:`~CallbackContext.propagate_callback_context` context manager.

# %%
# .. note::
#
#     A meta-estimator that supports callback can be used with sub-estimators that do
#     not. In that case a warning is raised when trying to propagate the callbacks
#     and the callbacks are ignored in the sub-estimator.


# %%
# Registering callbacks to the meta-estimator
# -------------------------------------------
# Callbacks are registered to a meta-estimator the same way as to regular estimators.
# The callbacks which respect the :class:`~AutoPropagatedCallback` protocol (such as
# :class:`~ProgressBar`) are propagated to the sub-estimators.


param_list = [{"n_clusters": 5, "n_iter": 20}, {"n_clusters": 4, "n_iter": 50}]


def score_func(estimator, X, y=None):
    # [FUNCTION DOC]
    # HELPER: scoring function returning the sum over samples of the distance to the
    # nearest centroid (inertia-like, lower is better). y is ignored.
    # KNOWN ISSUE: named a score although larger values mean worse clustering; harmless
    # here
    # since cv_results_ is never used for selection.
    return np.sum(estimator.transform(X).min(axis=1))


sub_estimator = SimpleKMeans(random_state=rng)
meta_estimator = SimpleGridSearch(
    estimator=sub_estimator, param_list=param_list, cv=4, score_func=score_func
)
callback = ProgressBar()
meta_estimator.set_callbacks(callback)
meta_estimator.fit(X)
