# [MODULE DOC]
# Defines PredictionErrorDisplay, a matplotlib scatter visualization of a
# regressor's predictions against targets ('actual_vs_predicted') or of residuals
# against predictions ('residual_vs_predicted'), with a reference line. No import-time
# side effects.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numbers

import numpy as np

from sklearn.utils import _safe_indexing, check_random_state
from sklearn.utils._optional_dependencies import check_matplotlib_support
from sklearn.utils._plotting import _validate_style_kwargs


class PredictionErrorDisplay:
    # [CLASS DOC]
    # Stores y_true / y_pred arrays and draws them. Attributes set by plot: line_
    # (Line2D reference line), scatter_ (PathCollection), ax_, figure_.
    """Visualization of the prediction error of a regression model.

    This tool can display "residuals vs predicted" or "actual vs predicted"
    using scatter plots to qualitatively assess the behavior of a regressor,
    preferably on held-out data points.

    See the details in the docstrings of
    :func:`~sklearn.metrics.PredictionErrorDisplay.from_estimator` or
    :func:`~sklearn.metrics.PredictionErrorDisplay.from_predictions` to
    create a visualizer. All parameters are stored as attributes.

    For general information regarding `scikit-learn` visualization tools, read
    more in the :ref:`Visualization Guide <visualizations>`.
    For details regarding interpreting these plots, refer to the
    :ref:`Model Evaluation Guide <visualization_regression_evaluation>`.

    .. versionadded:: 1.2

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True values.

    y_pred : ndarray of shape (n_samples,)
        Prediction values.

    Attributes
    ----------
    line_ : matplotlib Artist
        Optimal line representing `y_true == y_pred`. Therefore, it is a
        diagonal line for `kind="predictions"` and a horizontal line for
        `kind="residuals"`.

    errors_lines_ : matplotlib Artist or None
        Residual lines. If `with_errors=False`, then it is set to `None`.

    scatter_ : matplotlib Artist
        Scatter data points.

    ax_ : matplotlib Axes
        Axes with the different matplotlib axis.

    figure_ : matplotlib Figure
        Figure containing the scatter and lines.

    See Also
    --------
    PredictionErrorDisplay.from_estimator : Prediction error visualization
        given an estimator and some data.
    PredictionErrorDisplay.from_predictions : Prediction error visualization
        given the true and predicted targets.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.metrics import PredictionErrorDisplay
    >>> X, y = load_diabetes(return_X_y=True)
    >>> ridge = Ridge().fit(X, y)
    >>> y_pred = ridge.predict(X)
    >>> display = PredictionErrorDisplay(y_true=y, y_pred=y_pred)
    >>> display.plot()
    <...>
    >>> plt.show()
    """

    def __init__(self, *, y_true, y_pred):
        # [FUNCTION DOC]
        # Stores the inputs without validation.
        # Inputs: y_true, y_pred - array-like of shape (n_samples,).
        # Side effects: sets self.y_true and self.y_pred.
        self.y_true = y_true
        self.y_pred = y_pred

    def plot(
        self,
        ax=None,
        *,
        kind="residual_vs_predicted",
        scatter_kwargs=None,
        line_kwargs=None,
    ):
        # [FUNCTION DOC]
        # Validates kind, merges user style dicts over defaults (scatter: color
        #   tab:blue,
        # alpha 0.8; line: black, alpha 0.7, dashed) via _validate_style_kwargs, creates
        #   axes
        # if needed, then:
        # - actual_vs_predicted: draws the diagonal from the global min to the global
        #   max of
        # both arrays, scatters (y_pred, y_true), forces equal aspect (adjustable
        #   datalim) and
        # puts 5 evenly spaced ticks between min and max on both axes;
        # - residual_vs_predicted: draws a horizontal line at 0 spanning the range of
        #   y_pred and
        # scatters (y_pred, y_true - y_pred).
        # Sets axis labels accordingly.
        # Inputs: ax - Axes or None; kind - one of the two strings; scatter_kwargs,
        #   line_kwargs
        # - dicts or None.
        # Outputs: self.
        # Exceptions: ImportError without matplotlib; ValueError for an unknown kind;
        #   TypeError
        # from _validate_style_kwargs on conflicting aliases.
        # Side effects: draws; sets line_, scatter_, ax_, figure_.
        # KNOWN ISSUE: the defaults are merged a second time after
        #   _validate_style_kwargs
        # already merged them; redundant but harmless.
        """Plot visualization.

        Extra keyword arguments will be passed to matplotlib's ``plot``.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        kind : {"actual_vs_predicted", "residual_vs_predicted"}, \
                default="residual_vs_predicted"
            The type of plot to draw:

            - "actual_vs_predicted" draws the observed values (y-axis) vs.
              the predicted values (x-axis).
            - "residual_vs_predicted" draws the residuals, i.e. difference
              between observed and predicted values, (y-axis) vs. the predicted
              values (x-axis).

        scatter_kwargs : dict, default=None
            Dictionary with keywords passed to the `matplotlib.pyplot.scatter`
            call.

        line_kwargs : dict, default=None
            Dictionary with keyword passed to the `matplotlib.pyplot.plot`
            call to draw the optimal line.

        Returns
        -------
        display : :class:`~sklearn.metrics.PredictionErrorDisplay`

            Object that stores computed values.
        """
        check_matplotlib_support(f"{self.__class__.__name__}.plot")

        expected_kind = ("actual_vs_predicted", "residual_vs_predicted")
        if kind not in expected_kind:
            raise ValueError(
                f"`kind` must be one of {', '.join(expected_kind)}. "
                f"Got {kind!r} instead."
            )

        import matplotlib.pyplot as plt

        if scatter_kwargs is None:
            scatter_kwargs = {}
        if line_kwargs is None:
            line_kwargs = {}

        default_scatter_kwargs = {"color": "tab:blue", "alpha": 0.8}
        default_line_kwargs = {"color": "black", "alpha": 0.7, "linestyle": "--"}

        scatter_kwargs = _validate_style_kwargs(default_scatter_kwargs, scatter_kwargs)
        line_kwargs = _validate_style_kwargs(default_line_kwargs, line_kwargs)

        scatter_kwargs = {**default_scatter_kwargs, **scatter_kwargs}
        line_kwargs = {**default_line_kwargs, **line_kwargs}

        if ax is None:
            _, ax = plt.subplots()

        if kind == "actual_vs_predicted":
            max_value = max(np.max(self.y_true), np.max(self.y_pred))
            min_value = min(np.min(self.y_true), np.min(self.y_pred))
            self.line_ = ax.plot(
                [min_value, max_value], [min_value, max_value], **line_kwargs
            )[0]

            x_data, y_data = self.y_pred, self.y_true
            xlabel, ylabel = "Predicted values", "Actual values"

            self.scatter_ = ax.scatter(x_data, y_data, **scatter_kwargs)

            # force to have a squared axis
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks(np.linspace(min_value, max_value, num=5))
            ax.set_yticks(np.linspace(min_value, max_value, num=5))
        else:  # kind == "residual_vs_predicted"
            self.line_ = ax.plot(
                [np.min(self.y_pred), np.max(self.y_pred)],
                [0, 0],
                **line_kwargs,
            )[0]
            self.scatter_ = ax.scatter(
                self.y_pred, self.y_true - self.y_pred, **scatter_kwargs
            )
            xlabel, ylabel = "Predicted values", "Residuals (actual - predicted)"

        ax.set(xlabel=xlabel, ylabel=ylabel)

        self.ax_ = ax
        self.figure_ = ax.figure

        return self

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        *,
        kind="residual_vs_predicted",
        subsample=1_000,
        random_state=None,
        ax=None,
        scatter_kwargs=None,
        line_kwargs=None,
    ):
        # [FUNCTION DOC]
        # Calls estimator.predict(X) and delegates to from_predictions with all
        # remaining arguments.
        # Inputs: estimator - fitted regressor; X; y - true targets; others as in
        # from_predictions.
        # Outputs: PredictionErrorDisplay.
        # Exceptions: ImportError without matplotlib.
        # Side effects: calls predict; plots.
        """Plot the prediction error given a regressor and some data.

        For general information regarding `scikit-learn` visualization tools,
        read more in the :ref:`Visualization Guide <visualizations>`.
        For details regarding interpreting these plots, refer to the
        :ref:`Model Evaluation Guide <visualization_regression_evaluation>`.

        .. versionadded:: 1.2

        Parameters
        ----------
        estimator : estimator instance
            Fitted regressor or a fitted :class:`~sklearn.pipeline.Pipeline`
            in which the last estimator is a regressor.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input values.

        y : array-like of shape (n_samples,)
            Target values.

        kind : {"actual_vs_predicted", "residual_vs_predicted"}, \
                default="residual_vs_predicted"
            The type of plot to draw:

            - "actual_vs_predicted" draws the observed values (y-axis) vs.
              the predicted values (x-axis).
            - "residual_vs_predicted" draws the residuals, i.e. difference
              between observed and predicted values, (y-axis) vs. the predicted
              values (x-axis).

        subsample : float, int or None, default=1_000
            Sampling the samples to be shown on the scatter plot. If `float`,
            it should be between 0 and 1 and represents the proportion of the
            original dataset. If `int`, it represents the number of samples
            display on the scatter plot. If `None`, no subsampling will be
            applied. by default, 1000 samples or less will be displayed.

        random_state : int or RandomState, default=None
            Controls the randomness when `subsample` is not `None`.
            See :term:`Glossary <random_state>` for details.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        scatter_kwargs : dict, default=None
            Dictionary with keywords passed to the `matplotlib.pyplot.scatter`
            call.

        line_kwargs : dict, default=None
            Dictionary with keyword passed to the `matplotlib.pyplot.plot`
            call to draw the optimal line.

        Returns
        -------
        display : :class:`~sklearn.metrics.PredictionErrorDisplay`
            Object that stores the computed values.

        See Also
        --------
        PredictionErrorDisplay : Prediction error visualization for regression.
        PredictionErrorDisplay.from_predictions : Prediction error visualization
            given the true and predicted targets.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.metrics import PredictionErrorDisplay
        >>> X, y = load_diabetes(return_X_y=True)
        >>> ridge = Ridge().fit(X, y)
        >>> disp = PredictionErrorDisplay.from_estimator(ridge, X, y)
        >>> plt.show()
        """
        check_matplotlib_support(f"{cls.__name__}.from_estimator")

        y_pred = estimator.predict(X)

        return cls.from_predictions(
            y_true=y,
            y_pred=y_pred,
            kind=kind,
            subsample=subsample,
            random_state=random_state,
            ax=ax,
            scatter_kwargs=scatter_kwargs,
            line_kwargs=line_kwargs,
        )

    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_pred,
        *,
        kind="residual_vs_predicted",
        subsample=1_000,
        random_state=None,
        ax=None,
        scatter_kwargs=None,
        line_kwargs=None,
    ):
        # [FUNCTION DOC]
        # Optionally subsamples the points then plots. subsample: positive int = number
        #   of
        # points, float in (0, 1) = fraction of n_samples (converted with int
        #   truncation),
        # None = no subsampling. When the resulting size is smaller than n_samples,
        #   indices are
        # drawn with random_state.choice and both arrays are indexed with
        #   _safe_indexing.
        # Inputs: y_true, y_pred - (n_samples,); kind; subsample (default 1000);
        #   random_state -
        # None, int or RandomState; ax; scatter_kwargs; line_kwargs.
        # Outputs: PredictionErrorDisplay.
        # Exceptions: ImportError without matplotlib; ValueError for a non-positive int
        #   or a
        # float outside (0, 1); ValueError for invalid kind from plot.
        # Side effects: consumes randomness from random_state; plots.
        # KNOWN ISSUE: choice is called without replace=False, so the subsample is drawn
        #   with
        # replacement and may contain duplicate points while omitting others. A float
        #   subsample
        # so small that int truncation yields 0 creates an empty plot.
        """Plot the prediction error given the true and predicted targets.

        For general information regarding `scikit-learn` visualization tools,
        read more in the :ref:`Visualization Guide <visualizations>`.
        For details regarding interpreting these plots, refer to the
        :ref:`Model Evaluation Guide <visualization_regression_evaluation>`.

        .. versionadded:: 1.2

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True target values.

        y_pred : array-like of shape (n_samples,)
            Predicted target values.

        kind : {"actual_vs_predicted", "residual_vs_predicted"}, \
                default="residual_vs_predicted"
            The type of plot to draw:

            - "actual_vs_predicted" draws the observed values (y-axis) vs.
              the predicted values (x-axis).
            - "residual_vs_predicted" draws the residuals, i.e. difference
              between observed and predicted values, (y-axis) vs. the predicted
              values (x-axis).

        subsample : float, int or None, default=1_000
            Sampling the samples to be shown on the scatter plot. If `float`,
            it should be between 0 and 1 and represents the proportion of the
            original dataset. If `int`, it represents the number of samples
            display on the scatter plot. If `None`, no subsampling will be
            applied. by default, 1000 samples or less will be displayed.

        random_state : int or RandomState, default=None
            Controls the randomness when `subsample` is not `None`.
            See :term:`Glossary <random_state>` for details.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        scatter_kwargs : dict, default=None
            Dictionary with keywords passed to the `matplotlib.pyplot.scatter`
            call.

        line_kwargs : dict, default=None
            Dictionary with keyword passed to the `matplotlib.pyplot.plot`
            call to draw the optimal line.

        Returns
        -------
        display : :class:`~sklearn.metrics.PredictionErrorDisplay`
            Object that stores the computed values.

        See Also
        --------
        PredictionErrorDisplay : Prediction error visualization for regression.
        PredictionErrorDisplay.from_estimator : Prediction error visualization
            given an estimator and some data.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.metrics import PredictionErrorDisplay
        >>> X, y = load_diabetes(return_X_y=True)
        >>> ridge = Ridge().fit(X, y)
        >>> y_pred = ridge.predict(X)
        >>> disp = PredictionErrorDisplay.from_predictions(y_true=y, y_pred=y_pred)
        >>> plt.show()
        """
        check_matplotlib_support(f"{cls.__name__}.from_predictions")

        random_state = check_random_state(random_state)

        n_samples = len(y_true)
        if isinstance(subsample, numbers.Integral):
            if subsample <= 0:
                raise ValueError(
                    f"When an integer, subsample={subsample} should be positive."
                )
        elif isinstance(subsample, numbers.Real):
            if subsample <= 0 or subsample >= 1:
                raise ValueError(
                    f"When a floating-point, subsample={subsample} should"
                    " be in the (0, 1) range."
                )
            subsample = int(n_samples * subsample)

        if subsample is not None and subsample < n_samples:
            indices = random_state.choice(np.arange(n_samples), size=subsample)
            y_true = _safe_indexing(y_true, indices, axis=0)
            y_pred = _safe_indexing(y_pred, indices, axis=0)

        viz = cls(
            y_true=y_true,
            y_pred=y_pred,
        )

        return viz.plot(
            ax=ax,
            kind=kind,
            scatter_kwargs=scatter_kwargs,
            line_kwargs=line_kwargs,
        )
