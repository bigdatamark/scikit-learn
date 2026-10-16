# [MODULE DOC]
# Defines the deprecated PassiveAggressiveClassifier and PassiveAggressiveRegressor,
# which are configurations of the SGD linear-model base classes using the "pa1" / "pa2"
# adaptive step-size rules with no penalty. Both classes emit a deprecation warning on
# instantiation (via the deprecated decorator) and are scheduled for removal in 1.10.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Real

from sklearn.base import _fit_context
from sklearn.linear_model._stochastic_gradient import (
    DEFAULT_EPSILON,
    BaseSGDClassifier,
    BaseSGDRegressor,
)
from sklearn.utils import deprecated
from sklearn.utils._param_validation import Interval, StrOptions


# TODO(1.10): Remove
@deprecated(
    "this is deprecated in version 1.8 and will be removed in 1.10. "
    "Use `SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1', eta0=1.0)` "
    "instead."
)
class PassiveAggressiveClassifier(BaseSGDClassifier):
    # [CLASS DOC]
    # Passive-aggressive online linear classifier. The aggressiveness parameter C is
    # forwarded to the base class as eta0 (maximum step size); loss "hinge" selects the
    # PA-I update and "squared_hinge" selects PA-II. eta0 is removed from the public
    # parameter constraints because it is replaced by C.
    """Passive Aggressive Classifier.

    .. deprecated:: 1.8
        The whole class `PassiveAggressiveClassifier` was deprecated in version 1.8
        and will be removed in 1.10. Instead use:

        .. code-block:: python

            clf = SGDClassifier(
                loss="hinge",
                penalty=None,
                learning_rate="pa1",  # or "pa2"
                eta0=1.0,  # for parameter C
            )

    Read more in the :ref:`User Guide <passive_aggressive>`.

    Parameters
    ----------
    C : float, default=1.0
        Aggressiveness parameter for the passive-aggressive algorithm, see [1].
        For PA-I it is the maximum step size. For PA-II it regularizes the
        step size (the smaller `C` the more it regularizes).
        As a general rule-of-thumb, `C` should be small when the data is noisy.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        :meth:`~sklearn.linear_model.PassiveAggressiveClassifier.partial_fit` method.

        .. versionadded:: 0.19

    tol : float or None, default=1e-3
        The stopping criterion. If it is not None, the iterations will stop
        when (loss > previous_loss - tol).

        .. versionadded:: 0.19

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to True, it will automatically set aside
        a stratified fraction of training data as validation and terminate
        training when validation score is not improving by at least `tol` for
        `n_iter_no_change` consecutive epochs.

        .. versionadded:: 0.20

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True.

        .. versionadded:: 0.20

    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before early stopping.

        .. versionadded:: 0.20

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.

    verbose : int, default=0
        The verbosity level.

    loss : str, default="hinge"
        The loss function to be used:
        hinge: equivalent to PA-I in the reference paper.
        squared_hinge: equivalent to PA-II in the reference paper.

    n_jobs : int or None, default=None
        The number of CPUs to use to do the OVA (One Versus All, for
        multi-class problems) computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance, default=None
        Used to shuffle the training data, when ``shuffle`` is set to
        ``True``. Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

        Repeatedly calling fit or partial_fit when warm_start is True can
        result in a different solution than when calling fit a single time
        because of the way the data is shuffled.

    class_weight : dict, {class_label: weight} or "balanced" or None, \
            default=None
        Preset for the class_weight fit parameter.

        Weights associated with classes. If not given, all classes
        are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        .. versionadded:: 0.17
           parameter *class_weight* to automatically weight samples.

    average : bool or int, default=False
        When set to True, computes the averaged SGD weights and stores the
        result in the ``coef_`` attribute. If set to an int greater than 1,
        averaging will begin once the total number of samples seen reaches
        average. So average=10 will begin averaging after seeing 10 samples.

        .. versionadded:: 0.19
           parameter *average* to use weights averaging in SGD.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features) if n_classes == 2 else \
            (n_classes, n_features)
        Weights assigned to the features.

    intercept_ : ndarray of shape (1,) if n_classes == 2 else (n_classes,)
        Constants in decision function.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.
        For multiclass fits, it is the maximum over every binary fit.

    classes_ : ndarray of shape (n_classes,)
        The unique classes labels.

    t_ : int
        Number of weight updates performed during training.
        Same as ``(n_iter_ * n_samples + 1)``.

    See Also
    --------
    SGDClassifier : Incrementally trained logistic regression.
    Perceptron : Linear perceptron classifier.

    References
    ----------
    .. [1] Online Passive-Aggressive Algorithms
       <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>
       K. Crammer, O. Dekel, J. Keshat, S. Shalev-Shwartz, Y. Singer - JMLR (2006)

    Examples
    --------
    >>> from sklearn.linear_model import PassiveAggressiveClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_features=4, random_state=0)
    >>> clf = PassiveAggressiveClassifier(max_iter=1000, random_state=0,
    ... tol=1e-3)
    >>> clf.fit(X, y)
    PassiveAggressiveClassifier(random_state=0)
    >>> print(clf.coef_)
    [[0.26642044 0.45070924 0.67251877 0.64185414]]
    >>> print(clf.intercept_)
    [1.84127814]
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]
    """

    _parameter_constraints: dict = {
        **BaseSGDClassifier._parameter_constraints,
        "loss": [StrOptions({"hinge", "squared_hinge"})],
        "C": [Interval(Real, 0, None, closed="right")],
    }
    _parameter_constraints.pop("eta0")

    def __init__(
        self,
        *,
        C=1.0,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        shuffle=True,
        verbose=0,
        loss="hinge",
        n_jobs=None,
        random_state=None,
        warm_start=False,
        class_weight=None,
        average=False,
    ):
        # [FUNCTION DOC]
        # Stores hyper-parameters. Forwards everything to BaseSGDClassifier with penalty=None
        # and eta0=C, then additionally stores C and loss on self.
        # Inputs: see class parameters. Outputs: none.
        # Side effects: sets attributes on self; the class-level deprecated decorator emits a
        # FutureWarning when the object is constructed.
        super().__init__(
            penalty=None,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            shuffle=shuffle,
            verbose=verbose,
            random_state=random_state,
            eta0=C,
            warm_start=warm_start,
            class_weight=class_weight,
            average=average,
            n_jobs=n_jobs,
        )

        self.C = C
        self.loss = loss

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, classes=None):
        # [FUNCTION DOC]
        # Performs one epoch of passive-aggressive updates over the given mini-batch.
        # On the first call (no classes_ yet) runs the extra parameter validation for partial
        # fitting and rejects class_weight="balanced". Chooses learning rate "pa1" when
        # loss is "hinge" otherwise "pa2", then delegates to _partial_fit with alpha=1.0,
        # loss forced to "hinge" (the squared variant is realised through "pa2"), max_iter=1,
        # no sample weights and no initial coefficients.
        # Inputs: X (n_samples, n_features) dense or sparse; y (n_samples,); classes - array of
        # all class labels, required on the first call.
        # Outputs: self.
        # Exceptions: ValueError if class_weight == "balanced" on the first call; errors from
        # _partial_fit (e.g. missing classes on the first call).
        # Side effects: creates/updates coef_, intercept_, classes_, t_ and related attributes.
        """Fit linear model with Passive Aggressive algorithm.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Subset of the training data.

        y : array-like of shape (n_samples,)
            Subset of the target values.

        classes : ndarray of shape (n_classes,)
            Classes across all calls to partial_fit.
            Can be obtained by via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if not hasattr(self, "classes_"):
            self._more_validate_params(for_partial_fit=True)

            if self.class_weight == "balanced":
                raise ValueError(
                    "class_weight 'balanced' is not supported for "
                    "partial_fit. For 'balanced' weights, use "
                    "`sklearn.utils.compute_class_weight` with "
                    "`class_weight='balanced'`. In place of y you "
                    "can use a large enough subset of the full "
                    "training set target to properly estimate the "
                    "class frequency distributions. Pass the "
                    "resulting weights as the class_weight "
                    "parameter."
                )

        # For an explanation, see
        # https://github.com/scikit-learn/scikit-learn/pull/1259#issuecomment-9818044
        lr = "pa1" if self.loss == "hinge" else "pa2"
        return self._partial_fit(
            X,
            y,
            alpha=1.0,
            loss="hinge",
            learning_rate=lr,
            max_iter=1,
            classes=classes,
            sample_weight=None,
            coef_init=None,
            intercept_init=None,
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, coef_init=None, intercept_init=None):
        # [FUNCTION DOC]
        # Fits the model from scratch (or warm-started) over up to max_iter epochs.
        # Validates extra parameters, selects "pa1" for hinge loss else "pa2", and delegates to
        # _fit with alpha=1.0, loss forced to "hinge", and the optional initial coefficients.
        # Inputs: X (n_samples, n_features); y (n_samples,); coef_init (n_classes, n_features)
        # or None; intercept_init (n_classes,) or None.
        # Outputs: self.
        # Side effects: sets fitted attributes (coef_, intercept_, n_iter_, t_, classes_, ...).
        """Fit linear model with Passive Aggressive algorithm.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        coef_init : ndarray of shape (n_classes, n_features)
            The initial coefficients to warm-start the optimization.

        intercept_init : ndarray of shape (n_classes,)
            The initial intercept to warm-start the optimization.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._more_validate_params()

        lr = "pa1" if self.loss == "hinge" else "pa2"
        return self._fit(
            X,
            y,
            alpha=1.0,
            loss="hinge",
            learning_rate=lr,
            coef_init=coef_init,
            intercept_init=intercept_init,
        )


# TODO(1.10): Remove
@deprecated(
    "this is deprecated in version 1.8 and will be removed in 1.10. "
    "Use `SGDRegressor(loss='epsilon_insensitive', penalty=None, learning_rate='pa1', "
    "eta0 = 1.0)` instead."
)
class PassiveAggressiveRegressor(BaseSGDRegressor):
    # [CLASS DOC]
    # Passive-aggressive online linear regressor. C is forwarded as eta0;
    # loss "epsilon_insensitive" selects PA-I and "squared_epsilon_insensitive" selects
    # PA-II. epsilon sets the insensitive-zone width.
    """Passive Aggressive Regressor.

    .. deprecated:: 1.8
        The whole class `PassiveAggressiveRegressor` was deprecated in version 1.8
        and will be removed in 1.10. Instead use:

        .. code-block:: python

            reg = SGDRegressor(
                loss="epsilon_insensitive",
                penalty=None,
                learning_rate="pa1",  # or "pa2"
                eta0=1.0,  # for parameter C
            )

    Read more in the :ref:`User Guide <passive_aggressive>`.

    Parameters
    ----------

    C : float, default=1.0
        Aggressiveness parameter for the passive-aggressive algorithm, see [1].
        For PA-I it is the maximum step size. For PA-II it regularizes the
        step size (the smaller `C` the more it regularizes).
        As a general rule-of-thumb, `C` should be small when the data is noisy.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered. Defaults to True.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        :meth:`~sklearn.linear_model.PassiveAggressiveRegressor.partial_fit` method.

        .. versionadded:: 0.19

    tol : float or None, default=1e-3
        The stopping criterion. If it is not None, the iterations will stop
        when (loss > previous_loss - tol).

        .. versionadded:: 0.19

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation.
        score is not improving. If set to True, it will automatically set aside
        a fraction of training data as validation and terminate
        training when validation score is not improving by at least tol for
        n_iter_no_change consecutive epochs.

        .. versionadded:: 0.20

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True.

        .. versionadded:: 0.20

    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before early stopping.

        .. versionadded:: 0.20

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.

    verbose : int, default=0
        The verbosity level.

    loss : str, default="epsilon_insensitive"
        The loss function to be used:
        epsilon_insensitive: equivalent to PA-I in the reference paper.
        squared_epsilon_insensitive: equivalent to PA-II in the reference
        paper.

    epsilon : float, default=0.1
        If the difference between the current prediction and the correct label
        is below this threshold, the model is not updated.

    random_state : int, RandomState instance, default=None
        Used to shuffle the training data, when ``shuffle`` is set to
        ``True``. Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

        Repeatedly calling fit or partial_fit when warm_start is True can
        result in a different solution than when calling fit a single time
        because of the way the data is shuffled.

    average : bool or int, default=False
        When set to True, computes the averaged SGD weights and stores the
        result in the ``coef_`` attribute. If set to an int greater than 1,
        averaging will begin once the total number of samples seen reaches
        average. So average=10 will begin averaging after seeing 10 samples.

        .. versionadded:: 0.19
           parameter *average* to use weights averaging in SGD.

    Attributes
    ----------
    coef_ : array, shape = [1, n_features] if n_classes == 2 else [n_classes,\
            n_features]
        Weights assigned to the features.

    intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
        Constants in decision function.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.

    t_ : int
        Number of weight updates performed during training.
        Same as ``(n_iter_ * n_samples + 1)``.

    See Also
    --------
    SGDRegressor : Linear model fitted by minimizing a regularized
        empirical loss with SGD.

    References
    ----------
    Online Passive-Aggressive Algorithms
    <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>
    K. Crammer, O. Dekel, J. Keshat, S. Shalev-Shwartz, Y. Singer - JMLR (2006).

    Examples
    --------
    >>> from sklearn.linear_model import PassiveAggressiveRegressor
    >>> from sklearn.datasets import make_regression

    >>> X, y = make_regression(n_features=4, random_state=0)
    >>> regr = PassiveAggressiveRegressor(max_iter=100, random_state=0,
    ... tol=1e-3)
    >>> regr.fit(X, y)
    PassiveAggressiveRegressor(max_iter=100, random_state=0)
    >>> print(regr.coef_)
    [20.48736655 34.18818427 67.59122734 87.94731329]
    >>> print(regr.intercept_)
    [-0.02306214]
    >>> print(regr.predict([[0, 0, 0, 0]]))
    [-0.02306214]
    """

    _parameter_constraints: dict = {
        **BaseSGDRegressor._parameter_constraints,
        "loss": [StrOptions({"epsilon_insensitive", "squared_epsilon_insensitive"})],
        "C": [Interval(Real, 0, None, closed="right")],
        "epsilon": [Interval(Real, 0, None, closed="left")],
    }
    _parameter_constraints.pop("eta0")

    def __init__(
        self,
        *,
        C=1.0,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        shuffle=True,
        verbose=0,
        loss="epsilon_insensitive",
        epsilon=DEFAULT_EPSILON,
        random_state=None,
        warm_start=False,
        average=False,
    ):
        # [FUNCTION DOC]
        # Stores hyper-parameters, forwarding them to BaseSGDRegressor with penalty=None,
        # l1_ratio=0 and eta0=C, then stores C on self.
        # Inputs: see class parameters. Outputs: none.
        # Side effects: sets attributes on self; deprecation FutureWarning on construction.
        super().__init__(
            loss=loss,
            penalty=None,
            l1_ratio=0,
            epsilon=epsilon,
            eta0=C,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            shuffle=shuffle,
            verbose=verbose,
            random_state=random_state,
            warm_start=warm_start,
            average=average,
        )
        self.C = C

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y):
        # [FUNCTION DOC]
        # Performs one epoch of passive-aggressive regression updates on a mini-batch.
        # Runs partial-fit parameter validation on the first call (no coef_ yet), selects
        # "pa1" for "epsilon_insensitive" else "pa2", and delegates to _partial_fit with
        # alpha=1.0, loss forced to "epsilon_insensitive", max_iter=1, no sample weights.
        # Inputs: X (n_samples, n_features); y (n_samples,).
        # Outputs: self.
        # Side effects: creates/updates coef_, intercept_, t_ and averaging attributes.
        """Fit linear model with Passive Aggressive algorithm.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Subset of training data.

        y : numpy array of shape [n_samples]
            Subset of target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if not hasattr(self, "coef_"):
            self._more_validate_params(for_partial_fit=True)

        lr = "pa1" if self.loss == "epsilon_insensitive" else "pa2"
        return self._partial_fit(
            X,
            y,
            alpha=1.0,
            loss="epsilon_insensitive",
            learning_rate=lr,
            max_iter=1,
            sample_weight=None,
            coef_init=None,
            intercept_init=None,
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, coef_init=None, intercept_init=None):
        # [FUNCTION DOC]
        # Fits the regressor over up to max_iter epochs. Validates extra parameters, selects
        # "pa1"/"pa2" from loss, and delegates to _fit with alpha=1.0, loss forced to
        # "epsilon_insensitive", and optional warm-start coefficients.
        # Inputs: X (n_samples, n_features); y (n_samples,); coef_init (n_features,) or None;
        # intercept_init (1,) or None.
        # Outputs: self.
        # Side effects: sets fitted attributes (coef_, intercept_, n_iter_, t_).
        """Fit linear model with Passive Aggressive algorithm.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : numpy array of shape [n_samples]
            Target values.

        coef_init : array, shape = [n_features]
            The initial coefficients to warm-start the optimization.

        intercept_init : array, shape = [1]
            The initial intercept to warm-start the optimization.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._more_validate_params()

        lr = "pa1" if self.loss == "epsilon_insensitive" else "pa2"
        return self._fit(
            X,
            y,
            alpha=1.0,
            loss="epsilon_insensitive",
            learning_rate=lr,
            coef_init=coef_init,
            intercept_init=intercept_init,
        )
