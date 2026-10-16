# [MODULE DOC]
# Implements LatentDirichletAllocation, a topic model fitted by batch or online (mini-
# batch, Hoffman et al.) variational Bayes. Contains the per-document E-step helper
# (_update_doc_distribution, run in parallel over document slices via joblib) and the
# estimator with fit/partial_fit/transform plus variational-bound based score and
# perplexity. Cython helpers from _online_lda_fast compute Dirichlet expectations and
# mean absolute change.
# Module constant EPS: float64 machine epsilon (unused in this file). Side effects at
# import: none.
"""

=============================================================
Online Latent Dirichlet Allocation with variational inference
=============================================================

This implementation is modified from Matthew D. Hoffman's onlineldavb code
Link: https://github.com/blei-lab/onlineldavb
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Integral, Real

import numpy as np
import scipy.sparse as sp
from joblib import effective_n_jobs
from scipy.special import gammaln, logsumexp

from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.decomposition._online_lda_fast import (
    _dirichlet_expectation_1d as cy_dirichlet_expectation_1d,
)
from sklearn.decomposition._online_lda_fast import _dirichlet_expectation_2d
from sklearn.decomposition._online_lda_fast import mean_change as cy_mean_change
from sklearn.utils import check_random_state, gen_batches, gen_even_slices
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted, check_non_negative, validate_data

EPS = np.finfo(float).eps


def _update_doc_distribution(
    X,
    exp_topic_word_distr,
    doc_topic_prior,
    max_doc_update_iter,
    mean_change_tol,
    cal_sstats,
    random_state,
):
    # [FUNCTION DOC]
    # HELPER: Variational E-step for a block of documents. Initialises gamma
    #   (doc_topic_distr) from Gamma(100, 0.01) draws when random_state is given, else
    #   ones; computes exp(E[log theta]). For each document, takes its non-zero word ids
    #   and counts, then iterates up to max_doc_update_iter times: normaliser phi_norm =
    #   exp(E[log theta_d]) dotted with the topic-word columns of those words plus eps;
    #   new gamma_d = exp(E[log theta_d]) times (counts / phi_norm) dotted with those
    #   columns transposed; the Cython helper adds doc_topic_prior to gamma_d and
    #   refreshes exp(E[log theta_d]) in place; stop when the mean absolute change is
    #   below mean_change_tol. Stores gamma_d, and when cal_sstats accumulates the outer
    #   product of exp(E[log theta_d]) and counts/phi_norm into the word columns of the
    #   sufficient statistics.
    # Inputs: X (n_samples, n_features) dense or CSR counts; exp_topic_word_distr
    #   (n_topics, n_features); doc_topic_prior float; max_doc_update_iter int;
    #   mean_change_tol float; cal_sstats bool; random_state RandomState or None.
    # Outputs: tuple (doc_topic_distr (n_samples, n_topics) unnormalised gamma,
    #   suff_stats (n_topics, n_features) not yet multiplied by exp_topic_word_distr, or
    #   None).
    # Side effects: consumes random_state when given.
    """E-step: update document-topic distribution.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Document word matrix.

    exp_topic_word_distr : ndarray of shape (n_topics, n_features)
        Exponential value of expectation of log topic word distribution.
        In the literature, this is `exp(E[log(beta)])`.

    doc_topic_prior : float
        Prior of document topic distribution `theta`.

    max_doc_update_iter : int
        Max number of iterations for updating document topic distribution in
        the E-step.

    mean_change_tol : float
        Stopping tolerance for updating document topic distribution in E-step.

    cal_sstats : bool
        Parameter that indicate to calculate sufficient statistics or not.
        Set `cal_sstats` to `True` when we need to run M-step.

    random_state : RandomState instance or None
        Parameter that indicate how to initialize document topic distribution.
        Set `random_state` to None will initialize document topic distribution
        to a constant number.

    Returns
    -------
    (doc_topic_distr, suff_stats) :
        `doc_topic_distr` is unnormalized topic distribution for each document.
        In the literature, this is `gamma`. we can calculate `E[log(theta)]`
        from it.
        `suff_stats` is expected sufficient statistics for the M-step.
            When `cal_sstats == False`, this will be None.

    """
    is_sparse_x = sp.issparse(X)
    n_samples, n_features = X.shape
    n_topics = exp_topic_word_distr.shape[0]

    if random_state:
        doc_topic_distr = random_state.gamma(100.0, 0.01, (n_samples, n_topics)).astype(
            X.dtype, copy=False
        )
    else:
        doc_topic_distr = np.ones((n_samples, n_topics), dtype=X.dtype)

    # In the literature, this is `exp(E[log(theta)])`
    exp_doc_topic = np.exp(_dirichlet_expectation_2d(doc_topic_distr))

    # diff on `component_` (only calculate it when `cal_diff` is True)
    suff_stats = (
        np.zeros(exp_topic_word_distr.shape, dtype=X.dtype) if cal_sstats else None
    )

    if is_sparse_x:
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

    # These cython functions are called in a nested loop on usually very small arrays
    # (length=n_topics). In that case, finding the appropriate signature of the
    # fused-typed function can be more costly than its execution, hence the dispatch
    # is done outside of the loop.
    ctype = "float" if X.dtype == np.float32 else "double"
    mean_change = cy_mean_change[ctype]
    dirichlet_expectation_1d = cy_dirichlet_expectation_1d[ctype]
    eps = np.finfo(X.dtype).eps

    for idx_d in range(n_samples):
        if is_sparse_x:
            ids = X_indices[X_indptr[idx_d] : X_indptr[idx_d + 1]]
            cnts = X_data[X_indptr[idx_d] : X_indptr[idx_d + 1]]
        else:
            ids = np.nonzero(X[idx_d, :])[0]
            cnts = X[idx_d, ids]

        doc_topic_d = doc_topic_distr[idx_d, :]
        # The next one is a copy, since the inner loop overwrites it.
        exp_doc_topic_d = exp_doc_topic[idx_d, :].copy()
        exp_topic_word_d = exp_topic_word_distr[:, ids]

        # Iterate between `doc_topic_d` and `norm_phi` until convergence
        for _ in range(0, max_doc_update_iter):
            last_d = doc_topic_d

            # The optimal phi_{dwk} is proportional to
            # exp(E[log(theta_{dk})]) * exp(E[log(beta_{dw})]).
            norm_phi = np.dot(exp_doc_topic_d, exp_topic_word_d) + eps

            doc_topic_d = exp_doc_topic_d * np.dot(cnts / norm_phi, exp_topic_word_d.T)
            # Note: adds doc_topic_prior to doc_topic_d, in-place.
            dirichlet_expectation_1d(doc_topic_d, doc_topic_prior, exp_doc_topic_d)

            if mean_change(last_d, doc_topic_d) < mean_change_tol:
                break
        doc_topic_distr[idx_d, :] = doc_topic_d

        # Contribution of document d to the expected sufficient
        # statistics for the M step.
        if cal_sstats:
            norm_phi = np.dot(exp_doc_topic_d, exp_topic_word_d) + eps
            suff_stats[:, ids] += np.outer(exp_doc_topic_d, cnts / norm_phi)

    return (doc_topic_distr, suff_stats)


class LatentDirichletAllocation(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    # [CLASS DOC]
    # Topic model estimator. Hyperparameters set in __init__; fitting sets
    # random_state_, components_ (variational topic-word parameters lambda),
    # exp_dirichlet_component_, doc_topic_prior_, topic_word_prior_, n_batch_iter_,
    # n_iter_, bound_ (final training perplexity, fit only). Input must be non-negative;
    # float32/float64 preserved; sparse accepted.
    """Latent Dirichlet Allocation with online variational Bayes algorithm.

    The implementation is based on [1]_ and [2]_.

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <LatentDirichletAllocation>`.

    Parameters
    ----------
    n_components : int, default=10
        Number of topics.

        .. versionchanged:: 0.19
            ``n_topics`` was renamed to ``n_components``

    doc_topic_prior : float, default=None
        Prior of document topic distribution `theta`. If the value is None,
        defaults to `1 / n_components`.
        In [1]_, this is called `alpha`.

    topic_word_prior : float, default=None
        Prior of topic word distribution `beta`. If the value is None, defaults
        to `1 / n_components`.
        In [1]_, this is called `eta`.

    learning_method : {'batch', 'online'}, default='batch'
        Method used to update `_component`. Only used in :meth:`fit` method.
        In general, if the data size is large, the online update will be much
        faster than the batch update.

        Valid options:

        - 'batch': Batch variational Bayes method. Use all training data in each EM
          update. Old `components_` will be overwritten in each iteration.
        - 'online': Online variational Bayes method. In each EM update, use mini-batch
          of training data to update the ``components_`` variable incrementally. The
          learning rate is controlled by the ``learning_decay`` and the
          ``learning_offset`` parameters.

        .. versionchanged:: 0.20
            The default learning method is now ``"batch"``.

    learning_decay : float, default=0.7
        It is a parameter that control learning rate in the online learning
        method. The value should be set between (0.5, 1.0] to guarantee
        asymptotic convergence. When the value is 0.0 and batch_size is
        ``n_samples``, the update method is same as batch learning. In the
        literature, this is called kappa.

    learning_offset : float, default=10.0
        A (positive) parameter that downweights early iterations in online
        learning.  It should be greater than 1.0. In the literature, this is
        called tau_0.

    max_iter : int, default=10
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the :meth:`fit` method, and not the
        :meth:`partial_fit` method.

    batch_size : int, default=128
        Number of documents to use in each EM iteration. Only used in online
        learning.

    evaluate_every : int, default=-1
        How often to evaluate perplexity. Only used in `fit` method.
        set it to 0 or negative number to not evaluate perplexity in
        training at all. Evaluating perplexity can help you check convergence
        in training process, but it will also increase total training time.
        Evaluating perplexity in every iteration might increase training time
        up to two-fold.

    total_samples : int, default=1e6
        Total number of documents. Only used in the :meth:`partial_fit` method.

    perp_tol : float, default=1e-1
        Perplexity tolerance. Only used when ``evaluate_every`` is greater than 0.

    mean_change_tol : float, default=1e-3
        Stopping tolerance for updating document topic distribution in E-step.

    max_doc_update_iter : int, default=100
        Max number of iterations for updating document topic distribution in
        the E-step.

    n_jobs : int, default=None
        The number of jobs to use in the E-step.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        Verbosity level.

    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Variational parameters for topic word distribution. Since the complete
        conditional for topic word distribution is a Dirichlet,
        ``components_[i, j]`` can be viewed as pseudocount that represents the
        number of times word `j` was assigned to topic `i`.
        It can also be viewed as distribution over the words for each topic
        after normalization:
        ``model.components_ / model.components_.sum(axis=1)[:, np.newaxis]``.

    exp_dirichlet_component_ : ndarray of shape (n_components, n_features)
        Exponential value of expectation of log topic word distribution.
        In the literature, this is `exp(E[log(beta)])`.

    n_batch_iter_ : int
        Number of iterations of the EM step.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Number of passes over the dataset.

    bound_ : float
        Final perplexity score on training set.

    doc_topic_prior_ : float
        Prior of document topic distribution `theta`. If the value is None,
        it is `1 / n_components`.

    random_state_ : RandomState instance
        RandomState instance that is generated either from a seed, the random
        number generator or by `np.random`.

    topic_word_prior_ : float
        Prior of topic word distribution `beta`. If the value is None, it is
        `1 / n_components`.

    See Also
    --------
    sklearn.discriminant_analysis.LinearDiscriminantAnalysis:
        A classifier with a linear decision boundary, generated by fitting
        class conditional densities to the data and using Bayes' rule.

    References
    ----------
    .. [1] "Online Learning for Latent Dirichlet Allocation", Matthew D.
           Hoffman, David M. Blei, Francis Bach, 2010.
           https://github.com/blei-lab/onlineldavb

    .. [2] "Stochastic Variational Inference", Matthew D. Hoffman,
           David M. Blei, Chong Wang, John Paisley, 2013.
           https://jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf

    Examples
    --------
    >>> from sklearn.decomposition import LatentDirichletAllocation
    >>> from sklearn.datasets import make_multilabel_classification
    >>> # This produces a feature matrix of token counts, similar to what
    >>> # CountVectorizer would produce on text.
    >>> X, _ = make_multilabel_classification(random_state=0)
    >>> lda = LatentDirichletAllocation(n_components=5,
    ...     random_state=0)
    >>> lda.fit(X)
    LatentDirichletAllocation(...)
    >>> # get topics for some given samples:
    >>> lda.transform(X[-2:])
    array([[0.00360392, 0.25499205, 0.0036211 , 0.64236448, 0.09541846],
           [0.15297572, 0.00362644, 0.44412786, 0.39568399, 0.003586  ]])
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 0, None, closed="neither")],
        "doc_topic_prior": [None, Interval(Real, 0, 1, closed="both")],
        "topic_word_prior": [None, Interval(Real, 0, 1, closed="both")],
        "learning_method": [StrOptions({"batch", "online"})],
        "learning_decay": [Interval(Real, 0, 1, closed="both")],
        "learning_offset": [Interval(Real, 1.0, None, closed="left")],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "batch_size": [Interval(Integral, 0, None, closed="neither")],
        "evaluate_every": [Interval(Integral, None, None, closed="neither")],
        "total_samples": [Interval(Real, 0, None, closed="neither")],
        "perp_tol": [Interval(Real, 0, None, closed="left")],
        "mean_change_tol": [Interval(Real, 0, None, closed="left")],
        "max_doc_update_iter": [Interval(Integral, 0, None, closed="left")],
        "n_jobs": [None, Integral],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_components=10,
        *,
        doc_topic_prior=None,
        topic_word_prior=None,
        learning_method="batch",
        learning_decay=0.7,
        learning_offset=10.0,
        max_iter=10,
        batch_size=128,
        evaluate_every=-1,
        total_samples=1e6,
        perp_tol=1e-1,
        mean_change_tol=1e-3,
        max_doc_update_iter=100,
        n_jobs=None,
        verbose=0,
        random_state=None,
    ):
        # [FUNCTION DOC]
        # Stores hyperparameters as attributes without validation.
        # Inputs: n_components (topics, default 10), doc_topic_prior and
        #   topic_word_prior (float or None = 1/n_components), learning_method ('batch'
        #   or 'online'), learning_decay (kappa), learning_offset (tau0), max_iter,
        #   batch_size, evaluate_every (perplexity check period, <=0 disables),
        #   total_samples (corpus size for partial_fit), perp_tol, mean_change_tol,
        #   max_doc_update_iter, n_jobs, verbose, random_state.
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.learning_method = learning_method
        self.learning_decay = learning_decay
        self.learning_offset = learning_offset
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.total_samples = total_samples
        self.perp_tol = perp_tol
        self.mean_change_tol = mean_change_tol
        self.max_doc_update_iter = max_doc_update_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _init_latent_vars(self, n_features, dtype=np.float64):
        # [FUNCTION DOC]
        # HELPER: Initialises model state: random_state_, n_batch_iter_=1, n_iter_=0,
        #   the resolved priors, components_ drawn from Gamma(shape 100, scale 0.01) of
        #   shape (n_components, n_features) cast to dtype, and exp_dirichlet_component_
        #   = exp(E[log beta]).
        # Inputs: n_features int; dtype numpy dtype. Outputs: none.
        # Side effects: sets the listed attributes; consumes randomness.
        """Initialize latent variables."""

        self.random_state_ = check_random_state(self.random_state)
        self.n_batch_iter_ = 1
        self.n_iter_ = 0

        if self.doc_topic_prior is None:
            self.doc_topic_prior_ = 1.0 / self.n_components
        else:
            self.doc_topic_prior_ = self.doc_topic_prior

        if self.topic_word_prior is None:
            self.topic_word_prior_ = 1.0 / self.n_components
        else:
            self.topic_word_prior_ = self.topic_word_prior

        init_gamma = 100.0
        init_var = 1.0 / init_gamma
        # In the literature, this is called `lambda`
        self.components_ = self.random_state_.gamma(
            init_gamma, init_var, (self.n_components, n_features)
        ).astype(dtype, copy=False)

        # In the literature, this is `exp(E[log(beta)])`
        self.exp_dirichlet_component_ = np.exp(
            _dirichlet_expectation_2d(self.components_)
        )

    def _e_step(self, X, cal_sstats, random_init, parallel=None):
        # [FUNCTION DOC]
        # HELPER: Runs _update_doc_distribution in parallel over n_jobs even row slices
        #   of X, stacks the per-slice gamma matrices, and when cal_sstats sums the per-
        #   slice statistics and multiplies them elementwise by exp_dirichlet_component_
        #   to finish the M-step statistics.
        # Inputs: X (n_samples, n_features); cal_sstats bool; random_init bool (use
        #   random_state_ for gamma initialisation); parallel optional joblib Parallel
        #   (created if None).
        # Outputs: tuple (doc_topic_distr (n_samples, n_components), suff_stats
        #   (n_components, n_features) or None).
        # Side effects: consumes random_state_ when random_init.
        """E-step in EM update.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        cal_sstats : bool
            Parameter that indicate whether to calculate sufficient statistics
            or not. Set ``cal_sstats`` to True when we need to run M-step.

        random_init : bool
            Parameter that indicate whether to initialize document topic
            distribution randomly in the E-step. Set it to True in training
            steps.

        parallel : joblib.Parallel, default=None
            Pre-initialized instance of joblib.Parallel.

        Returns
        -------
        (doc_topic_distr, suff_stats) :
            `doc_topic_distr` is unnormalized topic distribution for each
            document. In the literature, this is called `gamma`.
            `suff_stats` is expected sufficient statistics for the M-step.
            When `cal_sstats == False`, it will be None.

        """

        # Run e-step in parallel
        random_state = self.random_state_ if random_init else None

        # TODO: make Parallel._effective_n_jobs public instead?
        n_jobs = effective_n_jobs(self.n_jobs)
        if parallel is None:
            parallel = Parallel(n_jobs=n_jobs, verbose=max(0, self.verbose - 1))
        results = parallel(
            delayed(_update_doc_distribution)(
                X[idx_slice, :],
                self.exp_dirichlet_component_,
                self.doc_topic_prior_,
                self.max_doc_update_iter,
                self.mean_change_tol,
                cal_sstats,
                random_state,
            )
            for idx_slice in gen_even_slices(X.shape[0], n_jobs)
        )

        # merge result
        doc_topics, sstats_list = zip(*results)
        doc_topic_distr = np.vstack(doc_topics)

        if cal_sstats:
            # This step finishes computing the sufficient statistics for the
            # M-step.
            suff_stats = np.zeros(self.components_.shape, dtype=self.components_.dtype)
            for sstats in sstats_list:
                suff_stats += sstats
            suff_stats *= self.exp_dirichlet_component_
        else:
            suff_stats = None

        return (doc_topic_distr, suff_stats)

    def _em_step(self, X, total_samples, batch_update, parallel=None):
        # [FUNCTION DOC]
        # HELPER: One variational EM step. Runs _e_step with sufficient statistics and
        #   random init; then for batch updates sets components_ to topic_word_prior_
        #   plus statistics; for online updates computes weight rho = (learning_offset +
        #   n_batch_iter_)^(-learning_decay) and sets components_ to (1 - rho) times old
        #   plus rho times (topic_word_prior_ + (total_samples / batch size) times
        #   statistics). Refreshes exp_dirichlet_component_ and increments
        #   n_batch_iter_.
        # Inputs: X batch; total_samples (used only for online); batch_update bool;
        #   parallel optional.
        # Outputs: None.
        # Side effects: mutates components_ (in place for online),
        #   exp_dirichlet_component_, n_batch_iter_.
        # KNOWN ISSUE: the docstring claims a doc_topic_distr return value but the
        #   function returns None.
        """EM update for 1 iteration.

        update `component_` by batch VB or online VB.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        total_samples : int
            Total number of documents. It is only used when
            batch_update is `False`.

        batch_update : bool
            Parameter that controls updating method.
            `True` for batch learning, `False` for online learning.

        parallel : joblib.Parallel, default=None
            Pre-initialized instance of joblib.Parallel

        Returns
        -------
        doc_topic_distr : ndarray of shape (n_samples, n_components)
            Unnormalized document topic distribution.
        """

        # E-step
        _, suff_stats = self._e_step(
            X, cal_sstats=True, random_init=True, parallel=parallel
        )

        # M-step
        if batch_update:
            self.components_ = self.topic_word_prior_ + suff_stats
        else:
            # online update
            # In the literature, the weight is `rho`
            weight = np.power(
                self.learning_offset + self.n_batch_iter_, -self.learning_decay
            )
            doc_ratio = float(total_samples) / X.shape[0]
            self.components_ *= 1 - weight
            self.components_ += weight * (
                self.topic_word_prior_ + doc_ratio * suff_stats
            )

        # update `component_` related variables
        self.exp_dirichlet_component_ = np.exp(
            _dirichlet_expectation_2d(self.components_)
        )
        self.n_batch_iter_ += 1
        return

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Extends parent tags: non-negative input only, sparse accepted, float32/float64
        # preserved.
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = True
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ["float32", "float64"]
        return tags

    def _check_non_neg_array(self, X, reset_n_features, whom):
        # [FUNCTION DOC]
        # HELPER: Validates X (CSR accepted; dtype float64/float32 when resetting,
        #   otherwise the fitted components_ dtype) and checks it has no negative
        #   values.
        # Inputs: X; reset_n_features bool (reset n_features_in_); whom str used in the
        #   error message.
        # Outputs: validated X.
        # Exceptions: ValueError on negative values or validation failure.
        # Side effects: may set n_features_in_/feature_names_in_.
        """check X format

        check X format and make sure no negative value in X.

        Parameters
        ----------
        X :  array-like or sparse matrix

        """
        dtype = [np.float64, np.float32] if reset_n_features else self.components_.dtype

        X = validate_data(
            self,
            X,
            reset=reset_n_features,
            accept_sparse="csr",
            dtype=dtype,
        )
        check_non_negative(X, whom)

        return X

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None):
        # [FUNCTION DOC]
        # Online update on one chunk: validates X (resetting feature info only on the
        # first call), initialises latent variables on the first call, checks the
        # feature count, then runs an online _em_step for each consecutive batch of
        # batch_size rows with total_samples as the corpus size.
        # Inputs: X (n_samples, n_features); y ignored. Outputs: self.
        # Exceptions: ValueError for negative values or a feature-count mismatch.
        # Side effects: updates components_, exp_dirichlet_component_, n_batch_iter_;
        #   consumes randomness.
        """Online VB with Mini-Batch update.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Partially fitted estimator.
        """
        first_time = not hasattr(self, "components_")

        X = self._check_non_neg_array(
            X, reset_n_features=first_time, whom="LatentDirichletAllocation.partial_fit"
        )
        n_samples, n_features = X.shape
        batch_size = self.batch_size

        # initialize parameters or check
        if first_time:
            self._init_latent_vars(n_features, dtype=X.dtype)

        if n_features != self.components_.shape[1]:
            raise ValueError(
                "The provided data has %d dimensions while "
                "the model was trained with feature size %d."
                % (n_features, self.components_.shape[1])
            )

        n_jobs = effective_n_jobs(self.n_jobs)
        with Parallel(n_jobs=n_jobs, verbose=max(0, self.verbose - 1)) as parallel:
            for idx_slice in gen_batches(n_samples, batch_size):
                self._em_step(
                    X[idx_slice, :],
                    total_samples=self.total_samples,
                    batch_update=False,
                    parallel=parallel,
                )

        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Fits from scratch. Validates X, initialises latent variables, then for
        # max_iter passes: online mode runs an online EM step per batch with the corpus
        # size n_samples; batch mode runs one batch EM step on all data. Every
        # evaluate_every passes (if positive) computes perplexity on X from a
        # deterministic E-step, prints it when verbose, and stops early if it changed by
        # less than perp_tol from the previous evaluation; n_iter_ is incremented per
        # completed pass. Finally computes bound_ as the training perplexity.
        # Inputs: X (n_samples, n_features) non-negative counts; y ignored. Outputs:
        #   self.
        # Exceptions: ValueError for negative input.
        # Side effects: sets all fitted attributes and bound_; prints progress when
        #   verbose; consumes randomness.
        # KNOWN ISSUE: the final E-step reuses the joblib Parallel object after its
        #   context manager has exited; it still works but without the managed worker
        #   pool.
        # KNOWN ISSUE: when stopping early, n_iter_ is not incremented for the breaking
        #   pass even though it was completed.
        """Learn model for the data X with variational Bayes method.

        When `learning_method` is 'online', use mini-batch update.
        Otherwise, use batch update.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Fitted estimator.
        """
        X = self._check_non_neg_array(
            X, reset_n_features=True, whom="LatentDirichletAllocation.fit"
        )
        n_samples, n_features = X.shape
        max_iter = self.max_iter
        evaluate_every = self.evaluate_every
        learning_method = self.learning_method

        batch_size = self.batch_size

        # initialize parameters
        self._init_latent_vars(n_features, dtype=X.dtype)
        # change to perplexity later
        last_bound = None
        n_jobs = effective_n_jobs(self.n_jobs)
        with Parallel(n_jobs=n_jobs, verbose=max(0, self.verbose - 1)) as parallel:
            for i in range(max_iter):
                if learning_method == "online":
                    for idx_slice in gen_batches(n_samples, batch_size):
                        self._em_step(
                            X[idx_slice, :],
                            total_samples=n_samples,
                            batch_update=False,
                            parallel=parallel,
                        )
                else:
                    # batch update
                    self._em_step(
                        X, total_samples=n_samples, batch_update=True, parallel=parallel
                    )

                # check perplexity
                if evaluate_every > 0 and (i + 1) % evaluate_every == 0:
                    doc_topics_distr, _ = self._e_step(
                        X, cal_sstats=False, random_init=False, parallel=parallel
                    )
                    bound = self._perplexity_precomp_distr(
                        X, doc_topics_distr, sub_sampling=False
                    )
                    if self.verbose:
                        print(
                            "iteration: %d of max_iter: %d, perplexity: %.4f"
                            % (i + 1, max_iter, bound)
                        )

                    if last_bound and abs(last_bound - bound) < self.perp_tol:
                        break
                    last_bound = bound

                elif self.verbose:
                    print("iteration: %d of max_iter: %d" % (i + 1, max_iter))
                self.n_iter_ += 1

        # calculate final perplexity value on train set
        doc_topics_distr, _ = self._e_step(
            X, cal_sstats=False, random_init=False, parallel=parallel
        )
        self.bound_ = self._perplexity_precomp_distr(
            X, doc_topics_distr, sub_sampling=False
        )

        return self

    def _unnormalized_transform(self, X):
        # [FUNCTION DOC]
        # HELPER: Runs a deterministic E-step (no statistics, no random init) and
        #   returns the unnormalised doc-topic matrix (n_samples, n_components). No
        #   validation.
        """Transform data X according to fitted model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        Returns
        -------
        doc_topic_distr : ndarray of shape (n_samples, n_components)
            Document topic distribution for X.
        """
        doc_topic_distr, _ = self._e_step(X, cal_sstats=False, random_init=False)

        return doc_topic_distr

    def transform(self, X, *, normalize=True):
        # [FUNCTION DOC]
        # Validates X against the fitted model, computes the unnormalised document-topic
        # distribution and, when normalize is True, divides each row by its sum.
        # Inputs: X (n_samples, n_features); normalize bool. Outputs: ndarray
        #   (n_samples, n_components).
        # Exceptions: NotFittedError; ValueError on negative input or feature mismatch.
        """Transform data X according to the fitted model.

        .. versionchanged:: 0.18
            `doc_topic_distr` is now normalized.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        normalize : bool, default=True
            Whether to normalize the document topic distribution.

        Returns
        -------
        doc_topic_distr : ndarray of shape (n_samples, n_components)
            Document topic distribution for X.
        """
        check_is_fitted(self)
        X = self._check_non_neg_array(
            X, reset_n_features=False, whom="LatentDirichletAllocation.transform"
        )
        doc_topic_distr = self._unnormalized_transform(X)
        if normalize:
            doc_topic_distr /= doc_topic_distr.sum(axis=1)[:, np.newaxis]
        return doc_topic_distr

    def fit_transform(self, X, y=None, *, normalize=True):
        # [FUNCTION DOC]
        # Calls fit(X, y) then transform(X, normalize=normalize) and returns the result.
        """
        Fit to data, then transform it.

        Fits transformer to `X` and `y` and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        normalize : bool, default=True
            Whether to normalize the document topic distribution in `transform`.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_components)
            Transformed array.
        """
        return self.fit(X, y).transform(X, normalize=normalize)

    def _approx_bound(self, X, doc_topic_distr, sub_sampling):
        # [FUNCTION DOC]
        # HELPER: Computes the evidence lower bound. Sums for each document the counts
        #   times the log-sum-exp over topics of E[log theta_dk] + E[log beta_kw]; adds
        #   the Dirichlet terms for theta (sum of (prior - gamma) times E[log theta]
        #   plus log-gamma differences); scales by total_samples / n_samples if
        #   sub_sampling; then adds the analogous Dirichlet terms for beta with
        #   lambda=components_.
        # Inputs: X (n_samples, n_features); doc_topic_distr (n_samples, n_components);
        #   sub_sampling bool.
        # Outputs: float bound.
        """Estimate the variational bound.

        Estimate the variational bound over "all documents" using only the
        documents passed in as X. Since log-likelihood of each word cannot
        be computed directly, we use this bound to estimate it.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        doc_topic_distr : ndarray of shape (n_samples, n_components)
            Document topic distribution. In the literature, this is called
            gamma.

        sub_sampling : bool, default=False
            Compensate for subsampling of documents.
            It is used in calculate bound in online learning.

        Returns
        -------
        score : float

        """

        def _loglikelihood(prior, distr, dirichlet_distr, size):
            # [FUNCTION DOC]
            # HELPER: Dirichlet KL-like term: sum of (prior - distr) times the Dirichlet
            #   expectation, plus the sum of log-gamma(distr) minus log-gamma(prior),
            #   plus for each row log-gamma(prior times size) minus log-gamma(row sum of
            #   distr).
            # Inputs: prior scalar; distr 2D array; dirichlet_distr its expected logs;
            #   size number of columns. Outputs: float.
            # calculate log-likelihood
            score = np.sum((prior - distr) * dirichlet_distr)
            score += np.sum(gammaln(distr) - gammaln(prior))
            score += np.sum(gammaln(prior * size) - gammaln(np.sum(distr, 1)))
            return score

        is_sparse_x = sp.issparse(X)
        n_samples, n_components = doc_topic_distr.shape
        n_features = self.components_.shape[1]
        score = 0

        dirichlet_doc_topic = _dirichlet_expectation_2d(doc_topic_distr)
        dirichlet_component_ = _dirichlet_expectation_2d(self.components_)
        doc_topic_prior = self.doc_topic_prior_
        topic_word_prior = self.topic_word_prior_

        if is_sparse_x:
            X_data = X.data
            X_indices = X.indices
            X_indptr = X.indptr

        # E[log p(docs | theta, beta)]
        for idx_d in range(0, n_samples):
            if is_sparse_x:
                ids = X_indices[X_indptr[idx_d] : X_indptr[idx_d + 1]]
                cnts = X_data[X_indptr[idx_d] : X_indptr[idx_d + 1]]
            else:
                ids = np.nonzero(X[idx_d, :])[0]
                cnts = X[idx_d, ids]
            temp = (
                dirichlet_doc_topic[idx_d, :, np.newaxis] + dirichlet_component_[:, ids]
            )
            norm_phi = logsumexp(temp, axis=0)
            score += np.dot(cnts, norm_phi)

        # compute E[log p(theta | alpha) - log q(theta | gamma)]
        score += _loglikelihood(
            doc_topic_prior, doc_topic_distr, dirichlet_doc_topic, self.n_components
        )

        # Compensate for the subsampling of the population of documents
        if sub_sampling:
            doc_ratio = float(self.total_samples) / n_samples
            score *= doc_ratio

        # E[log p(beta | eta) - log q (beta | lambda)]
        score += _loglikelihood(
            topic_word_prior, self.components_, dirichlet_component_, n_features
        )

        return score

    def score(self, X, y=None):
        # [FUNCTION DOC]
        # Returns the approximate log-likelihood (variational bound, no subsampling
        # correction) of X computed from a deterministic E-step.
        # Inputs: X; y ignored. Outputs: float.
        # Exceptions: NotFittedError; ValueError on negative input or feature mismatch.
        """Calculate approximate log-likelihood as score.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        score : float
            Use approximate bound as score.
        """
        check_is_fitted(self)
        X = self._check_non_neg_array(
            X, reset_n_features=False, whom="LatentDirichletAllocation.score"
        )

        doc_topic_distr = self._unnormalized_transform(X)
        score = self._approx_bound(X, doc_topic_distr, sub_sampling=False)
        return score

    def _perplexity_precomp_distr(self, X, doc_topic_distr=None, sub_sampling=False):
        # [FUNCTION DOC]
        # HELPER: Perplexity exp(-bound / word count) for X, where the bound comes from
        #   _approx_bound using a supplied or freshly computed doc-topic matrix and the
        #   word count is scaled by total_samples / n_samples when sub_sampling.
        # Inputs: X; doc_topic_distr optional (n_samples, n_components); sub_sampling
        #   bool. Outputs: float.
        # Exceptions: ValueError if doc_topic_distr row count or topic count does not
        #   match.
        """Calculate approximate perplexity for data X with ability to accept
        precomputed doc_topic_distr

        Perplexity is defined as exp(-1. * log-likelihood per word)

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        doc_topic_distr : ndarray of shape (n_samples, n_components), \
                default=None
            Document topic distribution.
            If it is None, it will be generated by applying transform on X.

        Returns
        -------
        score : float
            Perplexity score.
        """
        if doc_topic_distr is None:
            doc_topic_distr = self._unnormalized_transform(X)
        else:
            n_samples, n_components = doc_topic_distr.shape
            if n_samples != X.shape[0]:
                raise ValueError(
                    "Number of samples in X and doc_topic_distr do not match."
                )

            if n_components != self.n_components:
                raise ValueError("Number of topics does not match.")

        current_samples = X.shape[0]
        bound = self._approx_bound(X, doc_topic_distr, sub_sampling)

        if sub_sampling:
            word_cnt = X.sum() * (float(self.total_samples) / current_samples)
        else:
            word_cnt = X.sum()
        perword_bound = bound / word_cnt

        return np.exp(-1.0 * perword_bound)

    def perplexity(self, X, sub_sampling=False):
        # [FUNCTION DOC]
        # Validates X and returns _perplexity_precomp_distr(X,
        # sub_sampling=sub_sampling).
        # Exceptions: NotFittedError; ValueError on negative input.
        # KNOWN ISSUE: validates with reset_n_features=True, so calling perplexity
        #   overwrites n_features_in_/feature_names_in_ and accepts a different feature
        #   count, which then fails later with a less clear error.
        """Calculate approximate perplexity for data X.

        Perplexity is defined as exp(-1. * log-likelihood per word)

        .. versionchanged:: 0.19
           *doc_topic_distr* argument has been deprecated and is ignored
           because user no longer has access to unnormalized distribution

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        sub_sampling : bool
            Do sub-sampling or not.

        Returns
        -------
        score : float
            Perplexity score.
        """
        check_is_fitted(self)
        X = self._check_non_neg_array(
            X, reset_n_features=True, whom="LatentDirichletAllocation.perplexity"
        )
        return self._perplexity_precomp_distr(X, sub_sampling=sub_sampling)

    @property
    def _n_features_out(self):
        # [FUNCTION DOC]
        # HELPER: Property returning the number of output features, the number of topics
        #   in components_.
        """Number of transformed output features."""
        return self.components_.shape[0]
