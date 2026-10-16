# [MODULE DOC]
# Implements BernoulliRBM, a restricted Boltzmann machine with binary visible and
#   hidden units trained by persistent contrastive divergence (stochastic maximum
#   likelihood) and used as an unsupervised transformer to hidden-unit probabilities.
# Side effects at import: none.
"""Restricted Boltzmann Machine"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import time
from numbers import Integral, Real

import numpy as np
import scipy.sparse as sp
from scipy.special import expit  # logistic function

from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils import check_random_state, gen_even_slices
from sklearn.utils._param_validation import Interval
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted, validate_data


class BernoulliRBM(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    # [CLASS DOC]
    # Transformer estimator. Model: P(h_j=1|v) = logistic(v . W_j + b_hidden_j),
    #   P(v_i=1|h) = logistic(h . W[:, i] + b_visible_i), where W is components_
    #   (n_components x n_features).
    # Training (fit/partial_fit) runs mini-batch PCD updates via _fit; persistent
    #   negative chains are kept in h_samples_. Hyperparameters are validated through
    #   _parameter_constraints (n_components >= 1, learning_rate > 0, batch_size >= 1,
    #   n_iter >= 0).
    """Bernoulli Restricted Boltzmann Machine (RBM).

    A Restricted Boltzmann Machine with binary visible units and
    binary hidden units. Parameters are estimated using Stochastic Maximum
    Likelihood (SML), also known as Persistent Contrastive Divergence (PCD)
    [2].

    The time complexity of this implementation is ``O(d ** 2)`` assuming
    d ~ n_features ~ n_components.

    Read more in the :ref:`User Guide <rbm>`.

    Parameters
    ----------
    n_components : int, default=256
        Number of binary hidden units.

    learning_rate : float, default=0.1
        The learning rate for weight updates. It is *highly* recommended
        to tune this hyper-parameter. Reasonable values are in the
        10**[0., -3.] range.

    batch_size : int, default=10
        Number of examples per minibatch.

    n_iter : int, default=10
        Number of iterations/sweeps over the training dataset to perform
        during training.

    verbose : int, default=0
        The verbosity level. The default, zero, means silent mode. Range
        of values is [0, inf].

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for:

        - Gibbs sampling from visible and hidden layers.

        - Initializing components, sampling from layers during fit.

        - Corrupting the data when scoring samples.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    intercept_hidden_ : array-like of shape (n_components,)
        Biases of the hidden units.

    intercept_visible_ : array-like of shape (n_features,)
        Biases of the visible units.

    components_ : array-like of shape (n_components, n_features)
        Weight matrix, where `n_features` is the number of
        visible units and `n_components` is the number of hidden units.

    h_samples_ : array-like of shape (batch_size, n_components)
        Hidden Activation sampled from the model distribution,
        where `batch_size` is the number of examples per minibatch and
        `n_components` is the number of hidden units.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.neural_network.MLPRegressor : Multi-layer Perceptron regressor.
    sklearn.neural_network.MLPClassifier : Multi-layer Perceptron classifier.
    sklearn.decomposition.PCA : An unsupervised linear dimensionality
        reduction model.

    References
    ----------

    [1] Hinton, G. E., Osindero, S. and Teh, Y. A fast learning algorithm for
        deep belief nets. Neural Computation 18, pp 1527-1554.
        https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    [2] Tieleman, T. Training Restricted Boltzmann Machines using
        Approximations to the Likelihood Gradient. International Conference
        on Machine Learning (ICML) 2008

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.neural_network import BernoulliRBM
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = BernoulliRBM(n_components=2)
    >>> model.fit(X)
    BernoulliRBM(n_components=2)

    For a more detailed example usage, see
    :ref:`sphx_glr_auto_examples_neural_networks_plot_rbm_logistic_classification.py`.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="neither")],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "n_iter": [Interval(Integral, 0, None, closed="left")],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_components=256,
        *,
        learning_rate=0.1,
        batch_size=10,
        n_iter=10,
        verbose=0,
        random_state=None,
    ):
        # [FUNCTION DOC]
        # Stores hyperparameters unchanged (sklearn estimator convention, no
        #   validation here).
        # Inputs: n_components int (256); keyword-only learning_rate float (0.1),
        #   batch_size int (10), n_iter int (10), verbose int (0), random_state
        #   int/RandomState/None.
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state

    def transform(self, X):
        # [FUNCTION DOC]
        # Returns hidden-unit activation probabilities for X.
        # Inputs: X - dense or sparse (converted to CSR) of shape (n_samples,
        #   n_features); float64 or float32 kept.
        # Outputs: ndarray (n_samples, n_components).
        # Exceptions: NotFittedError if not fitted; ValueError on feature count
        #   mismatch.
        """Compute the hidden layer activation probabilities, P(h=1|v=X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        h : ndarray of shape (n_samples, n_components)
            Latent representations of the data.
        """
        check_is_fitted(self)

        X = validate_data(
            self, X, accept_sparse="csr", reset=False, dtype=(np.float64, np.float32)
        )
        return self._mean_hiddens(X)

    def _mean_hiddens(self, v):
        # [FUNCTION DOC]
        # HELPER: Computes logistic(v times components_ transposed plus
        #   intercept_hidden_).
        # Inputs: v - dense or sparse (n_samples, n_features).
        # Outputs: dense ndarray (n_samples, n_components) of probabilities (computed
        #   in place in a fresh buffer).
        """Computes the probabilities P(h=1|v).

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        h : ndarray of shape (n_samples, n_components)
            Corresponding mean field values for the hidden layer.
        """
        p = safe_sparse_dot(v, self.components_.T)
        p += self.intercept_hidden_
        return expit(p, out=p)

    def _sample_hiddens(self, v, rng):
        # [FUNCTION DOC]
        # HELPER: Draws binary hidden states: each unit is True where a uniform draw
        #   is below its mean activation.
        # Inputs: v - visible layer; rng - RandomState.
        # Outputs: boolean ndarray (n_samples, n_components).
        # Side effects: advances rng.
        """Sample from the distribution P(h|v).

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer to sample from.

        rng : RandomState instance
            Random number generator to use.

        Returns
        -------
        h : ndarray of shape (n_samples, n_components)
            Values of the hidden layer.
        """
        p = self._mean_hiddens(v)
        return rng.uniform(size=p.shape) < p

    def _sample_visibles(self, h, rng):
        # [FUNCTION DOC]
        # HELPER: Draws binary visible states from P(v|h): probabilities are
        #   logistic(h times components_ plus intercept_visible_), then compared with
        #   uniform draws.
        # Inputs: h - (n_samples, n_components); rng - RandomState.
        # Outputs: boolean ndarray (n_samples, n_features).
        # Side effects: advances rng.
        """Sample from the distribution P(v|h).

        Parameters
        ----------
        h : ndarray of shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        rng : RandomState instance
            Random number generator to use.

        Returns
        -------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer.
        """
        p = np.dot(h, self.components_)
        p += self.intercept_visible_
        expit(p, out=p)
        return rng.uniform(size=p.shape) < p

    def _free_energy(self, v):
        # [FUNCTION DOC]
        # HELPER: Free energy per sample: minus v . intercept_visible_ minus the sum
        #   over hidden units of log(1 + exp(v . W_j + b_hidden_j)) (computed stably
        #   with logaddexp).
        # Inputs: v - dense or sparse (n_samples, n_features).
        # Outputs: ndarray (n_samples,).
        """Computes the free energy F(v) = - log sum_h exp(-E(v,h)).

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        free_energy : ndarray of shape (n_samples,)
            The value of the free energy.
        """
        return -safe_sparse_dot(v, self.intercept_visible_) - np.logaddexp(
            0, safe_sparse_dot(v, self.components_.T) + self.intercept_hidden_
        ).sum(axis=1)

    def gibbs(self, v):
        # [FUNCTION DOC]
        # Performs one Gibbs step v -> sampled h -> sampled v.
        # Inputs: v - ndarray (n_samples, n_features); not validated.
        # Outputs: boolean ndarray (n_samples, n_features).
        # Exceptions: NotFittedError if not fitted.
        # Side effects: creates self.random_state_ from random_state on first use and
        #   advances it.
        """Perform one Gibbs sampling step.

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer to start from.

        Returns
        -------
        v_new : ndarray of shape (n_samples, n_features)
            Values of the visible layer after one Gibbs step.
        """
        check_is_fitted(self)
        if not hasattr(self, "random_state_"):
            self.random_state_ = check_random_state(self.random_state)
        h_ = self._sample_hiddens(v, self.random_state_)
        v_ = self._sample_visibles(h_, self.random_state_)

        return v_

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None):
        # [FUNCTION DOC]
        # Incremental fit on one chunk of data, treated as a single mini-batch
        #   (batch_size is not used to split it). On the first call validates X with
        #   reset (sets n_features_in_), creates random_state_, initialises components_
        #   from N(0, 0.01) in Fortran order, zero hidden and visible intercepts and
        #   zero h_samples_ of shape (batch_size, n_components); then calls _fit once.
        # Inputs: X - array or sparse (converted to CSR float64); y ignored.
        # Outputs: None.
        # Side effects: sets/updates fitted attributes and _n_features_out; advances
        #   random_state_.
        # KNOWN ISSUE: docstring says it returns self, but the method has no return
        #   statement and returns None, so chaining like partial_fit(X).transform(X)
        #   fails.
        """Fit the model to the partial segment of the data X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        first_pass = not hasattr(self, "components_")
        X = validate_data(
            self, X, accept_sparse="csr", dtype=np.float64, reset=first_pass
        )
        if not hasattr(self, "random_state_"):
            self.random_state_ = check_random_state(self.random_state)
        if not hasattr(self, "components_"):
            self.components_ = np.asarray(
                self.random_state_.normal(0, 0.01, (self.n_components, X.shape[1])),
                order="F",
            )
            self._n_features_out = self.components_.shape[0]
        if not hasattr(self, "intercept_hidden_"):
            self.intercept_hidden_ = np.zeros(
                self.n_components,
            )
        if not hasattr(self, "intercept_visible_"):
            self.intercept_visible_ = np.zeros(
                X.shape[1],
            )
        if not hasattr(self, "h_samples_"):
            self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        self._fit(X, self.random_state_)

    def _fit(self, v_pos, rng):
        # [FUNCTION DOC]
        # HELPER: One PCD update on a mini-batch. Steps: positive hidden probabilities
        #   from v_pos; negative visibles sampled from the persistent h_samples_ and
        #   their hidden probabilities h_neg; lr = learning_rate / batch rows;
        #   components_ += lr * (h_pos^T v_pos - h_neg^T v_neg); intercept_hidden_ += lr
        #   * (column sums of h_pos - column sums of h_neg); intercept_visible_ += lr *
        #   (column sums of v_pos - column sums of v_neg); finally h_neg is binarised by
        #   sampling (units set to 1 where uniform < prob, then floored) and stored as
        #   h_samples_.
        # Inputs: v_pos - dense or sparse batch (n_batch, n_features); rng -
        #   RandomState.
        # Outputs: None.
        # Side effects: mutates components_, intercepts, h_samples_; advances rng.
        # KNOWN ISSUE: lr divides by the rows of v_pos while the negative statistics
        #   are summed over the batch_size persistent chains; when the batch is
        #   smaller (last uneven batch, partial_fit chunks) the two terms are scaled
        #   inconsistently.
        """Inner fit for one mini-batch.

        Adjust the parameters to maximize the likelihood of v using
        Stochastic Maximum Likelihood (SML).

        Parameters
        ----------
        v_pos : ndarray of shape (n_samples, n_features)
            The data to use for training.

        rng : RandomState instance
            Random number generator to use for sampling.
        """
        h_pos = self._mean_hiddens(v_pos)
        v_neg = self._sample_visibles(self.h_samples_, rng)
        h_neg = self._mean_hiddens(v_neg)

        lr = float(self.learning_rate) / v_pos.shape[0]
        update = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T
        update -= np.dot(h_neg.T, v_neg)
        self.components_ += lr * update
        self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        self.intercept_visible_ += lr * (
            np.asarray(v_pos.sum(axis=0)).squeeze() - v_neg.sum(axis=0)
        )

        h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
        self.h_samples_ = np.floor(h_neg, h_neg)

    def score_samples(self, X):
        # [FUNCTION DOC]
        # Pseudo-likelihood proxy: for each sample flips one randomly chosen feature
        #   (x -> 1 - x; for sparse input by adding a CSR delta of 1 - 2x at that entry,
        #   as matrix or array matching the input type), computes free energies of the
        #   original and corrupted data and returns n_features times
        #   log(logistic(fe_corrupted - fe_original)).
        # Inputs: X - array or sparse (n_samples, n_features), assumed binary but not
        #   checked.
        # Outputs: ndarray (n_samples,).
        # Exceptions: NotFittedError; ValueError on feature mismatch.
        # Side effects: none persistent; randomness comes from a fresh
        #   check_random_state(random_state), so an int seed gives deterministic results
        #   while None does not.
        """Compute the pseudo-likelihood of X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Values of the visible layer. Must be all-boolean (not checked).

        Returns
        -------
        pseudo_likelihood : ndarray of shape (n_samples,)
            Value of the pseudo-likelihood (proxy for likelihood).

        Notes
        -----
        This method is not deterministic: it computes a quantity called the
        free energy on X, then on a randomly corrupted version of X, and
        returns the log of the logistic function of the difference.
        """
        check_is_fitted(self)

        v = validate_data(self, X, accept_sparse="csr", reset=False)
        rng = check_random_state(self.random_state)

        # Randomly corrupt one feature in each sample in v.
        ind = (np.arange(v.shape[0]), rng.randint(0, v.shape[1], v.shape[0]))
        if sp.issparse(v):
            data = -2 * v[ind] + 1
            if isinstance(data, np.matrix):  # v is a sparse matrix
                v_ = v + sp.csr_matrix((data.A.ravel(), ind), shape=v.shape)
            else:  # v is a sparse array
                v_ = v + sp.csr_array((data.ravel(), ind), shape=v.shape)
        else:
            v_ = v.copy()
            v_[ind] = 1 - v_[ind]

        fe = self._free_energy(v)
        fe_ = self._free_energy(v_)
        # log(expit(x)) = log(1 / (1 + exp(-x)) = -np.logaddexp(0, -x)
        return -v.shape[1] * np.logaddexp(0, -(fe_ - fe))

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        # [FUNCTION DOC]
        # Full fit from scratch. Validates X (CSR allowed, float64/float32 preserved),
        #   draws components_ from N(0, 0.01) in X's dtype (Fortran order), zero
        #   intercepts and h_samples_ of shape (batch_size, n_components); splits the
        #   n_samples rows into ceil(n_samples / batch_size) even contiguous slices and,
        #   for n_iter epochs, calls _fit on each slice in order (no shuffling). With
        #   verbose, after each epoch prints the mean pseudo-likelihood and elapsed
        #   time.
        # Inputs: X - (n_samples, n_features); y ignored.
        # Outputs: self.
        # Side effects: sets components_, intercept_hidden_, intercept_visible_,
        #   h_samples_, _n_features_out, n_features_in_ (and feature_names_in_); prints
        #   if verbose. Does not set random_state_ (uses a local rng).
        """Fit the model to the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        X = validate_data(self, X, accept_sparse="csr", dtype=(np.float64, np.float32))
        n_samples = X.shape[0]
        rng = check_random_state(self.random_state)

        self.components_ = np.asarray(
            rng.normal(0, 0.01, (self.n_components, X.shape[1])),
            order="F",
            dtype=X.dtype,
        )
        self._n_features_out = self.components_.shape[0]
        self.intercept_hidden_ = np.zeros(self.n_components, dtype=X.dtype)
        self.intercept_visible_ = np.zeros(X.shape[1], dtype=X.dtype)
        self.h_samples_ = np.zeros((self.batch_size, self.n_components), dtype=X.dtype)

        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(
            gen_even_slices(n_batches * self.batch_size, n_batches, n_samples=n_samples)
        )
        verbose = self.verbose
        begin = time.time()
        for iteration in range(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                self._fit(X[batch_slice], rng)

            if verbose:
                end = time.time()
                print(
                    "[%s] Iteration %d, pseudo-likelihood = %.2f, time = %.2fs"
                    % (
                        type(self).__name__,
                        iteration,
                        self.score_samples(X).mean(),
                        end - begin,
                    )
                )
                begin = end

        return self

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Declares sparse input support and that float64/float32 dtypes are preserved
        #   by transform.
        # Outputs: Tags object.
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags
