# [MODULE DOC]
# Multi-layer perceptron estimators: BaseMultilayerPerceptron (shared training
#   machinery: weight initialisation, forward pass, backpropagation, L-BFGS and
#   stochastic sgd/adam solvers, early stopping), MLPClassifier (log loss, logistic or
#   softmax output) and MLPRegressor (squared error with identity output or Poisson with
#   exp output).
# Parameters of all layers are kept as lists coefs_ (one (fan_in, fan_out) matrix per
#   layer) and intercepts_ (one vector per layer). The L-BFGS solver works on one flat
#   vector built by _pack.
# Side effects at import: none.
"""Multi-layer Perceptron"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from abc import ABC, abstractmethod
from itertools import chain, pairwise
from numbers import Integral, Real

import numpy as np
import scipy.optimize

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    _fit_context,
    is_classifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network._base import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer, SGDOptimizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import (
    _safe_indexing,
    check_random_state,
    column_or_1d,
    gen_batches,
    shuffle,
)
from sklearn.utils._param_validation import Interval, Options, StrOptions
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.fixes import _get_additional_lbfgs_options_dict
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import (
    _check_partial_fit_first_call,
    type_of_target,
    unique_labels,
)
from sklearn.utils.optimize import _check_optimize_result
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

_STOCHASTIC_SOLVERS = ["sgd", "adam"]


def _pack(coefs_, intercepts_):
    # [FUNCTION DOC]
    # HELPER: Flattens every coefficient matrix and then every intercept vector, in
    #   layer order, and concatenates them into one 1-D array (the layout that _unpack
    #   inverts via _coef_indptr/_intercept_indptr).
    # Inputs: coefs_, intercepts_ - lists of arrays.
    # Outputs: 1-D ndarray of length equal to the total number of parameters.
    """Pack the parameters into a single vector."""
    return np.hstack([l.ravel() for l in coefs_ + intercepts_])


class BaseMultilayerPerceptron(BaseEstimator, ABC):
    # [CLASS DOC]
    # Abstract base for the MLP estimators. Subclasses supply loss via __init__ and
    #   implement _validate_input, _predict and _score. Hyperparameters are validated by
    #   _parameter_constraints before fit/partial_fit.
    # Fitted state shared by subclasses: coefs_, intercepts_, n_layers_, n_outputs_,
    #   out_activation_, n_iter_, t_, loss_, plus for stochastic solvers loss_curve_,
    #   best_loss_, validation_scores_, best_validation_score_ and private _optimizer,
    #   _best_coefs, _best_intercepts, _no_improvement_count, _random_state.
    """Base class for MLP classification and regression.

    Warning: This class should not be used directly.
    Use derived classes instead.

    .. versionadded:: 0.18
    """

    _parameter_constraints: dict = {
        "hidden_layer_sizes": [
            "array-like",
            Interval(Integral, 1, None, closed="left"),
        ],
        "activation": [StrOptions({"identity", "logistic", "tanh", "relu"})],
        "solver": [StrOptions({"lbfgs", "sgd", "adam"})],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "batch_size": [
            StrOptions({"auto"}),
            Interval(Integral, 1, None, closed="left"),
        ],
        "learning_rate": [StrOptions({"constant", "invscaling", "adaptive"})],
        "learning_rate_init": [Interval(Real, 0, None, closed="neither")],
        "power_t": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "shuffle": ["boolean"],
        "random_state": ["random_state"],
        "tol": [Interval(Real, 0, None, closed="left")],
        "verbose": ["verbose"],
        "warm_start": ["boolean"],
        "momentum": [Interval(Real, 0, 1, closed="both")],
        "nesterovs_momentum": ["boolean"],
        "early_stopping": ["boolean"],
        "validation_fraction": [Interval(Real, 0, 1, closed="left")],
        "beta_1": [Interval(Real, 0, 1, closed="left")],
        "beta_2": [Interval(Real, 0, 1, closed="left")],
        "epsilon": [Interval(Real, 0, None, closed="neither")],
        "n_iter_no_change": [
            Interval(Integral, 1, None, closed="left"),
            Options(Real, {np.inf}),
        ],
        "max_fun": [Interval(Integral, 1, None, closed="left")],
    }

    @abstractmethod
    def __init__(
        self,
        hidden_layer_sizes,
        activation,
        solver,
        alpha,
        batch_size,
        learning_rate,
        learning_rate_init,
        power_t,
        max_iter,
        loss,
        shuffle,
        random_state,
        tol,
        verbose,
        warm_start,
        momentum,
        nesterovs_momentum,
        early_stopping,
        validation_fraction,
        beta_1,
        beta_2,
        epsilon,
        n_iter_no_change,
        max_fun,
    ):
        # [FUNCTION DOC]
        # Stores every hyperparameter unchanged (sklearn convention, no validation).
        #   Abstract: only subclasses are instantiated.
        # Inputs: see the class parameter list; loss is the loss name chosen by the
        #   subclass.
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.loss = loss
        self.hidden_layer_sizes = hidden_layer_sizes
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun

    def _unpack(self, packed_parameters):
        # [FUNCTION DOC]
        # HELPER: Inverse of _pack. For each layer, slices the flat vector using the
        #   precomputed (start, end, shape) entries in _coef_indptr and (start, end) in
        #   _intercept_indptr and assigns reshaped views into coefs_[i] and
        #   intercepts_[i].
        # Inputs: packed_parameters - 1-D array from L-BFGS.
        # Side effects: replaces the entries of self.coefs_ and self.intercepts_
        #   (views on the input array, not copies). Requires _fit_lbfgs to have set the
        #   index pointers.
        """Extract the coefficients and intercepts from packed_parameters."""
        for i in range(self.n_layers_ - 1):
            start, end, shape = self._coef_indptr[i]
            self.coefs_[i] = np.reshape(packed_parameters[start:end], shape)

            start, end = self._intercept_indptr[i]
            self.intercepts_[i] = packed_parameters[start:end]

    def _forward_pass(self, activations):
        # [FUNCTION DOC]
        # HELPER: Training-time forward pass. For each layer i, activations[i + 1] is
        #   activations[i] times coefs_[i] plus intercepts_[i]; the hidden activation
        #   function is applied in place on all but the last layer, and the output
        #   activation (out_activation_) in place on the last.
        # Inputs: activations - list whose element 0 is the input batch; other entries
        #   are overwritten.
        # Outputs: the same list, filled.
        # Side effects: mutates the activations list.
        """Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.

        Parameters
        ----------
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        """
        hidden_activation = ACTIVATIONS[self.activation]
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = safe_sparse_dot(activations[i], self.coefs_[i])
            activations[i + 1] += self.intercepts_[i]

            # For the hidden layers
            if (i + 1) != (self.n_layers_ - 1):
                hidden_activation(activations[i + 1])

        # For the last layer
        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activations[i + 1])

        return activations

    def _forward_pass_fast(self, X, check_input=True):
        # [FUNCTION DOC]
        # HELPER: Prediction-time forward pass that keeps only the current layer's
        #   activation instead of all of them. Optionally validates X (CSR/CSC accepted,
        #   reset=False).
        # Inputs: X - (n_samples, n_features) array or sparse; check_input - bool,
        #   default True.
        # Outputs: ndarray (n_samples, n_outputs_) of output-layer activations
        #   (probabilities for classifiers, predictions for regressors).
        # Exceptions: ValueError from validate_data on bad input or feature mismatch.
        """Predict using the trained model

        This is the same as _forward_pass but does not record the activations
        of all layers and only returns the last layer's activation.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        check_input : bool, default=True
            Perform input data validation or not.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The decision function of the samples for each class in the model.
        """
        if check_input:
            X = validate_data(self, X, accept_sparse=["csr", "csc"], reset=False)

        # Initialize first layer
        activation = X

        # Forward propagate
        hidden_activation = ACTIVATIONS[self.activation]
        for i in range(self.n_layers_ - 1):
            activation = safe_sparse_dot(activation, self.coefs_[i])
            activation += self.intercepts_[i]
            if i != self.n_layers_ - 2:
                hidden_activation(activation)
        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activation)

        return activation

    def _compute_loss_grad(
        self, layer, sw_sum, activations, deltas, coef_grads, intercept_grads
    ):
        # [FUNCTION DOC]
        # HELPER: Gradients for one layer: coef grad is activations[layer] transposed
        #   times deltas[layer] plus alpha times coefs_[layer], all divided by sw_sum;
        #   intercept grad is the column sum of deltas[layer] divided by sw_sum.
        # Inputs: layer - int index; sw_sum - number of samples or sum of sample
        #   weights; activations, deltas - lists from backprop; coef_grads,
        #   intercept_grads - output lists.
        # Outputs: None.
        # Side effects: overwrites coef_grads[layer] and intercept_grads[layer].
        """Compute the gradient of loss with respect to coefs and intercept for
        specified layer.

        This function does backpropagation for the specified one layer.
        """
        coef_grads[layer] = safe_sparse_dot(activations[layer].T, deltas[layer])
        coef_grads[layer] += self.alpha * self.coefs_[layer]
        coef_grads[layer] /= sw_sum

        intercept_grads[layer] = np.sum(deltas[layer], axis=0) / sw_sum

    def _loss_grad_lbfgs(
        self,
        packed_coef_inter,
        X,
        y,
        sample_weight,
        activations,
        deltas,
        coef_grads,
        intercept_grads,
    ):
        # [FUNCTION DOC]
        # HELPER: Objective function for scipy L-BFGS: unpacks the flat parameter
        #   vector into coefs_/intercepts_, runs _backprop and returns the loss with the
        #   gradient flattened by _pack.
        # Inputs: packed_coef_inter - flat parameter vector; X, y, sample_weight;
        #   preallocated activations, deltas, coef_grads, intercept_grads lists.
        # Outputs: tuple (loss float, gradient 1-D array).
        # Side effects: updates self.coefs_/intercepts_ and the work lists.
        """Compute the MLP loss function and its corresponding derivatives
        with respect to the different parameters given in the initialization.

        Returned gradients are packed in a single vector so it can be used
        in lbfgs

        Parameters
        ----------
        packed_coef_inter : ndarray
            A vector comprising the flattened coefficients and intercepts.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,)
            The target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.

        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function

        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.

        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.

        Returns
        -------
        loss : float
        grad : array-like, shape (number of nodes of all layers,)
        """
        self._unpack(packed_coef_inter)
        loss, coef_grads, intercept_grads = self._backprop(
            X, y, sample_weight, activations, deltas, coef_grads, intercept_grads
        )
        grad = _pack(coef_grads, intercept_grads)
        return loss, grad

    def _backprop(
        self, X, y, sample_weight, activations, deltas, coef_grads, intercept_grads
    ):
        # [FUNCTION DOC]
        # HELPER: Computes loss and gradients for one batch. Steps: forward pass; loss
        #   from LOSS_FUNCTIONS (log_loss is switched to binary_log_loss when the output
        #   is logistic) using sample_weight; add L2 penalty 0.5 * alpha * (sum of
        #   squared coefficients, intercepts excluded) divided by sw_sum (n_samples, or
        #   the sum of sample weights). Output delta is prediction minus y (valid for
        #   the canonical link/loss pairs used), multiplied row-wise by sample weights
        #   if given; gradient of the last layer; then for each hidden layer from last
        #   to first, delta is the next delta times the transposed next coefficients,
        #   multiplied in place by the activation derivative, and its gradients are
        #   computed.
        # Inputs: X - batch (n_samples, n_features), possibly sparse; y - 2-D targets
        #   (n_samples, n_outputs); sample_weight - (n_samples,) or None; activations,
        #   deltas, coef_grads, intercept_grads - work lists.
        # Outputs: tuple (loss float, coef_grads list, intercept_grads list).
        # Side effects: overwrites the work lists in place.
        """Compute the MLP loss function and its corresponding derivatives
        with respect to each parameter: weights and bias vectors.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,)
            The target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        activations : list, length = n_layers - 1
             The ith element of the list holds the values of the ith layer.

        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function

        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.

        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.

        Returns
        -------
        loss : float
        coef_grads : list, length = n_layers - 1
        intercept_grads : list, length = n_layers - 1
        """
        n_samples = X.shape[0]

        # Forward propagate
        activations = self._forward_pass(activations)

        # Get loss
        loss_func_name = self.loss
        if loss_func_name == "log_loss" and self.out_activation_ == "logistic":
            loss_func_name = "binary_log_loss"
        loss = LOSS_FUNCTIONS[loss_func_name](y, activations[-1], sample_weight)
        # Add L2 regularization term to loss
        values = 0
        for s in self.coefs_:
            s = s.ravel()
            values += np.dot(s, s)
        if sample_weight is None:
            sw_sum = n_samples
        else:
            sw_sum = sample_weight.sum()
        loss += (0.5 * self.alpha) * values / sw_sum

        # Backward propagate
        last = self.n_layers_ - 2

        # The calculation of delta[last] is as follows:
        #   delta[last] = d/dz loss(y, act(z)) = act(z) - y
        # with z=x@w + b being the output of the last layer before passing through the
        # output activation, act(z) = activations[-1].
        # The simple formula for delta[last] here works with following (canonical
        # loss-link) combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        deltas[last] = activations[-1] - y
        if sample_weight is not None:
            deltas[last] *= sample_weight.reshape(-1, 1)

        # Compute gradient for the last layer
        self._compute_loss_grad(
            last, sw_sum, activations, deltas, coef_grads, intercept_grads
        )

        inplace_derivative = DERIVATIVES[self.activation]
        # Iterate over the hidden layers
        for i in range(last, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative(activations[i], deltas[i - 1])

            self._compute_loss_grad(
                i - 1, sw_sum, activations, deltas, coef_grads, intercept_grads
            )

        return loss, coef_grads, intercept_grads

    def _initialize(self, y, layer_units, dtype):
        # [FUNCTION DOC]
        # HELPER: First-fit initialisation. Resets n_iter_ and t_ to 0, sets
        #   n_outputs_ and n_layers_, chooses out_activation_ (regressor: exp for
        #   poisson else identity; classifier: softmax if the label binarizer saw a
        #   multiclass target, otherwise logistic for binary and multilabel),
        #   initialises each layer with _init_coef, keeps copies as
        #   _best_coefs/_best_intercepts and, for sgd/adam, resets loss_curve_,
        #   _no_improvement_count and the early-stopping bookkeeping (validation_scores_
        #   list and best_validation_score_ = -inf when early_stopping, else best_loss_
        #   = inf).
        # Inputs: y - 2-D target array; layer_units - list of layer sizes including
        #   input and output; dtype - float dtype for the weights.
        # Side effects: sets the attributes above; consumes self._random_state.
        # set all attributes, allocate weights etc. for first call
        # Initialize parameters
        self.n_iter_ = 0
        self.t_ = 0
        self.n_outputs_ = y.shape[1]

        # Compute the number of layers
        self.n_layers_ = len(layer_units)

        # Output for regression
        if not is_classifier(self):
            if self.loss == "poisson":
                self.out_activation_ = "exp"
            else:
                # loss = "squared_error"
                self.out_activation_ = "identity"
        # Output for multi class
        elif self._label_binarizer.y_type_ == "multiclass":
            self.out_activation_ = "softmax"
        # Output for binary class and multi-label
        else:
            self.out_activation_ = "logistic"

        # Initialize coefficient and intercept layers
        self.coefs_ = []
        self.intercepts_ = []

        for i in range(self.n_layers_ - 1):
            coef_init, intercept_init = self._init_coef(
                layer_units[i], layer_units[i + 1], dtype
            )
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)

        self._best_coefs = [c.copy() for c in self.coefs_]
        self._best_intercepts = [i.copy() for i in self.intercepts_]

        if self.solver in _STOCHASTIC_SOLVERS:
            self.loss_curve_ = []
            self._no_improvement_count = 0
            if self.early_stopping:
                self.validation_scores_ = []
                self.best_validation_score_ = -np.inf
                self.best_loss_ = None
            else:
                self.best_loss_ = np.inf
                self.validation_scores_ = None
                self.best_validation_score_ = None

    def _init_coef(self, fan_in, fan_out, dtype):
        # [FUNCTION DOC]
        # HELPER: Glorot uniform initialisation: bound = sqrt(factor / (fan_in +
        #   fan_out)) with factor 2 for logistic activation and 6 otherwise; weights of
        #   shape (fan_in, fan_out) and intercepts of shape (fan_out,) are drawn
        #   uniformly in [-bound, bound] and cast to dtype.
        # Inputs: fan_in, fan_out - ints; dtype.
        # Outputs: (coef_init, intercept_init).
        # Side effects: advances self._random_state.
        # Use the initialization method recommended by
        # Glorot et al.
        factor = 6.0
        if self.activation == "logistic":
            factor = 2.0
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        # Generate weights and bias:
        coef_init = self._random_state.uniform(
            -init_bound, init_bound, (fan_in, fan_out)
        )
        intercept_init = self._random_state.uniform(-init_bound, init_bound, fan_out)
        coef_init = coef_init.astype(dtype, copy=False)
        intercept_init = intercept_init.astype(dtype, copy=False)
        return coef_init, intercept_init

    def _fit(self, X, y, sample_weight=None, incremental=False):
        # [FUNCTION DOC]
        # HELPER: Shared training entry for fit and partial_fit. Normalises
        #   hidden_layer_sizes to a list and checks all sizes are positive; decides
        #   first_pass (no coefs_ yet, or neither warm_start nor incremental); validates
        #   X and y through the subclass _validate_input (resetting feature info on
        #   first pass) and sample_weight; reshapes y to 2-D; builds layer_units =
        #   [n_features] + hidden sizes + [n_outputs]; creates self._random_state from
        #   random_state (recreated on every call); initialises the network on first
        #   pass; allocates activations, deltas and gradient buffers in X's dtype;
        #   dispatches to _fit_stochastic or _fit_lbfgs.
        # Inputs: X, y; sample_weight - (n_samples,) or None; incremental - True from
        #   partial_fit.
        # Outputs: self.
        # Exceptions: ValueError if a hidden layer size is <= 0, if validation fails,
        #   or if the solver produced non-finite weights.
        # Side effects: sets/updates all fitted attributes.
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError(
                "hidden_layer_sizes must be > 0, got %s." % hidden_layer_sizes
            )
        first_pass = not hasattr(self, "coefs_") or (
            not self.warm_start and not incremental
        )

        X, y = self._validate_input(X, y, incremental, reset=first_pass)
        n_samples, n_features = X.shape
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs_ = y.shape[1]

        layer_units = [n_features] + hidden_layer_sizes + [self.n_outputs_]

        # check random state
        self._random_state = check_random_state(self.random_state)

        if first_pass:
            # First time training the model
            self._initialize(y, layer_units, X.dtype)

        # Initialize lists
        activations = [X] + [None] * (len(layer_units) - 1)
        deltas = [None] * (len(activations) - 1)

        coef_grads = [
            np.empty((n_fan_in_, n_fan_out_), dtype=X.dtype)
            for n_fan_in_, n_fan_out_ in pairwise(layer_units)
        ]

        intercept_grads = [
            np.empty(n_fan_out_, dtype=X.dtype) for n_fan_out_ in layer_units[1:]
        ]

        # Run the Stochastic optimization solver
        if self.solver in _STOCHASTIC_SOLVERS:
            self._fit_stochastic(
                X,
                y,
                sample_weight,
                activations,
                deltas,
                coef_grads,
                intercept_grads,
                layer_units,
                incremental,
            )

        # Run the LBFGS solver
        elif self.solver == "lbfgs":
            self._fit_lbfgs(
                X,
                y,
                sample_weight,
                activations,
                deltas,
                coef_grads,
                intercept_grads,
                layer_units,
            )

        # validate parameter weights
        weights = chain(self.coefs_, self.intercepts_)
        if not all(np.isfinite(w).all() for w in weights):
            raise ValueError(
                "Solver produced non-finite parameter weights. The input data may"
                " contain large values and need to be preprocessed."
            )

        return self

    def _fit_lbfgs(
        self,
        X,
        y,
        sample_weight,
        activations,
        deltas,
        coef_grads,
        intercept_grads,
        layer_units,
    ):
        # [FUNCTION DOC]
        # HELPER: Trains with scipy L-BFGS-B on the flat parameter vector. Builds
        #   _coef_indptr (start, end, shape per layer) and _intercept_indptr (start, end
        #   per layer) matching _pack's layout, then minimises _loss_grad_lbfgs with
        #   jac=True, maxfun=max_fun, maxiter=max_iter, gtol=tol and iprint 1 when
        #   verbose else -1.
        # Inputs: X, y, sample_weight and the work buffers; layer_units.
        # Side effects: sets _coef_indptr, _intercept_indptr, n_iter_ (via
        #   _check_optimize_result, which warns ConvergenceWarning when not converged),
        #   loss_, and the final coefs_/intercepts_.
        # Store meta information for the parameters
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0

        # Save sizes and indices of coefficients for faster unpacking
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        # Save sizes and indices of intercepts for faster unpacking
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        # Run LBFGS
        packed_coef_inter = _pack(self.coefs_, self.intercepts_)

        if self.verbose is True or self.verbose >= 1:
            iprint = 1
        else:
            iprint = -1

        opt_res = scipy.optimize.minimize(
            self._loss_grad_lbfgs,
            packed_coef_inter,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxfun": self.max_fun,
                "maxiter": self.max_iter,
                "gtol": self.tol,
                **_get_additional_lbfgs_options_dict("iprint", iprint),
            },
            args=(
                X,
                y,
                sample_weight,
                activations,
                deltas,
                coef_grads,
                intercept_grads,
            ),
        )
        self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)
        self.loss_ = opt_res.fun
        self._unpack(opt_res.x)

    def _fit_stochastic(
        self,
        X,
        y,
        sample_weight,
        activations,
        deltas,
        coef_grads,
        intercept_grads,
        layer_units,
        incremental,
    ):
        # [FUNCTION DOC]
        # HELPER: Mini-batch training with sgd or adam. Creates a new optimizer unless
        #   incremental and one exists. If early_stopping: holds out validation_fraction
        #   of the data with train_test_split (stratified for single-output classifiers,
        #   sample weights split along), errors if fewer than 2 validation samples, and
        #   converts classifier validation targets back to labels. Batch size is
        #   min(200, n_samples) for auto, otherwise clipped to [1, n_samples] with a
        #   warning when larger. For each epoch up to max_iter: optionally shuffle
        #   sample indices; for each batch run _backprop and optimizer.update_params on
        #   coefs_ + intercepts_ (in place); accumulate loss weighted by batch length;
        #   set loss_ (accumulated loss divided by n_train), increase t_ by n_samples,
        #   append to loss_curve_, optionally print; update the no-improvement counter;
        #   let the optimizer update its learning rate; if the counter exceeds
        #   n_iter_no_change ask the optimizer to stop (break) or reset the counter;
        #   stop after one epoch when incremental; warn ConvergenceWarning when max_iter
        #   epochs are completed. KeyboardInterrupt is caught and turned into a warning.
        #   With early_stopping, best coefficients are restored at the end.
        # Inputs: X, y (2-D, binarized for classifiers), sample_weight, work buffers,
        #   layer_units, incremental.
        # Exceptions: ValueError if early_stopping is used with partial_fit or the
        #   validation set is too small.
        # Side effects: mutates many fitted attributes, prints when verbose, emits
        #   warnings, consumes _random_state.
        # KNOWN ISSUE: the batch_size warning text mentions "less than 1" although
        #   only the larger-than-n_samples case is checked (values below 1 are rejected
        #   by parameter validation).
        params = self.coefs_ + self.intercepts_
        if not incremental or not hasattr(self, "_optimizer"):
            if self.solver == "sgd":
                self._optimizer = SGDOptimizer(
                    params,
                    self.learning_rate_init,
                    self.learning_rate,
                    self.momentum,
                    self.nesterovs_momentum,
                    self.power_t,
                )
            elif self.solver == "adam":
                self._optimizer = AdamOptimizer(
                    params,
                    self.learning_rate_init,
                    self.beta_1,
                    self.beta_2,
                    self.epsilon,
                )

        # early_stopping in partial_fit doesn't make sense
        if self.early_stopping and incremental:
            raise ValueError("partial_fit does not support early_stopping=True")
        early_stopping = self.early_stopping
        if early_stopping:
            # don't stratify in multilabel classification
            should_stratify = is_classifier(self) and self.n_outputs_ == 1
            stratify = y if should_stratify else None
            if sample_weight is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y,
                    random_state=self._random_state,
                    test_size=self.validation_fraction,
                    stratify=stratify,
                )
                sample_weight_train = sample_weight_val = None
            else:
                # TODO: incorporate sample_weight in sampling here.
                (
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    sample_weight_train,
                    sample_weight_val,
                ) = train_test_split(
                    X,
                    y,
                    sample_weight,
                    random_state=self._random_state,
                    test_size=self.validation_fraction,
                    stratify=stratify,
                )
            if X_val.shape[0] < 2:
                raise ValueError(
                    "The validation set is too small. Increase 'validation_fraction' "
                    "or the size of your dataset."
                )

            if is_classifier(self):
                y_val = self._label_binarizer.inverse_transform(y_val)
        else:
            X_train, y_train, sample_weight_train = X, y, sample_weight
            X_val = y_val = sample_weight_val = None

        n_samples = X_train.shape[0]
        sample_idx = np.arange(n_samples, dtype=int)

        if self.batch_size == "auto":
            batch_size = min(200, n_samples)
        else:
            if self.batch_size > n_samples:
                warnings.warn(
                    "Got `batch_size` less than 1 or larger than "
                    "sample size. It is going to be clipped"
                )
            batch_size = np.clip(self.batch_size, 1, n_samples)

        try:
            self.n_iter_ = 0
            for it in range(self.max_iter):
                if self.shuffle:
                    # Only shuffle the sample indices instead of X and y to
                    # reduce the memory footprint. These indices will be used
                    # to slice the X and y.
                    sample_idx = shuffle(sample_idx, random_state=self._random_state)

                accumulated_loss = 0.0
                for batch_slice in gen_batches(n_samples, batch_size):
                    if self.shuffle:
                        batch_idx = sample_idx[batch_slice]
                        X_batch = _safe_indexing(X_train, batch_idx)
                    else:
                        batch_idx = batch_slice
                        X_batch = X_train[batch_idx]
                    y_batch = y_train[batch_idx]
                    if sample_weight is None:
                        sample_weight_batch = None
                    else:
                        sample_weight_batch = sample_weight_train[batch_idx]

                    activations[0] = X_batch
                    batch_loss, coef_grads, intercept_grads = self._backprop(
                        X_batch,
                        y_batch,
                        sample_weight_batch,
                        activations,
                        deltas,
                        coef_grads,
                        intercept_grads,
                    )
                    accumulated_loss += batch_loss * (
                        batch_slice.stop - batch_slice.start
                    )

                    # update weights
                    grads = coef_grads + intercept_grads
                    self._optimizer.update_params(params, grads)

                self.n_iter_ += 1
                self.loss_ = accumulated_loss / X_train.shape[0]

                self.t_ += n_samples
                self.loss_curve_.append(self.loss_)
                if self.verbose:
                    print("Iteration %d, loss = %.8f" % (self.n_iter_, self.loss_))

                # update no_improvement_count based on training loss or
                # validation score according to early_stopping
                self._update_no_improvement_count(
                    early_stopping, X_val, y_val, sample_weight_val
                )

                # for learning rate that needs to be updated at iteration end
                self._optimizer.iteration_ends(self.t_)

                if self._no_improvement_count > self.n_iter_no_change:
                    # not better than last `n_iter_no_change` iterations by tol
                    # stop or decrease learning rate
                    if early_stopping:
                        msg = (
                            "Validation score did not improve more than "
                            "tol=%f for %d consecutive epochs."
                            % (self.tol, self.n_iter_no_change)
                        )
                    else:
                        msg = (
                            "Training loss did not improve more than tol=%f"
                            " for %d consecutive epochs."
                            % (self.tol, self.n_iter_no_change)
                        )

                    is_stopping = self._optimizer.trigger_stopping(msg, self.verbose)
                    if is_stopping:
                        break
                    else:
                        self._no_improvement_count = 0

                if incremental:
                    break

                if self.n_iter_ == self.max_iter:
                    warnings.warn(
                        "Stochastic Optimizer: Maximum iterations (%d) "
                        "reached and the optimization hasn't converged yet."
                        % self.max_iter,
                        ConvergenceWarning,
                    )
        except KeyboardInterrupt:
            warnings.warn("Training interrupted by user.")

        if early_stopping:
            # restore best weights
            self.coefs_ = self._best_coefs
            self.intercepts_ = self._best_intercepts

    def _update_no_improvement_count(self, early_stopping, X, y, sample_weight):
        # [FUNCTION DOC]
        # HELPER: Updates the convergence counter after an epoch. With early_stopping:
        #   computes the validation score (via _score, may be NaN), appends it to
        #   validation_scores_, optionally prints it; increments _no_improvement_count
        #   if the score is below best_validation_score_ + tol, otherwise resets it; on
        #   a new best score stores copies of the coefficients/intercepts. Without early
        #   stopping: same logic on the last training loss versus best_loss_ - tol,
        #   updating best_loss_ on improvement.
        # Inputs: early_stopping bool; X, y, sample_weight - validation data (unused
        #   when not early stopping).
        # Side effects: mutates counters, best scores/losses and best-weight
        #   snapshots.
        # KNOWN ISSUE: a NaN validation score compares false against the threshold, so
        #   it resets the no-improvement counter instead of counting as no improvement.
        if early_stopping:
            # compute validation score (can be NaN), use that for stopping
            val_score = self._score(X, y, sample_weight=sample_weight)

            self.validation_scores_.append(val_score)

            if self.verbose:
                print("Validation score: %f" % self.validation_scores_[-1])
            # update best parameters
            # use validation_scores_, not loss_curve_
            # let's hope no-one overloads .score with mse
            last_valid_score = self.validation_scores_[-1]

            if last_valid_score < (self.best_validation_score_ + self.tol):
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0

            if last_valid_score > self.best_validation_score_:
                self.best_validation_score_ = last_valid_score
                self._best_coefs = [c.copy() for c in self.coefs_]
                self._best_intercepts = [i.copy() for i in self.intercepts_]
        else:
            if self.loss_curve_[-1] > self.best_loss_ - self.tol:
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0
            if self.loss_curve_[-1] < self.best_loss_:
                self.best_loss_ = self.loss_curve_[-1]

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        # [FUNCTION DOC]
        # Fits the network from scratch (or from the current weights when warm_start)
        #   on X, y with optional sample_weight by delegating to _fit with
        #   incremental=False.
        # Outputs: self.
        # Side effects: see _fit.
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

            .. versionadded:: 1.7

        Returns
        -------
        self : object
            Returns a trained MLP model.
        """
        return self._fit(X, y, sample_weight=sample_weight, incremental=False)

    def _check_solver(self):
        # [FUNCTION DOC]
        # HELPER: Availability check for partial_fit: returns True for sgd/adam.
        # Exceptions: AttributeError when solver is lbfgs (makes partial_fit appear
        #   absent).
        if self.solver not in _STOCHASTIC_SOLVERS:
            raise AttributeError(
                "partial_fit is only available for stochastic"
                " optimizers. %s is not stochastic." % self.solver
            )
        return True

    def _score_with_function(self, X, y, sample_weight, score_function):
        # [FUNCTION DOC]
        # HELPER: Scores predictions with the given metric without re-validating X (to
        #   keep feature names). Returns NaN when predictions are floating point and
        #   contain non-finite values.
        # Inputs: X, y, sample_weight; score_function - metric taking y_true, y_pred,
        #   sample_weight.
        # Outputs: float score or NaN.
        """Private score method without input validation."""
        # Input validation would remove feature names, so we disable it
        y_pred = self._predict(X, check_input=False)

        if np.issubdtype(y_pred.dtype, np.floating) and not np.isfinite(y_pred).all():
            return np.nan

        return score_function(y, y_pred, sample_weight=sample_weight)

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Declares sparse input support.
        # Outputs: Tags object.
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class MLPClassifier(ClassifierMixin, BaseMultilayerPerceptron):
    # [CLASS DOC]
    # Multi-layer perceptron classifier trained with log loss. Targets are binarized
    #   by an internal LabelBinarizer (_label_binarizer); the output layer is softmax
    #   for multiclass, logistic for binary and multilabel targets. Supports partial_fit
    #   only for sgd/adam.
    """Multi-layer Perceptron classifier.

    This model optimizes the log-loss function using LBFGS or stochastic
    gradient descent.

    .. versionadded:: 0.18

    Parameters
    ----------
    hidden_layer_sizes : array-like of shape(n_layers - 2,), default=(100,)
        The ith element represents the number of neurons in the ith
        hidden layer.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layer.

        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x

        - 'logistic', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).

        - 'tanh', the hyperbolic tan function,
          returns f(x) = tanh(x).

        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)

    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization.

        - 'lbfgs' is an optimizer in the family of quasi-Newton methods.

        - 'sgd' refers to stochastic gradient descent.

        - 'adam' refers to a stochastic gradient-based optimizer proposed
          by Kingma, Diederik, and Jimmy Ba

        For a comparison between Adam optimizer and SGD, see
        :ref:`sphx_glr_auto_examples_neural_networks_plot_mlp_training_curves.py`.

        Note: The default solver 'adam' works pretty well on relatively
        large datasets (with thousands of training samples or more) in terms of
        both training time and validation score.
        For small datasets, however, 'lbfgs' can converge faster and perform
        better.

    alpha : float, default=0.0001
        Strength of the L2 regularization term. The L2 regularization term
        is divided by the sample size when added to the loss.

        For an example usage and visualization of varying regularization, see
        :ref:`sphx_glr_auto_examples_neural_networks_plot_mlp_alpha.py`.

    batch_size : int, default='auto'
        Size of minibatches for stochastic optimizers.
        If the solver is 'lbfgs', the classifier will not use minibatch.
        When set to "auto", `batch_size=min(200, n_samples)`.

    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates.

        - 'constant' is a constant learning rate given by
          'learning_rate_init'.

        - 'invscaling' gradually decreases the learning rate at each
          time step 't' using an inverse scaling exponent of 'power_t'.
          effective_learning_rate = learning_rate_init / pow(t, power_t)

        - 'adaptive' keeps the learning rate constant to
          'learning_rate_init' as long as training loss keeps decreasing.
          Each time two consecutive epochs fail to decrease training loss by at
          least tol, or fail to increase validation score by at least tol if
          'early_stopping' is on, the current learning rate is divided by 5.

        Only used when ``solver='sgd'``.

    learning_rate_init : float, default=0.001
        The initial learning rate used. It controls the step-size
        in updating the weights. Only used when solver='sgd' or 'adam'.

    power_t : float, default=0.5
        The exponent for inverse scaling learning rate.
        It is used in updating effective learning rate when the learning_rate
        is set to 'invscaling'. Only used when solver='sgd'.

    max_iter : int, default=200
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations. For stochastic
        solvers ('sgd', 'adam'), note that this determines the number of epochs
        (how many times each data point will be used), not the number of
        gradient steps.

    shuffle : bool, default=True
        Whether to shuffle samples in each iteration. Only used when
        solver='sgd' or 'adam'.

    random_state : int, RandomState instance, default=None
        Determines random number generation for weights and bias
        initialization, train-test split if early stopping is used, and batch
        sampling when solver='sgd' or 'adam'.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    tol : float, default=1e-4
        Tolerance for the optimization. When the loss or score is not improving
        by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
        unless ``learning_rate`` is set to 'adaptive', convergence is
        considered to be reached and training stops.

    verbose : bool, default=False
        Whether to print progress messages to stdout.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous
        call to fit as initialization, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    momentum : float, default=0.9
        Momentum for gradient descent update. Should be between 0 and 1. Only
        used when solver='sgd'.

    nesterovs_momentum : bool, default=True
        Whether to use Nesterov's momentum. Only used when solver='sgd' and
        momentum > 0.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to True, it will automatically set
        aside ``validation_fraction`` of training data as validation and
        terminate training when validation score is not improving by at least
        ``tol`` for ``n_iter_no_change`` consecutive epochs. The split is
        stratified, except in a multilabel setting.
        If early stopping is False, then the training stops when the training
        loss does not improve by more than ``tol`` for ``n_iter_no_change``
        consecutive passes over the training set.
        Only effective when solver='sgd' or 'adam'.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True.

    beta_1 : float, default=0.9
        Exponential decay rate for estimates of first moment vector in adam,
        should be in [0, 1). Only used when solver='adam'.

    beta_2 : float, default=0.999
        Exponential decay rate for estimates of second moment vector in adam,
        should be in [0, 1). Only used when solver='adam'.

    epsilon : float, default=1e-8
        Value for numerical stability in adam. Only used when solver='adam'.

    n_iter_no_change : int, default=10
        Maximum number of epochs to not meet ``tol`` improvement.
        Only effective when solver='sgd' or 'adam'.

        .. versionadded:: 0.20

    max_fun : int, default=15000
        Only used when solver='lbfgs'. Maximum number of loss function calls.
        The solver iterates until convergence (determined by 'tol'), number
        of iterations reaches max_iter, or this number of loss function calls.
        Note that number of loss function calls will be greater than or equal
        to the number of iterations for the `MLPClassifier`.

        .. versionadded:: 0.22

    Attributes
    ----------
    classes_ : ndarray or list of ndarray of shape (n_classes,)
        Class labels for each output.

    loss_ : float
        The current loss computed with the loss function.

    best_loss_ : float or None
        The minimum loss reached by the solver throughout fitting.
        If `early_stopping=True`, this attribute is set to `None`. Refer to
        the `best_validation_score_` fitted attribute instead.

    loss_curve_ : list of shape (`n_iter_`,)
        The ith element in the list represents the loss at the ith iteration.

    validation_scores_ : list of shape (`n_iter_`,) or None
        The score at each iteration on a held-out validation set. The score
        reported is the accuracy score. Only available if `early_stopping=True`,
        otherwise the attribute is set to `None`.

    best_validation_score_ : float or None
        The best validation score (i.e. accuracy score) that triggered the
        early stopping. Only available if `early_stopping=True`, otherwise the
        attribute is set to `None`.

    t_ : int
        The number of training samples seen by the solver during fitting.

    coefs_ : list of shape (n_layers - 1,)
        The ith element in the list represents the weight matrix corresponding
        to layer i.

    intercepts_ : list of shape (n_layers - 1,)
        The ith element in the list represents the bias vector corresponding to
        layer i + 1.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        The number of iterations the solver has run.

    n_layers_ : int
        Number of layers.

    n_outputs_ : int
        Number of outputs.

    out_activation_ : str
        Name of the output activation function.

    See Also
    --------
    MLPRegressor : Multi-layer Perceptron regressor.
    BernoulliRBM : Bernoulli Restricted Boltzmann Machine (RBM).

    Notes
    -----
    MLPClassifier trains iteratively since at each time step
    the partial derivatives of the loss function with respect to the model
    parameters are computed to update the parameters.

    It can also have a regularization term added to the loss function
    that shrinks model parameters to prevent overfitting.

    This implementation works with data represented as dense numpy arrays or
    sparse scipy arrays of floating point values.

    References
    ----------
    Hinton, Geoffrey E. "Connectionist learning procedures."
    Artificial intelligence 40.1 (1989): 185-234.

    Glorot, Xavier, and Yoshua Bengio.
    "Understanding the difficulty of training deep feedforward neural networks."
    International Conference on Artificial Intelligence and Statistics. 2010.

    :arxiv:`He, Kaiming, et al (2015). "Delving deep into rectifiers:
    Surpassing human-level performance on imagenet classification." <1502.01852>`

    :arxiv:`Kingma, Diederik, and Jimmy Ba (2014)
    "Adam: A method for stochastic optimization." <1412.6980>`

    Examples
    --------
    >>> from sklearn.neural_network import MLPClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, random_state=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
    ...                                                     random_state=1)
    >>> clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    >>> clf.predict_proba(X_test[:1])
    array([[0.0383, 0.961]])
    >>> clf.predict(X_test[:5, :])
    array([1, 0, 1, 0, 1])
    >>> clf.score(X_test, y_test)
    0.8...
    """

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        *,
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
    ):
        # [FUNCTION DOC]
        # Stores hyperparameters via the base constructor with loss fixed to log_loss.
        #   Defaults: one hidden layer of 100 relu units, adam solver, alpha 1e-4,
        #   batch_size auto, max_iter 200, tol 1e-4, n_iter_no_change 10, max_fun 15000.
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            loss="log_loss",
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )

    def _validate_input(self, X, y, incremental, reset):
        # [FUNCTION DOC]
        # HELPER: Validates X (CSR/CSC, float64/float32) and y (multi-output allowed;
        #   single column ravelled with a warning). Fits a new LabelBinarizer and sets
        #   classes_ when classes_ is absent or when neither warm_start nor incremental.
        #   Otherwise checks label compatibility: with warm_start the class set must be
        #   identical; for incremental fits y must not contain unseen classes. Finally
        #   binarizes y and casts it to bool (to avoid upcasting float32 data).
        # Inputs: X, y; incremental bool; reset - whether to reset n_features_in_.
        # Outputs: (X, y_binarized 2-D bool array).
        # Exceptions: ValueError on class mismatches or invalid input.
        # Side effects: may set _label_binarizer and classes_.
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=["csr", "csc"],
            multi_output=True,
            dtype=(np.float64, np.float32),
            reset=reset,
        )
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)

        # Matrix of actions to be taken under the possible combinations:
        # The case that incremental == True and classes_ not defined is
        # already checked by _check_partial_fit_first_call that is called
        # in _partial_fit below.
        # The cases are already grouped into the respective if blocks below.
        #
        # incremental warm_start classes_ def  action
        #    0            0         0        define classes_
        #    0            1         0        define classes_
        #    0            0         1        redefine classes_
        #
        #    0            1         1        check compat warm_start
        #    1            1         1        check compat warm_start
        #
        #    1            0         1        check compat last fit
        #
        # Note the reliance on short-circuiting here, so that the second
        # or part implies that classes_ is defined.
        if (not hasattr(self, "classes_")) or (not self.warm_start and not incremental):
            self._label_binarizer = LabelBinarizer()
            self._label_binarizer.fit(y)
            self.classes_ = self._label_binarizer.classes_
        else:
            classes = unique_labels(y)
            if self.warm_start:
                if set(classes) != set(self.classes_):
                    raise ValueError(
                        "warm_start can only be used where `y` has the same "
                        "classes as in the previous call to fit. Previously "
                        f"got {self.classes_}, `y` has {classes}"
                    )
            elif len(np.setdiff1d(classes, self.classes_, assume_unique=True)):
                raise ValueError(
                    "`y` has classes not in `self.classes_`. "
                    f"`self.classes_` has {self.classes_}. 'y' has {classes}."
                )

        # This downcast to bool is to prevent upcasting when working with
        # float32 data
        y = self._label_binarizer.transform(y).astype(bool)
        return X, y

    def predict(self, X):
        # [FUNCTION DOC]
        # Predicts class labels for X.
        # Outputs: array (n_samples,) of labels, or (n_samples, n_classes) indicator
        #   matrix for multilabel.
        # Exceptions: NotFittedError if not fitted.
        """Predict using the multi-layer perceptron classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray, shape (n_samples,) or (n_samples, n_classes)
            The predicted classes.
        """
        check_is_fitted(self)
        return self._predict(X)

    def _predict(self, X, check_input=True):
        # [FUNCTION DOC]
        # HELPER: Forward pass then inverse label binarization (thresholding / argmax
        #   done by LabelBinarizer). Output is ravelled for single-output networks.
        # Inputs: X; check_input - validate X when True.
        """Private predict method with optional input validation"""
        y_pred = self._forward_pass_fast(X, check_input=check_input)

        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()

        return self._label_binarizer.inverse_transform(y_pred)

    def _score(self, X, y, sample_weight=None):
        # [FUNCTION DOC]
        # HELPER: Accuracy of predictions on X versus y, NaN if outputs are non-
        #   finite; used for early-stopping validation.
        return super()._score_with_function(
            X, y, sample_weight=sample_weight, score_function=accuracy_score
        )

    @available_if(lambda est: est._check_solver())
    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, sample_weight=None, classes=None):
        # [FUNCTION DOC]
        # One epoch of sgd/adam on the given data. On the first call classes must be
        #   provided (checked by _check_partial_fit_first_call, which also sets
        #   classes_); the label binarizer is fit on y for multilabel targets, otherwise
        #   on classes. Then delegates to _fit with incremental=True.
        # Inputs: X, y; sample_weight optional; classes - array of all labels,
        #   required on first call.
        # Outputs: self.
        # Exceptions: ValueError if classes is missing on the first call or
        #   inconsistent later, or with early_stopping; AttributeError (method
        #   unavailable) for the lbfgs solver.
        """Update the model with a single iteration over the given data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

            .. versionadded:: 1.7

        classes : array of shape (n_classes,), default=None
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.

        Returns
        -------
        self : object
            Trained MLP model.
        """
        if _check_partial_fit_first_call(self, classes):
            self._label_binarizer = LabelBinarizer()
            if type_of_target(y).startswith("multilabel"):
                self._label_binarizer.fit(y)
            else:
                self._label_binarizer.fit(classes)

        return self._fit(X, y, sample_weight=sample_weight, incremental=True)

    def predict_log_proba(self, X):
        # [FUNCTION DOC]
        # Natural log of predict_proba, computed in place on the probability array.
        # Outputs: ndarray (n_samples, n_classes); may contain -inf for zero
        #   probabilities.
        """Return the log of probability estimates.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        log_y_prob : ndarray of shape (n_samples, n_classes)
            The predicted log-probability of the sample for each class
            in the model, where classes are ordered as they are in
            `self.classes_`. Equivalent to `log(predict_proba(X))`.
        """
        y_prob = self.predict_proba(X)
        return np.log(y_prob, out=y_prob)

    def predict_proba(self, X):
        # [FUNCTION DOC]
        # Class probabilities from the forward pass. For a single logistic output the
        #   result is stacked as two columns [1 - p, p]; otherwise the output layer is
        #   returned as is (softmax rows, or independent logistic columns for
        #   multilabel).
        # Outputs: ndarray (n_samples, n_classes).
        # Exceptions: NotFittedError.
        """Probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_prob : ndarray of shape (n_samples, n_classes)
            The predicted probability of the sample for each class in the
            model, where classes are ordered as they are in `self.classes_`.
        """
        check_is_fitted(self)
        y_pred = self._forward_pass_fast(X)

        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()

        if y_pred.ndim == 1:
            return np.vstack([1 - y_pred, y_pred]).T
        else:
            return y_pred

    def __sklearn_tags__(self):
        # [FUNCTION DOC]
        # Adds the multi_label classifier tag on top of the base tags.
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_label = True
        return tags


class MLPRegressor(RegressorMixin, BaseMultilayerPerceptron):
    # [CLASS DOC]
    # Multi-layer perceptron regressor. loss squared_error uses an identity output
    #   layer; poisson uses an exp output layer. Multi-output targets are supported.
    """Multi-layer Perceptron regressor.

    This model optimizes the squared error using LBFGS or stochastic gradient
    descent.

    .. versionadded:: 0.18

    Parameters
    ----------
    loss : {'squared_error', 'poisson'}, default='squared_error'
        The loss function to use when training the weights. Note that the
        "squared error" and "poisson" losses actually implement
        "half squares error" and "half poisson deviance" to simplify the
        computation of the gradient. Furthermore, the "poisson" loss internally uses
        a log-link (exponential as the output activation function) and requires
        ``y >= 0``.

        .. versionchanged:: 1.7
           Added parameter `loss` and option 'poisson'.

    hidden_layer_sizes : array-like of shape(n_layers - 2,), default=(100,)
        The ith element represents the number of neurons in the ith
        hidden layer.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layer.

        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x

        - 'logistic', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).

        - 'tanh', the hyperbolic tan function,
          returns f(x) = tanh(x).

        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)

    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization.

        - 'lbfgs' is an optimizer in the family of quasi-Newton methods.

        - 'sgd' refers to stochastic gradient descent.

        - 'adam' refers to a stochastic gradient-based optimizer proposed by
          Kingma, Diederik, and Jimmy Ba

        For a comparison between Adam optimizer and SGD, see
        :ref:`sphx_glr_auto_examples_neural_networks_plot_mlp_training_curves.py`.

        Note: The default solver 'adam' works pretty well on relatively
        large datasets (with thousands of training samples or more) in terms of
        both training time and validation score.
        For small datasets, however, 'lbfgs' can converge faster and perform
        better.

    alpha : float, default=0.0001
        Strength of the L2 regularization term. The L2 regularization term
        is divided by the sample size when added to the loss.

    batch_size : int, default='auto'
        Size of minibatches for stochastic optimizers.
        If the solver is 'lbfgs', the regressor will not use minibatch.
        When set to "auto", `batch_size=min(200, n_samples)`.

    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates.

        - 'constant' is a constant learning rate given by
          'learning_rate_init'.

        - 'invscaling' gradually decreases the learning rate ``learning_rate_``
          at each time step 't' using an inverse scaling exponent of 'power_t'.
          effective_learning_rate = learning_rate_init / pow(t, power_t)

        - 'adaptive' keeps the learning rate constant to
          'learning_rate_init' as long as training loss keeps decreasing.
          Each time two consecutive epochs fail to decrease training loss by at
          least tol, or fail to increase validation score by at least tol if
          'early_stopping' is on, the current learning rate is divided by 5.

        Only used when solver='sgd'.

    learning_rate_init : float, default=0.001
        The initial learning rate used. It controls the step-size
        in updating the weights. Only used when solver='sgd' or 'adam'.

    power_t : float, default=0.5
        The exponent for inverse scaling learning rate.
        It is used in updating effective learning rate when the learning_rate
        is set to 'invscaling'. Only used when solver='sgd'.

    max_iter : int, default=200
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations. For stochastic
        solvers ('sgd', 'adam'), note that this determines the number of epochs
        (how many times each data point will be used), not the number of
        gradient steps.

    shuffle : bool, default=True
        Whether to shuffle samples in each iteration. Only used when
        solver='sgd' or 'adam'.

    random_state : int, RandomState instance, default=None
        Determines random number generation for weights and bias
        initialization, train-test split if early stopping is used, and batch
        sampling when solver='sgd' or 'adam'.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    tol : float, default=1e-4
        Tolerance for the optimization. When the loss or score is not improving
        by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
        unless ``learning_rate`` is set to 'adaptive', convergence is
        considered to be reached and training stops.

    verbose : bool, default=False
        Whether to print progress messages to stdout.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous
        call to fit as initialization, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    momentum : float, default=0.9
        Momentum for gradient descent update. Should be between 0 and 1. Only
        used when solver='sgd'.

    nesterovs_momentum : bool, default=True
        Whether to use Nesterov's momentum. Only used when solver='sgd' and
        momentum > 0.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to True, it will automatically set
        aside ``validation_fraction`` of training data as validation and
        terminate training when validation score is not improving by at
        least ``tol`` for ``n_iter_no_change`` consecutive epochs.
        Only effective when solver='sgd' or 'adam'.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True.

    beta_1 : float, default=0.9
        Exponential decay rate for estimates of first moment vector in adam,
        should be in [0, 1). Only used when solver='adam'.

    beta_2 : float, default=0.999
        Exponential decay rate for estimates of second moment vector in adam,
        should be in [0, 1). Only used when solver='adam'.

    epsilon : float, default=1e-8
        Value for numerical stability in adam. Only used when solver='adam'.

    n_iter_no_change : int, default=10
        Maximum number of epochs to not meet ``tol`` improvement.
        Only effective when solver='sgd' or 'adam'.

        .. versionadded:: 0.20

    max_fun : int, default=15000
        Only used when solver='lbfgs'. Maximum number of function calls.
        The solver iterates until convergence (determined by ``tol``), number
        of iterations reaches max_iter, or this number of function calls.
        Note that number of function calls will be greater than or equal to
        the number of iterations for the MLPRegressor.

        .. versionadded:: 0.22

    Attributes
    ----------
    loss_ : float
        The current loss computed with the loss function.

    best_loss_ : float
        The minimum loss reached by the solver throughout fitting.
        If `early_stopping=True`, this attribute is set to `None`. Refer to
        the `best_validation_score_` fitted attribute instead.
        Only accessible when solver='sgd' or 'adam'.

    loss_curve_ : list of shape (`n_iter_`,)
        Loss value evaluated at the end of each training step.
        The ith element in the list represents the loss at the ith iteration.
        Only accessible when solver='sgd' or 'adam'.

    validation_scores_ : list of shape (`n_iter_`,) or None
        The score at each iteration on a held-out validation set. The score
        reported is the R2 score. Only available if `early_stopping=True`,
        otherwise the attribute is set to `None`.
        Only accessible when solver='sgd' or 'adam'.

    best_validation_score_ : float or None
        The best validation score (i.e. R2 score) that triggered the
        early stopping. Only available if `early_stopping=True`, otherwise the
        attribute is set to `None`.
        Only accessible when solver='sgd' or 'adam'.

    t_ : int
        The number of training samples seen by the solver during fitting.
        Mathematically equals `n_iters * X.shape[0]`, it means
        `time_step` and it is used by optimizer's learning rate scheduler.

    coefs_ : list of shape (n_layers - 1,)
        The ith element in the list represents the weight matrix corresponding
        to layer i.

    intercepts_ : list of shape (n_layers - 1,)
        The ith element in the list represents the bias vector corresponding to
        layer i + 1.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        The number of iterations the solver has run.

    n_layers_ : int
        Number of layers.

    n_outputs_ : int
        Number of outputs.

    out_activation_ : str
        Name of the output activation function.

    See Also
    --------
    BernoulliRBM : Bernoulli Restricted Boltzmann Machine (RBM).
    MLPClassifier : Multi-layer Perceptron classifier.
    sklearn.linear_model.SGDRegressor : Linear model fitted by minimizing
        a regularized empirical loss with SGD.

    Notes
    -----
    MLPRegressor trains iteratively since at each time step
    the partial derivatives of the loss function with respect to the model
    parameters are computed to update the parameters.

    It can also have a regularization term added to the loss function
    that shrinks model parameters to prevent overfitting.

    This implementation works with data represented as dense and sparse numpy
    arrays of floating point values.

    References
    ----------
    Hinton, Geoffrey E. "Connectionist learning procedures."
    Artificial intelligence 40.1 (1989): 185-234.

    Glorot, Xavier, and Yoshua Bengio.
    "Understanding the difficulty of training deep feedforward neural networks."
    International Conference on Artificial Intelligence and Statistics. 2010.

    :arxiv:`He, Kaiming, et al (2015). "Delving deep into rectifiers:
    Surpassing human-level performance on imagenet classification." <1502.01852>`

    :arxiv:`Kingma, Diederik, and Jimmy Ba (2014)
    "Adam: A method for stochastic optimization." <1412.6980>`

    Examples
    --------
    >>> from sklearn.neural_network import MLPRegressor
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_regression(n_samples=200, n_features=20, random_state=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=1)
    >>> regr = MLPRegressor(random_state=1, max_iter=2000, tol=0.1)
    >>> regr.fit(X_train, y_train)
    MLPRegressor(max_iter=2000, random_state=1, tol=0.1)
    >>> regr.predict(X_test[:2])
    array([  28.98, -291])
    >>> regr.score(X_test, y_test)
    0.98
    """

    _parameter_constraints: dict = {
        **BaseMultilayerPerceptron._parameter_constraints,
        "loss": [StrOptions({"squared_error", "poisson"})],
    }

    def __init__(
        self,
        loss="squared_error",
        hidden_layer_sizes=(100,),
        activation="relu",
        *,
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
    ):
        # [FUNCTION DOC]
        # Stores hyperparameters via the base constructor; loss is the first
        #   positional parameter (default squared_error). Other defaults match
        #   MLPClassifier.
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            loss=loss,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )

    def predict(self, X):
        # [FUNCTION DOC]
        # Predicts targets for X; output is (n_samples,) for single-output models,
        #   otherwise (n_samples, n_outputs).
        # Exceptions: NotFittedError.
        """Predict using the multi-layer perceptron model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self)
        return self._predict(X)

    def _predict(self, X, check_input=True):
        # [FUNCTION DOC]
        # HELPER: Forward pass; ravels the single-output case.
        # Inputs: X; check_input bool.
        """Private predict method with optional input validation"""
        y_pred = self._forward_pass_fast(X, check_input=check_input)
        if y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred

    def _score(self, X, y, sample_weight=None):
        # [FUNCTION DOC]
        # HELPER: R2 score of predictions on X versus y, NaN if outputs are non-
        #   finite; used for early-stopping validation.
        return super()._score_with_function(
            X, y, sample_weight=sample_weight, score_function=r2_score
        )

    def _validate_input(self, X, y, incremental, reset):
        # [FUNCTION DOC]
        # HELPER: Validates X (CSR/CSC, float64/float32) and numeric y (multi-output
        #   allowed; single column ravelled with a warning).
        # Outputs: (X, y). Exceptions: ValueError on invalid input.
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=["csr", "csc"],
            multi_output=True,
            y_numeric=True,
            dtype=(np.float64, np.float32),
            reset=reset,
        )
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)
        return X, y

    @available_if(lambda est: est._check_solver)
    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, sample_weight=None):
        # [FUNCTION DOC]
        # One epoch of sgd/adam on the given data via _fit with incremental=True.
        # Inputs: X, y, optional sample_weight.
        # Outputs: self.
        # KNOWN ISSUE: the available_if check passes the bound method _check_solver
        #   without calling it, so the condition is always truthy and partial_fit is
        #   exposed even for solver lbfgs (unlike MLPClassifier, which calls it).
        """Update the model with a single iteration over the given data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,)
            The target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

            .. versionadded:: 1.6

        Returns
        -------
        self : object
            Trained MLP model.
        """
        return self._fit(X, y, sample_weight=sample_weight, incremental=True)
