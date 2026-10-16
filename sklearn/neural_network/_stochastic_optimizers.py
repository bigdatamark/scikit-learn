# [MODULE DOC]
# HELPER: Stochastic optimizers used internally by the MLP estimators (sgd and adam
#   solvers).
# Defines BaseOptimizer (applies additive updates in place and provides learning rate
#   / stopping hooks), SGDOptimizer (momentum and Nesterov momentum with constant,
#   invscaling or adaptive schedules) and AdamOptimizer (bias-corrected Adam).
# Side effects at import: none.
"""Stochastic optimization methods for MLP"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


class BaseOptimizer:
    # [CLASS DOC]
    # HELPER: Abstract base of the MLP optimizers. Subclasses must implement
    #   _get_updates(grads) returning one additive update array per parameter.
    # State: learning_rate_init (as given) and learning_rate (current float value).
    """Base (Stochastic) gradient descent optimizer

    Parameters
    ----------
    learning_rate_init : float, default=0.1
        The initial learning rate used. It controls the step-size in updating
        the weights

    Attributes
    ----------
    learning_rate : float
        the current learning rate
    """

    def __init__(self, learning_rate_init=0.1):
        # [FUNCTION DOC]
        # Stores learning_rate_init and initialises learning_rate as its float
        #   conversion.
        # Inputs: learning_rate_init - positive float, default 0.1.
        # Side effects: sets self.learning_rate_init and self.learning_rate.
        self.learning_rate_init = learning_rate_init
        self.learning_rate = float(learning_rate_init)

    def update_params(self, params, grads):
        # [FUNCTION DOC]
        # Computes updates via the subclass _get_updates and adds each update in place
        #   to the matching parameter array.
        # Inputs: params - list of numpy arrays (coefs then intercepts), mutated in
        #   place; grads - list of arrays with the same shapes, same order.
        # Outputs: None.
        # Side effects: params arrays are modified in place; subclass state
        #   (velocities, moments) is updated.
        # Exceptions: NotImplementedError-like AttributeError if a subclass does not
        #   define _get_updates.
        """Update parameters with given gradients

        Parameters
        ----------
        params : list of length = len(coefs_) + len(intercepts_)
            The concatenated list containing coefs_ and intercepts_ in MLP
            model. Used for initializing velocities and updating params

        grads : list of length = len(params)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params
        """
        updates = self._get_updates(grads)
        for param, update in zip((p for p in params), updates):
            param += update

    def iteration_ends(self, time_step):
        # [FUNCTION DOC]
        # Hook called at the end of each epoch; no-op in the base class.
        # Inputs: time_step - number of samples seen so far (unused here).
        """Perform update to learning rate and potentially other states at the
        end of an iteration
        """
        pass

    def trigger_stopping(self, msg, verbose):
        # [FUNCTION DOC]
        # Decides whether training should stop when the convergence criterion fires.
        #   The base class always stops.
        # Inputs: msg - message prefix; verbose - bool, print the message plus "
        #   Stopping." to stdout when true.
        # Outputs: True.
        # Side effects: optional print to stdout (docstring says stdin, it is stdout).
        """Decides whether it is time to stop training

        Parameters
        ----------
        msg : str
            Message passed in for verbose output

        verbose : bool
            Print message to stdin if True

        Returns
        -------
        is_stopping : bool
            True if training needs to stop
        """
        if verbose:
            print(msg + " Stopping.")
        return True


class SGDOptimizer(BaseOptimizer):
    # [CLASS DOC]
    # HELPER: SGD with classical or Nesterov momentum. Keeps one velocity array per
    #   parameter (initialised to zeros) and supports constant, invscaling and adaptive
    #   learning rate schedules.
    """Stochastic gradient descent optimizer with momentum

    Parameters
    ----------
    params : list, length = len(coefs_) + len(intercepts_)
        The concatenated list containing coefs_ and intercepts_ in MLP model.
        Used for initializing velocities and updating params

    learning_rate_init : float, default=0.1
        The initial learning rate used. It controls the step-size in updating
        the weights

    lr_schedule : {'constant', 'adaptive', 'invscaling'}, default='constant'
        Learning rate schedule for weight updates.

        -'constant', is a constant learning rate given by
         'learning_rate_init'.

        -'invscaling' gradually decreases the learning rate 'learning_rate_' at
          each time step 't' using an inverse scaling exponent of 'power_t'.
          learning_rate_ = learning_rate_init / pow(t, power_t)

        -'adaptive', keeps the learning rate constant to
         'learning_rate_init' as long as the training keeps decreasing.
         Each time 2 consecutive epochs fail to decrease the training loss by
         tol, or fail to increase validation score by tol if 'early_stopping'
         is on, the current learning rate is divided by 5.

    momentum : float, default=0.9
        Value of momentum used, must be larger than or equal to 0

    nesterov : bool, default=True
        Whether to use nesterov's momentum or not. Use nesterov's if True

    power_t : float, default=0.5
        Power of time step 't' in inverse scaling. See `lr_schedule` for
        more details.

    Attributes
    ----------
    learning_rate : float
        the current learning rate

    velocities : list, length = len(params)
        velocities that are used to update params
    """

    def __init__(
        self,
        params,
        learning_rate_init=0.1,
        lr_schedule="constant",
        momentum=0.9,
        nesterov=True,
        power_t=0.5,
    ):
        # [FUNCTION DOC]
        # Stores hyperparameters and allocates zero velocity arrays shaped like each
        #   parameter.
        # Inputs: params - list of arrays (only shapes/dtypes used);
        #   learning_rate_init float (0.1); lr_schedule one of
        #   constant/adaptive/invscaling; momentum float >= 0 (0.9); nesterov bool
        #   (True); power_t float (0.5).
        # Side effects: sets lr_schedule, momentum, nesterov, power_t, velocities on
        #   self. No validation is done.
        super().__init__(learning_rate_init)

        self.lr_schedule = lr_schedule
        self.momentum = momentum
        self.nesterov = nesterov
        self.power_t = power_t
        self.velocities = [np.zeros_like(param) for param in params]

    def iteration_ends(self, time_step):
        # [FUNCTION DOC]
        # For the invscaling schedule, sets learning_rate to learning_rate_init
        #   divided by (time_step + 1) raised to power_t; other schedules leave it
        #   unchanged.
        # Inputs: time_step - int, number of samples processed so far.
        # Side effects: may update self.learning_rate.
        """Perform updates to learning rate and potential other states at the
        end of an iteration

        Parameters
        ----------
        time_step : int
            number of training samples trained on so far, used to update
            learning rate for 'invscaling'
        """
        if self.lr_schedule == "invscaling":
            self.learning_rate = (
                float(self.learning_rate_init) / (time_step + 1) ** self.power_t
            )

    def trigger_stopping(self, msg, verbose):
        # [FUNCTION DOC]
        # Stopping decision. For non-adaptive schedules always returns True. For
        #   adaptive: if learning_rate is already <= 1e-6 return True; otherwise divide
        #   learning_rate by 5 and return False so training continues.
        # Inputs: msg - message prefix; verbose - print a status message to stdout
        #   when true.
        # Outputs: bool, True if training should stop.
        # Side effects: may divide self.learning_rate by 5; optional print.
        if self.lr_schedule != "adaptive":
            if verbose:
                print(msg + " Stopping.")
            return True

        if self.learning_rate <= 1e-6:
            if verbose:
                print(msg + " Learning rate too small. Stopping.")
            return True

        self.learning_rate /= 5.0
        if verbose:
            print(msg + " Setting learning rate to %f" % self.learning_rate)
        return False

    def _get_updates(self, grads):
        # [FUNCTION DOC]
        # Computes momentum updates. For each parameter the new velocity is momentum
        #   times the old velocity minus learning_rate times the gradient; these
        #   velocities are stored. With nesterov, the returned update is recomputed as
        #   momentum times the new velocity minus learning_rate times the gradient
        #   (look-ahead); otherwise the velocity itself is returned.
        # Inputs: grads - list of arrays aligned with the parameters.
        # Outputs: list of update arrays, same shapes as grads.
        # Side effects: replaces self.velocities.
        """Get the values used to update params with given gradients

        Parameters
        ----------
        grads : list, length = len(coefs_) + len(intercepts_)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params

        Returns
        -------
        updates : list, length = len(grads)
            The values to add to params
        """
        updates = [
            self.momentum * velocity - self.learning_rate * grad
            for velocity, grad in zip(self.velocities, grads)
        ]
        self.velocities = updates

        if self.nesterov:
            updates = [
                self.momentum * velocity - self.learning_rate * grad
                for velocity, grad in zip(self.velocities, grads)
            ]

        return updates


class AdamOptimizer(BaseOptimizer):
    # [CLASS DOC]
    # HELPER: Adam optimizer (Kingma and Ba 2014). Keeps a step counter t and
    #   first/second moment arrays ms and vs, all zero-initialised.
    """Stochastic gradient descent optimizer with Adam

    Note: All default values are from the original Adam paper

    Parameters
    ----------
    params : list, length = len(coefs_) + len(intercepts_)
        The concatenated list containing coefs_ and intercepts_ in MLP model.
        Used for initializing velocities and updating params

    learning_rate_init : float, default=0.001
        The initial learning rate used. It controls the step-size in updating
        the weights

    beta_1 : float, default=0.9
        Exponential decay rate for estimates of first moment vector, should be
        in [0, 1)

    beta_2 : float, default=0.999
        Exponential decay rate for estimates of second moment vector, should be
        in [0, 1)

    epsilon : float, default=1e-8
        Value for numerical stability

    Attributes
    ----------
    learning_rate : float
        The current learning rate

    t : int
        Timestep

    ms : list, length = len(params)
        First moment vectors

    vs : list, length = len(params)
        Second moment vectors

    References
    ----------
    :arxiv:`Kingma, Diederik, and Jimmy Ba (2014) "Adam: A method for
        stochastic optimization." <1412.6980>
    """

    def __init__(
        self, params, learning_rate_init=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8
    ):
        # [FUNCTION DOC]
        # Stores beta_1, beta_2, epsilon, sets t to 0 and allocates zero first and
        #   second moment arrays shaped like each parameter.
        # Inputs: params - list of arrays; learning_rate_init (0.001); beta_1 (0.9)
        #   and beta_2 (0.999) in [0, 1); epsilon (1e-8). No validation.
        # Side effects: sets attributes on self.
        super().__init__(learning_rate_init)

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.ms = [np.zeros_like(param) for param in params]
        self.vs = [np.zeros_like(param) for param in params]

    def _get_updates(self, grads):
        # [FUNCTION DOC]
        # One Adam step: increment t; update each first moment as beta_1 * m + (1 -
        #   beta_1) * grad and each second moment as beta_2 * v + (1 - beta_2) * grad
        #   squared; set learning_rate to learning_rate_init times sqrt(1 - beta_2^t)
        #   divided by (1 - beta_1^t) (folded bias correction); each update is minus
        #   learning_rate times m divided by (sqrt(v) + epsilon).
        # Inputs: grads - list of arrays aligned with parameters.
        # Outputs: list of update arrays.
        # Side effects: mutates self.t, self.ms, self.vs and self.learning_rate.
        """Get the values used to update params with given gradients

        Parameters
        ----------
        grads : list, length = len(coefs_) + len(intercepts_)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params

        Returns
        -------
        updates : list, length = len(grads)
            The values to add to params
        """
        self.t += 1
        self.ms = [
            self.beta_1 * m + (1 - self.beta_1) * grad
            for m, grad in zip(self.ms, grads)
        ]
        self.vs = [
            self.beta_2 * v + (1 - self.beta_2) * (grad**2)
            for v, grad in zip(self.vs, grads)
        ]
        self.learning_rate = (
            self.learning_rate_init
            * np.sqrt(1 - self.beta_2**self.t)
            / (1 - self.beta_1**self.t)
        )
        updates = [
            -self.learning_rate * m / (np.sqrt(v) + self.epsilon)
            for m, v in zip(self.ms, self.vs)
        ]
        return updates
