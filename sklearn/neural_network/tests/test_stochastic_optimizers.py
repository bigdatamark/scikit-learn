# [MODULE DOC]
# Unit tests for the MLP stochastic optimizers (BaseOptimizer, SGDOptimizer,
#   AdamOptimizer). Each test applies one update to zero-initialised parameters of
#   shapes in the module-level shapes list and compares against a hand-computed expected
#   result.
import numpy as np

from sklearn.neural_network._stochastic_optimizers import (
    AdamOptimizer,
    BaseOptimizer,
    SGDOptimizer,
)
from sklearn.utils._testing import assert_array_equal

shapes = [(4, 6), (6, 8), (7, 8, 9)]


def test_base_optimizer():
    # [FUNCTION DOC]
    # Checks that BaseOptimizer.trigger_stopping always returns True for learning
    #   rates from 1e-3 to 1e3.
    for lr in [10**i for i in range(-3, 4)]:
        optimizer = BaseOptimizer(lr)
        assert optimizer.trigger_stopping("", False)


def test_sgd_optimizer_no_momentum():
    # [FUNCTION DOC]
    # Checks that SGD with momentum 0 and no Nesterov performs plain gradient descent:
    #   after update_params each parameter equals its previous value minus lr times the
    #   random gradient, for several learning rates. The params list is reused and
    #   mutated across iterations.
    params = [np.zeros(shape) for shape in shapes]
    rng = np.random.RandomState(0)

    for lr in [10**i for i in range(-3, 4)]:
        optimizer = SGDOptimizer(params, lr, momentum=0, nesterov=False)
        grads = [rng.random_sample(shape) for shape in shapes]
        expected = [param - lr * grad for param, grad in zip(params, grads)]
        optimizer.update_params(params, grads)

        for exp, param in zip(expected, params):
            assert_array_equal(exp, param)


def test_sgd_optimizer_momentum():
    # [FUNCTION DOC]
    # Checks classical momentum: with velocities set manually to random arrays, the
    #   parameter update must equal momentum * velocity - lr * grad, for momentum values
    #   0.5 to 0.8.
    params = [np.zeros(shape) for shape in shapes]
    lr = 0.1
    rng = np.random.RandomState(0)

    for momentum in np.arange(0.5, 0.9, 0.1):
        optimizer = SGDOptimizer(params, lr, momentum=momentum, nesterov=False)
        velocities = [rng.random_sample(shape) for shape in shapes]
        optimizer.velocities = velocities
        grads = [rng.random_sample(shape) for shape in shapes]
        updates = [
            momentum * velocity - lr * grad for velocity, grad in zip(velocities, grads)
        ]
        expected = [param + update for param, update in zip(params, updates)]
        optimizer.update_params(params, grads)

        for exp, param in zip(expected, params):
            assert_array_equal(exp, param)


def test_sgd_optimizer_trigger_stopping():
    # [FUNCTION DOC]
    # Checks the adaptive schedule: with lr 2e-6 the first trigger_stopping call
    #   divides the rate by 5 and returns False; the second call (rate now below 1e-6)
    #   returns True.
    params = [np.zeros(shape) for shape in shapes]
    lr = 2e-6
    optimizer = SGDOptimizer(params, lr, lr_schedule="adaptive")
    assert not optimizer.trigger_stopping("", False)
    assert lr / 5 == optimizer.learning_rate
    assert optimizer.trigger_stopping("", False)


def test_sgd_optimizer_nesterovs_momentum():
    # [FUNCTION DOC]
    # Checks Nesterov momentum: the applied update must equal momentum times the new
    #   velocity minus lr * grad, where the new velocity is momentum * old velocity - lr
    #   * grad.
    params = [np.zeros(shape) for shape in shapes]
    lr = 0.1
    rng = np.random.RandomState(0)

    for momentum in np.arange(0.5, 0.9, 0.1):
        optimizer = SGDOptimizer(params, lr, momentum=momentum, nesterov=True)
        velocities = [rng.random_sample(shape) for shape in shapes]
        optimizer.velocities = velocities
        grads = [rng.random_sample(shape) for shape in shapes]
        updates = [
            momentum * velocity - lr * grad for velocity, grad in zip(velocities, grads)
        ]
        updates = [
            momentum * update - lr * grad for update, grad in zip(updates, grads)
        ]
        expected = [param + update for param, update in zip(params, updates)]
        optimizer.update_params(params, grads)

        for exp, param in zip(expected, params):
            assert_array_equal(exp, param)


def test_adam_optimizer():
    # [FUNCTION DOC]
    # Checks one Adam step against a manual computation over a grid of beta_1 and
    #   beta_2 values, with moments and step counter (t = 9 before the step) injected
    #   into the optimizer; expects exact equality of the updated parameters.
    params = [np.zeros(shape) for shape in shapes]
    lr = 0.001
    epsilon = 1e-8
    rng = np.random.RandomState(0)

    for beta_1 in np.arange(0.9, 1.0, 0.05):
        for beta_2 in np.arange(0.995, 1.0, 0.001):
            optimizer = AdamOptimizer(params, lr, beta_1, beta_2, epsilon)
            ms = [rng.random_sample(shape) for shape in shapes]
            vs = [rng.random_sample(shape) for shape in shapes]
            t = 10
            optimizer.ms = ms
            optimizer.vs = vs
            optimizer.t = t - 1
            grads = [rng.random_sample(shape) for shape in shapes]

            ms = [beta_1 * m + (1 - beta_1) * grad for m, grad in zip(ms, grads)]
            vs = [beta_2 * v + (1 - beta_2) * (grad**2) for v, grad in zip(vs, grads)]
            learning_rate = lr * np.sqrt(1 - beta_2**t) / (1 - beta_1**t)
            updates = [
                -learning_rate * m / (np.sqrt(v) + epsilon) for m, v in zip(ms, vs)
            ]
            expected = [param + update for param, update in zip(params, updates)]

            optimizer.update_params(params, grads)
            for exp, param in zip(expected, params):
                assert_array_equal(exp, param)
