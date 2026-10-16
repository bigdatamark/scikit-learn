# [MODULE DOC]
# HELPER: Small utility module for ARPACK-based eigensolvers (scipy eigsh/svds), providing
# a reproducible starting vector consistent with ARPACK's own initialisation.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.utils.validation import check_random_state


def _init_arpack_v0(size, random_state):
    # [FUNCTION DOC]
    # HELPER: Draws a starting vector for ARPACK iterations from Uniform[-1, 1].
    # Inputs: size - int length of the vector; random_state - None, int seed or RandomState,
    # normalised through check_random_state.
    # Outputs: float64 ndarray of shape (size,).
    # Exceptions: ValueError from check_random_state for an invalid random_state.
    # Side effects: advances the state of a passed RandomState instance (or the global one
    # when None).
    # KNOWN ISSUE: the docstring documents a default of None for random_state, but the
    # parameter has no default and must be passed.
    """Initialize the starting vector for iteration in ARPACK functions.

    Initialize an ndarray with values sampled from the uniform distribution on
    [-1, 1]. This initialization model has been chosen to be consistent with
    the ARPACK one as another initialization can lead to convergence issues.

    Parameters
    ----------
    size : int
        The size of the eigenvalue vector to be initialized.

    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator used to generate a
        uniform distribution. If int, random_state is the seed used by the
        random number generator; If RandomState instance, random_state is the
        random number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    Returns
    -------
    v0 : ndarray of shape (size,)
        The initialized vector.
    """
    random_state = check_random_state(random_state)
    v0 = random_state.uniform(-1, 1, size)
    return v0
