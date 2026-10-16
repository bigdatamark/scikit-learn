# [MODULE DOC]
# HELPER: Test utility module defining MiniSeqKernel, a tiny kernel over variable-length
# sequences (e.g. strings) used to check that GP estimators accept non-vector inputs.
import numpy as np

from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
    GenericKernelMixin,
    Hyperparameter,
    Kernel,
    StationaryKernelMixin,
)


class MiniSeqKernel(GenericKernelMixin, StationaryKernelMixin, Kernel):
    # [CLASS DOC]
    # HELPER: Generic, stationary kernel on sequences. For two sequences, sums over all
    # pairs of elements: 1 for equal elements, baseline_similarity otherwise.
    """
    A minimal (but valid) convolutional kernel for sequences of variable
    length.
    """

    def __init__(self, baseline_similarity=0.5, baseline_similarity_bounds=(1e-5, 1)):
        # [FUNCTION DOC]
        # Stores baseline_similarity (float, default 0.5) and its bounds (default (1e-5,
        # 1)).
        self.baseline_similarity = baseline_similarity
        self.baseline_similarity_bounds = baseline_similarity_bounds

    @property
    def hyperparameter_baseline_similarity(self):
        # [FUNCTION DOC]
        # Hyperparameter spec for baseline_similarity.
        return Hyperparameter(
            "baseline_similarity", "numeric", self.baseline_similarity_bounds
        )

    def _f(self, s1, s2):
        # [FUNCTION DOC]
        # HELPER: Kernel value between sequences s1 and s2: over every element pair, add
        # 1.0 if equal else baseline_similarity. Outputs: float.
        return sum(
            [1.0 if c1 == c2 else self.baseline_similarity for c1 in s1 for c2 in s2]
        )

    def _g(self, s1, s2):
        # [FUNCTION DOC]
        # HELPER: Derivative of _f with respect to baseline_similarity: number of
        # unequal element pairs. Outputs: float.
        return sum([0.0 if c1 == c2 else 1.0 for c1 in s1 for c2 in s2])

    def __call__(self, X, Y=None, eval_gradient=False):
        # [FUNCTION DOC]
        # Builds the kernel matrix by applying _f to every pair (x in X, y in Y); Y
        # defaults to X.
        # Inputs: X, Y - sequences of sequences; eval_gradient - bool.
        # Outputs: K (n_X, n_Y) ndarray; if eval_gradient, also gradient array (n_X,
        # n_Y, 1) from _g.
        # KNOWN ISSUE: the gradient is with respect to baseline_similarity itself, not
        # its log as the Kernel API expects (would need multiplying by
        # baseline_similarity); it is also returned with one column even when the
        # hyperparameter is fixed, and Y is accepted with eval_gradient=True.
        if Y is None:
            Y = X

        if eval_gradient:
            return (
                np.array([[self._f(x, y) for y in Y] for x in X]),
                np.array([[[self._g(x, y)] for y in Y] for x in X]),
            )
        else:
            return np.array([[self._f(x, y) for y in Y] for x in X])

    def diag(self, X):
        # [FUNCTION DOC]
        # Diagonal: _f of each sequence with itself. Outputs: 1D ndarray (n_X,).
        return np.array([self._f(x, x) for x in X])

    def clone_with_theta(self, theta):
        # [FUNCTION DOC]
        # Returns a sklearn clone with theta set (same as the base Kernel
        # implementation).
        cloned = clone(self)
        cloned.theta = theta
        return cloned
