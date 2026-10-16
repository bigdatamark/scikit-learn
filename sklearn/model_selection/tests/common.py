# [MODULE DOC]
# HELPER: Shared test utilities for model_selection tests. Currently provides a
# CV splitter that can only be iterated once, used to check that consumers do not
# iterate cv.split more than once.
"""
Common utilities for testing model selection.
"""

import numpy as np

from sklearn.model_selection import KFold


class OneTimeSplitter:
    # [CLASS DOC]
    # HELPER: Single-use CV splitter wrapping KFold. The KFold split generator over a
    # dummy array of n_samples rows is created once at construction; after the first
    # full iteration of split the generator is exhausted and later calls yield nothing.
    """A wrapper to make KFold single entry cv iterator"""

    def __init__(self, n_splits=4, n_samples=99):
        # [FUNCTION DOC]
        # HELPER: Stores n_splits (int, default 4) and n_samples (int, default 99) and
        # creates the underlying KFold(n_splits) split iterator over n_samples dummy rows.
        # Side effects: sets self.n_splits, self.n_samples, self.indices.
        self.n_splits = n_splits
        self.n_samples = n_samples
        self.indices = iter(KFold(n_splits=n_splits).split(np.ones(n_samples)))

    def split(self, X=None, y=None, groups=None):
        # [FUNCTION DOC]
        # HELPER: Yields the (train, test) index-array pairs from the shared iterator.
        # Inputs: X, y, groups - ignored. Outputs: generator of (train, test) ndarray pairs.
        # Side effects: consumes self.indices; a second call yields nothing (silently, no
        # error), which is the point of this class.
        """Split can be called only once"""
        for index in self.indices:
            yield index

    def get_n_splits(self, X=None, y=None, groups=None):
        # [FUNCTION DOC]
        # HELPER: Returns self.n_splits; X, y, groups are ignored. Does not reflect whether
        # the iterator has already been consumed.
        return self.n_splits
