# [MODULE DOC]
# Tests for sklearn.ensemble._base helpers: BaseEnsemble estimator construction
# (via BaggingClassifier) and _set_random_states seeding of nested estimators.
"""
Testing for the base module (sklearn.ensemble.base).
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from collections import OrderedDict

import numpy as np

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble._base import _set_random_states
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline


def test_base():
    # [FUNCTION DOC]
    # Verifies BaseEnsemble._make_estimator: after emptying estimators_ on a fitted
    # BaggingClassifier of Perceptrons, creates three appended estimators (no seed,
    # then two seeded from one RandomState) plus one non-appended; checks length 3,
    # type Perceptron, first random_state stays None, seeded ones get distinct ints.
    # Also smoke-tests fitting with n_estimators given as a numpy int32.
    # Check BaseEnsemble methods.
    ensemble = BaggingClassifier(
        estimator=Perceptron(random_state=None), n_estimators=3
    )

    iris = load_iris()
    ensemble.fit(iris.data, iris.target)
    ensemble.estimators_ = []  # empty the list and create estimators manually

    ensemble._make_estimator()
    random_state = np.random.RandomState(3)
    ensemble._make_estimator(random_state=random_state)
    ensemble._make_estimator(random_state=random_state)
    ensemble._make_estimator(append=False)

    assert 3 == len(ensemble)
    assert 3 == len(ensemble.estimators_)

    assert isinstance(ensemble[0], Perceptron)
    assert ensemble[0].random_state is None
    assert isinstance(ensemble[1].random_state, int)
    assert isinstance(ensemble[2].random_state, int)
    assert ensemble[1].random_state != ensemble[2].random_state

    np_int_ensemble = BaggingClassifier(
        estimator=Perceptron(), n_estimators=np.int32(3)
    )
    np_int_ensemble.fit(iris.data, iris.target)


def test_set_random_states():
    # [FUNCTION DOC]
    # Verifies _set_random_states: smoke test on an estimator without random_state
    # (LDA); None seed still sets an int; the same seed yields identical seeds on two
    # estimators; nested Pipeline/SelectFromModel parameters all get int seeds that
    # differ between parameters; and the assigned seeds do not depend on get_params
    # ordering (compared using pipelines whose get_params sorts ascending/descending).
    # Linear Discriminant Analysis doesn't have random state: smoke test
    _set_random_states(LinearDiscriminantAnalysis(), random_state=17)

    clf1 = Perceptron(random_state=None)
    assert clf1.random_state is None
    # check random_state is None still sets
    _set_random_states(clf1, None)
    assert isinstance(clf1.random_state, int)

    # check random_state fixes results in consistent initialisation
    _set_random_states(clf1, 3)
    assert isinstance(clf1.random_state, int)
    clf2 = Perceptron(random_state=None)
    _set_random_states(clf2, 3)
    assert clf1.random_state == clf2.random_state

    # nested random_state

    def make_steps():
        # [FUNCTION DOC]
        # HELPER: Builds fresh steps: a SelectFromModel(Perceptron) selector and a
        # Perceptron classifier, both with random_state None. Outputs: list of (name, est).
        return [
            ("sel", SelectFromModel(Perceptron(random_state=None))),
            ("clf", Perceptron(random_state=None)),
        ]

    est1 = Pipeline(make_steps())
    _set_random_states(est1, 3)
    assert isinstance(est1.steps[0][1].estimator.random_state, int)
    assert isinstance(est1.steps[1][1].random_state, int)
    assert (
        est1.get_params()["sel__estimator__random_state"]
        != est1.get_params()["clf__random_state"]
    )

    # ensure multiple random_state parameters are invariant to get_params()
    # iteration order

    class AlphaParamPipeline(Pipeline):
        # [CLASS DOC]
        # HELPER: Pipeline subclass whose get_params returns parameters sorted by key
        # ascending, used to test ordering invariance.
        def get_params(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns Pipeline.get_params output as an OrderedDict sorted ascending.
            params = Pipeline.get_params(self, *args, **kwargs).items()
            return OrderedDict(sorted(params))

    class RevParamPipeline(Pipeline):
        # [CLASS DOC]
        # HELPER: Pipeline subclass whose get_params returns parameters sorted by key
        # descending.
        def get_params(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns Pipeline.get_params output as an OrderedDict sorted descending.
            params = Pipeline.get_params(self, *args, **kwargs).items()
            return OrderedDict(sorted(params, reverse=True))

    for cls in [AlphaParamPipeline, RevParamPipeline]:
        est2 = cls(make_steps())
        _set_random_states(est2, 3)
        assert (
            est1.get_params()["sel__estimator__random_state"]
            == est2.get_params()["sel__estimator__random_state"]
        )
        assert (
            est1.get_params()["clf__random_state"]
            == est2.get_params()["clf__random_state"]
        )
