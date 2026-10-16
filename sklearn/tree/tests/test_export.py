# [MODULE DOC]
# Tests for the decision tree exporters in sklearn.tree._export: export_graphviz
# (exact DOT output on a 6-sample toy dataset, options, errors, precision),
# export_text (exact text reports and errors) and plot_tree (matplotlib annotations,
# via the pyplot fixture). Module-level toy data: X (6x2), binary y, multi-output y2,
# sample weights w and a single-class y_degraded.
# KNOWN ISSUE: the module docstring refers to sklearn.tree.export; the code now
# lives in the private module sklearn.tree._export.
"""
Testing for export functions of decision trees (sklearn.tree.export).
"""

from io import StringIO
from re import finditer, search
from textwrap import dedent

import numpy as np
import pytest
from numpy.random import RandomState

from sklearn.base import is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    export_graphviz,
    export_text,
    plot_tree,
)
from sklearn.tree._export import _rgb_to_hexstring

CLF_CRITERIONS = ("gini", "log_loss")
REG_CRITERIONS = ("squared_error", "absolute_error", "poisson")

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
y2 = [[-1, 1], [-1, 1], [-1, 1], [1, 2], [1, 2], [1, 3]]
w = [1, 1, 1, 0.5, 0.5, 0.5]
y_degraded = [1, 1, 1, 1, 1, 1]


def test_rgb_to_hexstring():
    # [FUNCTION DOC]
    # Checks that _rgb_to_hexstring zero-pads each channel, e.g. green becomes #00ff00.
    """
    Test that _rgb_to_hexstring correctly converts an RGB tuple to a hex color string.

    A previous bug caused incorrect hex color string generation for zero values
    in the RGB tuple.
    """

    assert _rgb_to_hexstring((0, 255, 0)) == "#00ff00"


def test_graphviz_toy():
    # [FUNCTION DOC]
    # Compares export_graphviz output string-for-string against expected DOT text for a
    # depth-3 gini classifier: defaults, feature_names (plain and with quotes escaped),
    # class_names (plain and escaped), filled/proportion/special_characters/rounded/
    # fontname options, max_depth=0 truncation with class_names=True and with
    # filled/node_ids, a weighted multi-output classifier, and a regressor with
    # leaves_parallel and rotate.
    # KNOWN ISSUE: for the degraded single-class classifier the expected string is built
    # but never compared to the output, so that last case is not actually checked.
    # Check correctness of export_graphviz
    clf = DecisionTreeClassifier(
        max_depth=3, min_samples_split=2, criterion="gini", random_state=2
    )
    clf.fit(X, y)

    # Test export code
    contents1 = export_graphviz(clf, out_file=None)
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="x[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]"] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )
    assert contents1 == contents2

    # Test with feature_names
    contents1 = export_graphviz(
        clf, feature_names=["feature0", "feature1"], out_file=None
    )
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="feature0 <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]"] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2

    # Test with feature_names (escaped)
    contents1 = export_graphviz(
        clf, feature_names=['feature"0"', 'feature"1"'], out_file=None
    )
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="feature\\"0\\" <= 0.0\\n'
        "gini = 0.5\\nsamples = 6\\n"
        'value = [3, 3]"] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2

    # Test with class_names
    contents1 = export_graphviz(clf, class_names=["yes", "no"], out_file=None)
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="x[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]\\nclass = yes"] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]\\n'
        'class = yes"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]\\n'
        'class = no"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2

    # Test with class_names (escaped)
    contents1 = export_graphviz(clf, class_names=['"yes"', '"no"'], out_file=None)
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="x[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]\\nclass = \\"yes\\""] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]\\n'
        'class = \\"yes\\""] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]\\n'
        'class = \\"no\\""] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2

    # Test plot_options
    contents1 = export_graphviz(
        clf,
        filled=True,
        impurity=False,
        proportion=True,
        special_characters=True,
        rounded=True,
        out_file=None,
        fontname="sans",
    )
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, style="filled, rounded", color="black", '
        'fontname="sans"] ;\n'
        'edge [fontname="sans"] ;\n'
        "0 [label=<x<SUB>0</SUB> &le; 0.0<br/>samples = 100.0%<br/>"
        'value = [0.5, 0.5]>, fillcolor="#ffffff"] ;\n'
        "1 [label=<samples = 50.0%<br/>value = [1.0, 0.0]>, "
        'fillcolor="#e58139"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        "2 [label=<samples = 50.0%<br/>value = [0.0, 1.0]>, "
        'fillcolor="#399de5"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2

    # Test max_depth
    contents1 = export_graphviz(clf, max_depth=0, class_names=True, out_file=None)
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="x[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]\\nclass = y[0]"] ;\n'
        '1 [label="(...)"] ;\n'
        "0 -> 1 ;\n"
        '2 [label="(...)"] ;\n'
        "0 -> 2 ;\n"
        "}"
    )

    assert contents1 == contents2

    # Test max_depth with plot_options
    contents1 = export_graphviz(
        clf, max_depth=0, filled=True, out_file=None, node_ids=True
    )
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, style="filled", color="black", '
        'fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="node #0\\nx[0] <= 0.0\\ngini = 0.5\\n'
        'samples = 6\\nvalue = [3, 3]", fillcolor="#ffffff"] ;\n'
        '1 [label="(...)", fillcolor="#C0C0C0"] ;\n'
        "0 -> 1 ;\n"
        '2 [label="(...)", fillcolor="#C0C0C0"] ;\n'
        "0 -> 2 ;\n"
        "}"
    )

    assert contents1 == contents2

    # Test multi-output with weighted samples
    clf = DecisionTreeClassifier(
        max_depth=2, min_samples_split=2, criterion="gini", random_state=2
    )
    clf = clf.fit(X, y2, sample_weight=w)

    contents1 = export_graphviz(clf, filled=True, impurity=False, out_file=None)
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, style="filled", color="black", '
        'fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="x[0] <= 0.0\\nsamples = 6\\n'
        "value = [[3.0, 1.5, 0.0]\\n"
        '[3.0, 1.0, 0.5]]", fillcolor="#ffffff"] ;\n'
        '1 [label="samples = 3\\nvalue = [[3, 0, 0]\\n'
        '[3, 0, 0]]", fillcolor="#e58139"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="x[0] <= 1.5\\nsamples = 3\\n'
        "value = [[0.0, 1.5, 0.0]\\n"
        '[0.0, 1.0, 0.5]]", fillcolor="#f1bd97"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        '3 [label="samples = 2\\nvalue = [[0, 1, 0]\\n'
        '[0, 1, 0]]", fillcolor="#e58139"] ;\n'
        "2 -> 3 ;\n"
        '4 [label="samples = 1\\nvalue = [[0.0, 0.5, 0.0]\\n'
        '[0.0, 0.0, 0.5]]", fillcolor="#e58139"] ;\n'
        "2 -> 4 ;\n"
        "}"
    )

    assert contents1 == contents2

    # Test regression output with plot_options
    clf = DecisionTreeRegressor(
        max_depth=3, min_samples_split=2, criterion="squared_error", random_state=2
    )
    clf.fit(X, y)

    contents1 = export_graphviz(
        clf,
        filled=True,
        leaves_parallel=True,
        out_file=None,
        rotate=True,
        rounded=True,
        fontname="sans",
    )
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, style="filled, rounded", color="black", '
        'fontname="sans"] ;\n'
        "graph [ranksep=equally, splines=polyline] ;\n"
        'edge [fontname="sans"] ;\n'
        "rankdir=LR ;\n"
        '0 [label="x[0] <= 0.0\\nsquared_error = 1.0\\nsamples = 6\\n'
        'value = 0.0", fillcolor="#f2c09c"] ;\n'
        '1 [label="squared_error = 0.0\\nsamples = 3\\'
        'nvalue = -1.0", '
        'fillcolor="#ffffff"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=-45, "
        'headlabel="True"] ;\n'
        '2 [label="squared_error = 0.0\\nsamples = 3\\nvalue = 1.0", '
        'fillcolor="#e58139"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=45, "
        'headlabel="False"] ;\n'
        "{rank=same ; 0} ;\n"
        "{rank=same ; 1; 2} ;\n"
        "}"
    )

    assert contents1 == contents2

    # Test classifier with degraded learning set
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y_degraded)

    contents1 = export_graphviz(clf, filled=True, out_file=None)
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, style="filled", color="black", '
        'fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="gini = 0.0\\nsamples = 6\\nvalue = 6.0", '
        'fillcolor="#ffffff"] ;\n'
        "}"
    )


@pytest.mark.parametrize("constructor", [list, np.array])
def test_graphviz_feature_class_names_array_support(constructor):
    # [FUNCTION DOC]
    # Checks export_graphviz accepts feature_names and class_names given as lists or
    # numpy arrays, producing the same DOT text as in test_graphviz_toy.
    # Check that export_graphviz treats feature names
    # and class names correctly and supports arrays
    clf = DecisionTreeClassifier(
        max_depth=3, min_samples_split=2, criterion="gini", random_state=2
    )
    clf.fit(X, y)

    # Test with feature_names
    contents1 = export_graphviz(
        clf, feature_names=constructor(["feature0", "feature1"]), out_file=None
    )
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="feature0 <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]"] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2

    # Test with class_names
    contents1 = export_graphviz(
        clf, class_names=constructor(["yes", "no"]), out_file=None
    )
    contents2 = (
        "digraph Tree {\n"
        'node [shape=box, fontname="helvetica"] ;\n'
        'edge [fontname="helvetica"] ;\n'
        '0 [label="x[0] <= 0.0\\ngini = 0.5\\nsamples = 6\\n'
        'value = [3, 3]\\nclass = yes"] ;\n'
        '1 [label="gini = 0.0\\nsamples = 3\\nvalue = [3, 0]\\n'
        'class = yes"] ;\n'
        "0 -> 1 [labeldistance=2.5, labelangle=45, "
        'headlabel="True"] ;\n'
        '2 [label="gini = 0.0\\nsamples = 3\\nvalue = [0, 3]\\n'
        'class = no"] ;\n'
        "0 -> 2 [labeldistance=2.5, labelangle=-45, "
        'headlabel="False"] ;\n'
        "}"
    )

    assert contents1 == contents2


def test_graphviz_errors():
    # [FUNCTION DOC]
    # Checks export_graphviz errors: NotFittedError for an unfitted tree, ValueError for
    # feature_names of wrong length or with non-string entries, TypeError when given a
    # Tree object instead of an estimator, and IndexError for empty class_names.
    # Check for errors of export_graphviz
    clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2)

    # Check not-fitted decision tree error
    out = StringIO()
    with pytest.raises(NotFittedError):
        export_graphviz(clf, out)

    clf.fit(X, y)

    # Check if it errors when length of feature_names
    # mismatches with number of features
    message = "Length of feature_names, 1 does not match number of features, 2"
    with pytest.raises(ValueError, match=message):
        export_graphviz(clf, None, feature_names=["a"])

    message = "Length of feature_names, 3 does not match number of features, 2"
    with pytest.raises(ValueError, match=message):
        export_graphviz(clf, None, feature_names=["a", "b", "c"])

    # Check error when feature_names contains non-string elements
    message = "All feature names must be strings."
    with pytest.raises(ValueError, match=message):
        export_graphviz(clf, None, feature_names=["a", 1])

    # Check error when argument is not an estimator
    message = "is not an estimator instance"
    with pytest.raises(TypeError, match=message):
        export_graphviz(clf.fit(X, y).tree_)

    # Check class_names error
    out = StringIO()
    with pytest.raises(IndexError):
        export_graphviz(clf, out, class_names=[])


@pytest.mark.parametrize("criterion", CLF_CRITERIONS + REG_CRITERIONS)
def test_criterion_in_gradient_boosting_graphviz(criterion):
    # [FUNCTION DOC]
    # For every classification and regression criterion, fits a tree (targets shifted
    # by +2 for regressors so poisson is valid) and checks that every bracketed node
    # label containing "samples" also mentions the criterion name.
    dot_data = StringIO()

    is_reg = criterion in REG_CRITERIONS
    Tree = DecisionTreeRegressor if is_reg else DecisionTreeClassifier
    clf = Tree(random_state=0, criterion=criterion)
    # positive values for poisson criterion:
    y_ = [yi + 2 for yi in y] if is_reg else y
    clf.fit(X, y_)
    export_graphviz(clf, out_file=dot_data)

    for finding in finditer(r"\[.*?samples.*?\]", dot_data.getvalue()):
        assert criterion in finding.group()


def test_precision():
    # [FUNCTION DOC]
    # Fits a depth-1 regressor and classifier on random data and, for precision 4 and 3
    # with proportion=True, checks via regexes on the DOT text that values have at most
    # precision decimals while impurities and thresholds have exactly precision decimals.
    rng_reg = RandomState(2)
    rng_clf = RandomState(8)
    for X, y, clf in zip(
        (rng_reg.random_sample((5, 2)), rng_clf.random_sample((1000, 4))),
        (rng_reg.random_sample((5,)), rng_clf.randint(2, size=(1000,))),
        (
            DecisionTreeRegressor(random_state=0, max_depth=1),
            DecisionTreeClassifier(max_depth=1, random_state=0),
        ),
    ):
        clf.fit(X, y)
        for precision in (4, 3):
            dot_data = export_graphviz(
                clf, out_file=None, precision=precision, proportion=True
            )

            # With the current random state, the impurity and the threshold
            # will have the number of precision set in the export_graphviz
            # function. We will check the number of precision with a strict
            # equality. The value reported will have only 2 precision and
            # therefore, only a less equal comparison will be done.

            # check value
            for finding in finditer(r"value = \d+\.\d+", dot_data):
                assert len(search(r"\.\d+", finding.group()).group()) <= precision + 1
            # check impurity
            if is_classifier(clf):
                pattern = r"gini = \d+\.\d+"
            else:
                pattern = r"squared_error = \d+\.\d+"

            # check impurity
            for finding in finditer(pattern, dot_data):
                assert len(search(r"\.\d+", finding.group()).group()) == precision + 1
            # check threshold
            for finding in finditer(r"<= \d+\.\d+", dot_data):
                assert len(search(r"\.\d+", finding.group()).group()) == precision + 1


def test_export_text_errors():
    # [FUNCTION DOC]
    # Checks export_text ValueErrors for feature_names of the wrong length and for a
    # class_names array whose length differs from the number of classes.
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    err_msg = "feature_names must contain 2 elements, got 1"
    with pytest.raises(ValueError, match=err_msg):
        export_text(clf, feature_names=["a"])
    err_msg = (
        "When `class_names` is an array, it should contain as"
        " many items as `decision_tree.classes_`. Got 1 while"
        " the tree was fitted with 2 classes."
    )
    with pytest.raises(ValueError, match=err_msg):
        export_text(clf, class_names=["a"])


def test_export_text():
    # [FUNCTION DOC]
    # Compares export_text output with exact expected reports: default classifier report,
    # max_depth 0 and 10 (level-1 leaves not truncated), show_weights, spacing=1,
    # truncated branch message on a deeper 3-class tree, multi-output regressor with
    # decimals=1 (show_weights has no effect for regressors), and a single-feature
    # regressor with custom feature_names.
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)

    expected_report = dedent(
        """
    |--- feature_1 <= 0.00
    |   |--- class: -1
    |--- feature_1 >  0.00
    |   |--- class: 1
    """
    ).lstrip()

    assert export_text(clf) == expected_report
    # testing that leaves at level 1 are not truncated
    assert export_text(clf, max_depth=0) == expected_report
    # testing that the rest of the tree is truncated
    assert export_text(clf, max_depth=10) == expected_report

    expected_report = dedent(
        """
    |--- feature_1 <= 0.00
    |   |--- weights: [3.00, 0.00] class: -1
    |--- feature_1 >  0.00
    |   |--- weights: [0.00, 3.00] class: 1
    """
    ).lstrip()
    assert export_text(clf, show_weights=True) == expected_report

    expected_report = dedent(
        """
    |- feature_1 <= 0.00
    | |- class: -1
    |- feature_1 >  0.00
    | |- class: 1
    """
    ).lstrip()
    assert export_text(clf, spacing=1) == expected_report

    X_l = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [-1, 1]]
    y_l = [-1, -1, -1, 1, 1, 1, 2]
    clf = DecisionTreeClassifier(max_depth=4, random_state=0)
    clf.fit(X_l, y_l)
    expected_report = dedent(
        """
    |--- feature_1 <= 0.00
    |   |--- class: -1
    |--- feature_1 >  0.00
    |   |--- truncated branch of depth 2
    """
    ).lstrip()
    assert export_text(clf, max_depth=0) == expected_report

    X_mo = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
    y_mo = [[-1, -1], [-1, -1], [-1, -1], [1, 1], [1, 1], [1, 1]]

    reg = DecisionTreeRegressor(max_depth=2, random_state=0)
    reg.fit(X_mo, y_mo)

    expected_report = dedent(
        """
    |--- feature_1 <= 0.0
    |   |--- value: [-1.0, -1.0]
    |--- feature_1 >  0.0
    |   |--- value: [1.0, 1.0]
    """
    ).lstrip()
    assert export_text(reg, decimals=1) == expected_report
    assert export_text(reg, decimals=1, show_weights=True) == expected_report

    X_single = [[-2], [-1], [-1], [1], [1], [2]]
    reg = DecisionTreeRegressor(max_depth=2, random_state=0)
    reg.fit(X_single, y_mo)

    expected_report = dedent(
        """
    |--- first <= 0.0
    |   |--- value: [-1.0, -1.0]
    |--- first >  0.0
    |   |--- value: [1.0, 1.0]
    """
    ).lstrip()
    assert export_text(reg, decimals=1, feature_names=["first"]) == expected_report
    assert (
        export_text(reg, decimals=1, show_weights=True, feature_names=["first"])
        == expected_report
    )


@pytest.mark.parametrize("constructor", [list, np.array])
def test_export_text_feature_class_names_array_support(constructor):
    # [FUNCTION DOC]
    # Checks export_text accepts feature_names and class_names as lists or numpy arrays.
    # KNOWN ISSUE: the inline comment says export_graphviz, but export_text is tested.
    # Check that export_graphviz treats feature names
    # and class names correctly and supports arrays
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)

    expected_report = dedent(
        """
    |--- b <= 0.00
    |   |--- class: -1
    |--- b >  0.00
    |   |--- class: 1
    """
    ).lstrip()
    assert export_text(clf, feature_names=constructor(["a", "b"])) == expected_report

    expected_report = dedent(
        """
    |--- feature_1 <= 0.00
    |   |--- class: cat
    |--- feature_1 >  0.00
    |   |--- class: dog
    """
    ).lstrip()
    assert export_text(clf, class_names=constructor(["cat", "dog"])) == expected_report


def test_plot_tree_entropy(pyplot):
    # [FUNCTION DOC]
    # Smoke/correctness test of plot_tree for an entropy classifier: checks that 5
    # annotations are returned (3 nodes and the True/False edge labels) with the exact
    # expected texts. Inputs: pyplot fixture (matplotlib, closes figures).
    # mostly smoke tests
    # Check correctness of export_graphviz for criterion = entropy
    clf = DecisionTreeClassifier(
        max_depth=3, min_samples_split=2, criterion="entropy", random_state=2
    )
    clf.fit(X, y)

    # Test export code
    feature_names = ["first feat", "sepal_width"]
    nodes = plot_tree(clf, feature_names=feature_names)
    assert len(nodes) == 5
    assert (
        nodes[0].get_text()
        == "first feat <= 0.0\nentropy = 1.0\nsamples = 6\nvalue = [3, 3]"
    )
    assert nodes[1].get_text() == "entropy = 0.0\nsamples = 3\nvalue = [3, 0]"
    assert nodes[2].get_text() == "True  "
    assert nodes[3].get_text() == "entropy = 0.0\nsamples = 3\nvalue = [0, 3]"
    assert nodes[4].get_text() == "  False"


@pytest.mark.parametrize("fontsize", [None, 10, 20])
def test_plot_tree_gini(pyplot, fontsize):
    # [FUNCTION DOC]
    # Same as the entropy test for a gini classifier, additionally checking that all
    # annotations use the requested fontsize when one is given.
    # Inputs: pyplot fixture; fontsize - None, 10 or 20.
    # mostly smoke tests
    # Check correctness of export_graphviz for criterion = gini
    clf = DecisionTreeClassifier(
        max_depth=3,
        min_samples_split=2,
        criterion="gini",
        random_state=2,
    )
    clf.fit(X, y)

    # Test export code
    feature_names = ["first feat", "sepal_width"]
    nodes = plot_tree(clf, feature_names=feature_names, fontsize=fontsize)
    assert len(nodes) == 5
    if fontsize is not None:
        assert all(node.get_fontsize() == fontsize for node in nodes)
    assert (
        nodes[0].get_text()
        == "first feat <= 0.0\ngini = 0.5\nsamples = 6\nvalue = [3, 3]"
    )
    assert nodes[1].get_text() == "gini = 0.0\nsamples = 3\nvalue = [3, 0]"
    assert nodes[2].get_text() == "True  "
    assert nodes[3].get_text() == "gini = 0.0\nsamples = 3\nvalue = [0, 3]"
    assert nodes[4].get_text() == "  False"


def test_not_fitted_tree(pyplot):
    # [FUNCTION DOC]
    # Checks that plot_tree raises NotFittedError on an unfitted regressor.
    # Testing if not fitted tree throws the correct error
    clf = DecisionTreeRegressor()
    with pytest.raises(NotFittedError):
        plot_tree(clf)
