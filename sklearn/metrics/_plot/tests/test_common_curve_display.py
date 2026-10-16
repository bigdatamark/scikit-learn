# [MODULE DOC]
# Common tests shared across the binary-classifier curve displays (Calibration,
# DET, PrecisionRecall, ROC, plus PredictionError and ConfusionMatrix where relevant):
# error handling for non-binary or regressor inputs, missing response methods, unfitted
# estimators, inconsistent lengths, pos_label; name overriding; subclass return types;
# multi-curve legend labels; from_cv_results validation and curve_kwargs; deprecations.
# Also provides _check_pos_label_statistics, imported by the ROC and PR test modules.
from collections.abc import Mapping
from contextlib import suppress

import numpy as np
import pytest

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    DetCurveDisplay,
    PrecisionRecallDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import shuffle


@pytest.fixture(scope="module")
def data():
    # [FUNCTION DOC]
    # HELPER: Module-scoped fixture returning the full 3-class iris (X, y).
    return load_iris(return_X_y=True)


@pytest.fixture(scope="module")
def data_binary(data):
    # [FUNCTION DOC]
    # HELPER: Module-scoped fixture returning the iris samples of classes 0 and 1 only.
    # Inputs: data fixture. Outputs: (X, y) binary subset.
    X, y = data
    return X[y < 2], y[y < 2]


def _check_pos_label_statistics(
    display_class, response_method, constructor_name, check_metric
):
    # [FUNCTION DOC]
    # HELPER: Builds a highly imbalanced binary problem from breast cancer (all
    # negatives plus 25 positives, shuffled, first 2 features only, string labels
    #   'cancer'
    # and 'not cancer'), splits it stratified, fits a LogisticRegression on the train
    #   set
    # and runs a 3-fold cross_validate on all data. Asserts classes_ is ['cancer', 'not
    # cancer'], then for pos_label 'cancer' (scores: column 0 of probabilities or
    #   negated
    # decision function) and 'not cancer' (column 1 or raw decision function) builds the
    # display with the requested constructor and calls check_metric on it.
    # Inputs: display_class; response_method; constructor_name - 'from_estimator',
    # 'from_predictions' or anything else meaning 'from_cv_results'; check_metric -
    # callable(display, constructor_name, pos_label) performing the assertions.
    # Outputs: none. Side effects: creates figures.
    """Test switching `pos_label` gives correct statistics, using imbalanced data."""
    X, y = load_breast_cancer(return_X_y=True)
    # create highly imbalanced classes
    idx_positive = np.flatnonzero(y == 1)
    idx_negative = np.flatnonzero(y == 0)
    idx_selected = np.hstack([idx_negative, idx_positive[:25]])
    X, y = X[idx_selected], y[idx_selected]
    X, y = shuffle(X, y, random_state=42)
    # only use 2 features to make the problem even harder
    X = X[:, :2]
    y = np.array(["cancer" if c == 1 else "not cancer" for c in y], dtype=object)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        random_state=0,
    )

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    cv_results = cross_validate(
        LogisticRegression(), X, y, cv=3, return_estimator=True, return_indices=True
    )

    # Sanity check to be sure the positive class is `classes_[0]`.
    # Class imbalance ensures a large difference in prediction values between classes,
    # allowing us to catch errors when we switch `pos_label`.
    assert classifier.classes_.tolist() == ["cancer", "not cancer"]

    y_score = getattr(classifier, response_method)(X_test)
    # we select the corresponding probability columns or reverse the decision
    # function otherwise
    y_score_cancer = -1 * y_score if y_score.ndim == 1 else y_score[:, 0]
    y_score_not_cancer = y_score if y_score.ndim == 1 else y_score[:, 1]

    pos_label = "cancer"
    y_score = y_score_cancer
    if constructor_name == "from_estimator":
        display = display_class.from_estimator(
            classifier,
            X_test,
            y_test,
            pos_label=pos_label,
            response_method=response_method,
        )
    elif constructor_name == "from_predictions":
        display = display_class.from_predictions(
            y_test,
            y_score,
            pos_label=pos_label,
        )
    else:  # constructor_name = "from_cv_results"
        display = display_class.from_cv_results(
            cv_results,
            X,
            y,
            response_method=response_method,
            pos_label=pos_label,
        )

    check_metric(display, constructor_name, pos_label)

    pos_label = "not cancer"
    y_score = y_score_not_cancer
    if constructor_name == "from_estimator":
        display = display_class.from_estimator(
            classifier,
            X_test,
            y_test,
            response_method=response_method,
            pos_label=pos_label,
        )
    elif constructor_name == "from_predictions":
        display = display_class.from_predictions(
            y_test,
            y_score,
            pos_label=pos_label,
        )
    else:  # constructor_name = "from_cv_results"
        display = display_class.from_cv_results(
            cv_results,
            X,
            y,
            response_method=response_method,
            pos_label=pos_label,
        )

    check_metric(display, constructor_name, pos_label)


@pytest.mark.parametrize(
    "Display",
    [CalibrationDisplay, DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay],
)
def test_display_curve_error_binary_classifier(pyplot, data, data_binary, Display):
    # [FUNCTION DOC]
    # Verifies that from_estimator raises ValueError for a multiclass classifier (with
    # multiclass or binary target) and for a binary classifier given a multiclass
    #   target.
    """Check correct error raised when only binary classification supported."""
    X, y = data
    X_binary, y_binary = data_binary
    clf = DecisionTreeClassifier().fit(X, y)

    # Case 1: multiclass classifier with multiclass target
    msg = "Expected 'estimator' to be a binary classifier. Got 3 classes instead."
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(clf, X, y)

    # Case 2: multiclass classifier with binary target
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(clf, X_binary, y_binary)

    # Case 3: binary classifier with multiclass target
    clf = DecisionTreeClassifier().fit(X_binary, y_binary)
    msg = "The target y is not binary. Got multiclass type of target."
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(clf, X, y)


@pytest.mark.parametrize(
    "Display",
    [CalibrationDisplay, DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay],
)
def test_display_curve_error_regression(pyplot, data_binary, Display):
    # [FUNCTION DOC]
    # Verifies ValueError when from_estimator gets a regressor, and when the target is
    # continuous for from_estimator and from_predictions.
    """Check that we raise an error with regressor."""

    # Case 1: regressor
    X, y = data_binary
    regressor = DecisionTreeRegressor().fit(X, y)

    msg = "Expected 'estimator' to be a binary classifier. Got DecisionTreeRegressor"
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(regressor, X, y)

    # Case 2: regression target
    classifier = DecisionTreeClassifier().fit(X, y)
    # Force `y_true` to be seen as a regression problem
    y = y + 0.5
    msg = "The target y is not binary. Got continuous type of target."
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(classifier, X, y)
    with pytest.raises(ValueError, match=msg):
        Display.from_predictions(y, regressor.fit(X, y).predict(X))


@pytest.mark.parametrize(
    "response_method, msg",
    [
        (
            "predict_proba",
            "MyClassifier has none of the following attributes: predict_proba.",
        ),
        (
            "decision_function",
            "MyClassifier has none of the following attributes: decision_function.",
        ),
        (
            "auto",
            (
                "MyClassifier has none of the following attributes: predict_proba,"
                " decision_function."
            ),
        ),
        (
            "bad_method",
            "MyClassifier has none of the following attributes: bad_method.",
        ),
    ],
)
@pytest.mark.parametrize(
    "Display", [DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay]
)
def test_display_curve_error_no_response_method(
    pyplot,
    data_binary,
    response_method,
    msg,
    Display,
):
    # [FUNCTION DOC]
    # Verifies that from_estimator raises AttributeError listing the missing
    # method(s) when the classifier lacks predict_proba/decision_function or the
    #   requested
    # method does not exist.
    """Check error raised when `response_method` not defined for `estimator`."""
    X, y = data_binary

    class MyClassifier(ClassifierMixin, BaseEstimator):
        # [CLASS DOC]
        # HELPER: Minimal classifier with only fit; has no predict_proba nor
        # decision_function.
        def fit(self, X, y):
            # [FUNCTION DOC]
            # HELPER: Sets classes_ to [0, 1] and returns self; ignores the data.
            # Side effects: sets self.classes_.
            self.classes_ = [0, 1]
            return self

    clf = MyClassifier().fit(X, y)

    with pytest.raises(AttributeError, match=msg):
        Display.from_estimator(clf, X, y, response_method=response_method)


@pytest.mark.parametrize(
    "Display",
    [CalibrationDisplay, DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay],
)
@pytest.mark.parametrize(
    "constructor_name", ["from_estimator", "from_predictions", "from_cv_results"]
)
def test_display_curve_name_overwritten_by_plot_multiple_calls(
    pyplot,
    data_binary,
    Display,
    constructor_name,
):
    # [FUNCTION DOC]
    # Verifies for each display and constructor (skipping from_cv_results where not
    # implemented) that the name given to the constructor is stored (name or
    # estimator_name), reused when calling plot without a name, and replaced when plot
    # receives a new name.
    """Check passing `name` in `plot` overwrites name passed in `from_*` method."""
    X, y = data_binary
    clf_name = "my hand-crafted name"
    clf = LogisticRegression().fit(X, y)
    y_pred = clf.predict_proba(X)[:, 1]
    cv_results = cross_validate(
        LogisticRegression(), X, y, cv=3, return_estimator=True, return_indices=True
    )

    if constructor_name == "from_estimator":
        disp = Display.from_estimator(clf, X, y, name=clf_name)
    elif constructor_name == "from_predictions":
        disp = Display.from_predictions(y, y_pred, name=clf_name)
    else:  # constructor_name = "from_cv_results"
        if Display in (RocCurveDisplay, PrecisionRecallDisplay):
            disp = Display.from_cv_results(cv_results, X, y, name=clf_name)
        else:
            pytest.skip(f"`from_cv_results` not implemented in {Display}")

    # TODO: Clean-up once `estimator_name` deprecated in all displays
    if Display in (PrecisionRecallDisplay, RocCurveDisplay):
        assert disp.name == clf_name
    else:
        assert disp.estimator_name == clf_name
    pyplot.close("all")
    disp.plot()
    if constructor_name == "from_cv_results":
        assert clf_name in disp.line_[0].get_label()
    else:
        assert clf_name in disp.line_.get_label()
    pyplot.close("all")
    clf_name = "another_name"
    disp.plot(name=clf_name)
    if constructor_name == "from_cv_results":
        assert clf_name in disp.line_[0].get_label()
    else:
        assert clf_name in disp.line_.get_label()


@pytest.mark.parametrize(
    "clf",
    [
        LogisticRegression(),
        make_pipeline(StandardScaler(), LogisticRegression()),
        make_pipeline(
            make_column_transformer((StandardScaler(), [0, 1])), LogisticRegression()
        ),
    ],
)
@pytest.mark.parametrize(
    "Display",
    [CalibrationDisplay, DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay],
)
def test_display_curve_not_fitted_errors(pyplot, data_binary, clf, Display):
    # [FUNCTION DOC]
    # Verifies that from_estimator raises NotFittedError on unfitted classifiers or
    # pipelines, and after fitting the class name appears in the line label and stored
    # name.
    """Check correct error raised when `estimator` is not fitted."""
    X, y = data_binary
    # clone since we parametrize the test and the classifier will be fitted
    # when testing the second and subsequent plotting function
    model = clone(clf)
    with pytest.raises(NotFittedError):
        Display.from_estimator(model, X, y)
    model.fit(X, y)
    disp = Display.from_estimator(model, X, y)
    assert model.__class__.__name__ in disp.line_.get_label()
    # TODO: Clean-up once `estimator_name` deprecated in all displays
    if Display in (PrecisionRecallDisplay, RocCurveDisplay):
        assert disp.name == model.__class__.__name__
    else:
        assert disp.estimator_name == model.__class__.__name__


@pytest.mark.parametrize(
    "Display",
    [CalibrationDisplay, DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay],
)
def test_display_curve_n_samples_consistency(pyplot, data_binary, Display):
    # [FUNCTION DOC]
    # Verifies ValueError for inconsistent numbers of samples between X and y, and for a
    # short sample_weight (except CalibrationDisplay which has no sample_weight).
    """Check error raised when `y_pred` or `sample_weight` have inconsistent length."""
    X, y = data_binary
    classifier = DecisionTreeClassifier().fit(X, y)

    msg = "Found input variables with inconsistent numbers of samples"
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(classifier, X[:-2], y)
    with pytest.raises(ValueError, match=msg):
        Display.from_estimator(classifier, X, y[:-2])
    # `CalibrationDisplay` does not support `sample_weight`
    if Display != CalibrationDisplay:
        with pytest.raises(ValueError, match=msg):
            Display.from_estimator(
                classifier, X, y, sample_weight=np.ones(X.shape[0] - 2)
            )


@pytest.mark.parametrize(
    "Display",
    [CalibrationDisplay, DetCurveDisplay, PrecisionRecallDisplay, RocCurveDisplay],
)
def test_display_curve_error_pos_label(pyplot, data_binary, Display):
    # [FUNCTION DOC]
    # Verifies that from_predictions on labels {10, 11} without pos_label raises the
    # standard ValueError asking to specify pos_label.
    """Check consistency of error message when `pos_label` should be specified."""
    X, y = data_binary
    y = y + 10

    classifier = DecisionTreeClassifier().fit(X, y)
    y_pred = classifier.predict_proba(X)[:, -1]
    msg = r"y_true takes value in {10, 11} and pos_label is not specified"
    with pytest.raises(ValueError, match=msg):
        Display.from_predictions(y, y_pred)


@pytest.mark.parametrize(
    "Display",
    [
        CalibrationDisplay,
        DetCurveDisplay,
        PrecisionRecallDisplay,
        RocCurveDisplay,
        PredictionErrorDisplay,
        ConfusionMatrixDisplay,
    ],
)
@pytest.mark.parametrize(
    "constructor",
    ["from_predictions", "from_estimator", "from_cv_results"],
)
def test_classifier_display_curve_named_constructor_return_type(
    pyplot, data_binary, Display, constructor
):
    # [FUNCTION DOC]
    # Non-regression for PR #27675: verifies that from_predictions, from_estimator and
    # (where implemented) from_cv_results called on a subclass return an instance of
    #   that
    # subclass.
    """Check that named constructors return the correct type when subclassed.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/pull/27675
    """
    X, y = data_binary

    # This can be anything - we just need to check the named constructor return
    # type so the only requirement here is instantiating the class without error
    y_pred = y

    classifier = LogisticRegression().fit(X, y)
    cv_results = cross_validate(
        LogisticRegression(), X, y, cv=3, return_estimator=True, return_indices=True
    )

    class SubclassOfDisplay(Display):
        # [CLASS DOC]
        # HELPER: Empty subclass of the parametrized display class.
        pass

    if constructor == "from_predictions":
        curve = SubclassOfDisplay.from_predictions(y, y_pred)
    elif constructor == "from_estimator":
        curve = SubclassOfDisplay.from_estimator(classifier, X, y)
    else:  # `from_cv_results`
        if Display in (RocCurveDisplay, PrecisionRecallDisplay):
            curve = SubclassOfDisplay.from_cv_results(cv_results, X, y)
        else:
            pytest.skip(f"`from_cv_results` not implemented in {Display}")

    assert isinstance(curve, SubclassOfDisplay)


@pytest.mark.parametrize(
    "Display, display_args",
    [
        (
            PrecisionRecallDisplay,
            {
                "precision": np.array([1, 0.5, 0]),
                "recall": [np.array([0, 0.5, 1])],
                "average_precision": None,
                "name": "test_curve",
                "prevalence_pos_label": 0.5,
            },
        ),
        (
            RocCurveDisplay,
            {
                "fpr": np.array([0, 0.5, 1]),
                "tpr": [np.array([0, 0.5, 1])],
                "roc_auc": None,
                "name": "test_curve",
            },
        ),
    ],
)
def test_display_validate_plot_params(pyplot, Display, display_args):
    # [FUNCTION DOC]
    # Verifies that _validate_plot_params of the PR and ROC displays returns as many
    # values as constructor args, wraps names and arrays into single-element lists and
    #   keeps
    # None values as None.
    # KNOWN ISSUE: the name check 'assert x == [value] if cond else value' parses as a
    # conditional around the comparison, so for non-string names it only asserts value
    #   is
    # truthy.
    """Check `_validate_plot_params` returns the correct variables.

    `display_args` should be given in the same order as output by
    `_validate_plot_params`. All `display_args` should be for a single curve.
    """
    display = Display(**display_args)
    results = display._validate_plot_params(ax=None, name=None)

    # Check if the number of parameters match
    assert len(results) == len(display_args)

    for idx, (param, value) in enumerate(display_args.items()):
        if param == "name":
            assert results[idx] == [value] if isinstance(value, str) else value
        elif value is None:
            assert results[idx] is None
        else:
            assert isinstance(results[idx], list)
            assert len(results[idx]) == 1


auc_metrics = [[1.0, 1.0, 1.0], None]


@pytest.mark.parametrize(
    "Display, auc_metric_name, auc_arg_name, display_args",
    [
        pytest.param(
            RocCurveDisplay,
            "AUC",
            "roc_auc",
            {
                "fpr": [np.array([0, 0.5, 1])] * 3,
                "tpr": [np.array([0, 0.5, 1])] * 3,
                "roc_auc": auc_metric,
            },
        )
        for auc_metric in auc_metrics
    ]
    + [
        pytest.param(
            PrecisionRecallDisplay,
            "AP",
            "average_precision",
            {
                "precision": [np.array([1, 0.5, 0])] * 3,
                "recall": [np.array([0, 0.5, 1])] * 3,
                "average_precision": auc_metric,
            },
        )
        for auc_metric in auc_metrics
    ],
)
@pytest.mark.parametrize(
    "curve_kwargs",
    [None, {"color": "red"}, [{"c": "red"}, {"c": "green"}, {"c": "yellow"}]],
)
@pytest.mark.parametrize("name", [None, "single", ["one", "two", "three"]])
def test_display_plot_legend_label(
    pyplot, Display, auc_metric_name, auc_arg_name, display_args, curve_kwargs, name
):
    # [FUNCTION DOC]
    # Verifies multi-curve (3 curves) legend labels from direct construction for PR and
    # ROC across curve_kwargs (None, dict, list) and name (None, str, list) with or
    #   without
    # metric values: ValueError when name is a list but curve_kwargs is not; no legend
    #   when
    # neither name nor metric exists; one label per curve with list curve_kwargs;
    #   otherwise
    # a single aggregate label with mean +/- std.
    """Check legend label correct with all `curve_kwargs`, `name` combinations.

    Checks `from_estimator` and `from_predictions` methods, when plotting multiple
    curves.
    """
    if not isinstance(curve_kwargs, list) and isinstance(name, list):
        with pytest.raises(ValueError, match="To avoid labeling individual curves"):
            Display(**display_args).plot(name=name, curve_kwargs=curve_kwargs)
        return

    display = Display(**display_args).plot(name=name, curve_kwargs=curve_kwargs)
    legend = display.ax_.get_legend()
    auc_metric = display_args[auc_arg_name]

    if legend is None:
        # No legend is created, exit test early
        assert name is None
        assert auc_metric is None
        return
    else:
        legend_labels = [text.get_text() for text in legend.get_texts()]

    if isinstance(curve_kwargs, list):
        # Multiple labels in legend
        assert len(legend_labels) == 3
        for idx, label in enumerate(legend_labels):
            if name is None:
                expected_label = f"{auc_metric_name} = 1.00" if auc_metric else None
                assert label == expected_label
            elif isinstance(name, str):
                expected_label = (
                    f"single ({auc_metric_name} = 1.00)" if auc_metric else "single"
                )
                assert label == expected_label
            else:
                # `name` is a list of different strings
                expected_label = (
                    f"{name[idx]} ({auc_metric_name} = 1.00)"
                    if auc_metric
                    else f"{name[idx]}"
                )
                assert label == expected_label
    else:
        # Single label in legend
        assert len(legend_labels) == 1
        if name is None:
            expected_label = (
                f"{auc_metric_name} = 1.00 +/- 0.00" if auc_metric else None
            )
            assert legend_labels[0] == expected_label
        else:
            # name is single string
            expected_label = (
                f"single ({auc_metric_name} = 1.00 +/- 0.00)"
                if auc_metric
                else "single"
            )
            assert legend_labels[0] == expected_label
    # Close plots, prevents "more than 20 figures" opened warning
    pyplot.close("all")


@pytest.mark.parametrize("Display", [PrecisionRecallDisplay, RocCurveDisplay])
@pytest.mark.parametrize(
    "constructor_name, expected_clf_name",
    [
        ("from_estimator", "LogisticRegression"),
        ("from_predictions", "Classifier"),
    ],
)
def test_display_default_name(
    pyplot,
    data_binary,
    constructor_name,
    expected_clf_name,
    Display,
):
    # [FUNCTION DOC]
    # Verifies default names ('LogisticRegression' or 'Classifier') appear in disp.name
    # and the line label for PR and ROC.
    # Check the default name display in the figure when `name` is not provided
    X, y = data_binary

    lr = LogisticRegression().fit(X, y)
    y_score = lr.predict_proba(X)[:, 1]

    if constructor_name == "from_estimator":
        disp = Display.from_estimator(lr, X, y)
    else:  # constructor_name = "from_predictions"
        disp = Display.from_predictions(y, y_score)

    assert expected_clf_name in disp.name
    assert expected_clf_name in disp.line_.get_label()


@pytest.mark.parametrize(
    "Display, auc_metrics, auc_metric_name",
    [
        (PrecisionRecallDisplay, [0.97, 1.00, 1.00], "AP"),
        (RocCurveDisplay, [0.96, 1.00, 1.00], "AUC"),
    ],
)
@pytest.mark.parametrize(
    "curve_kwargs",
    [None, {"color": "red"}, [{"c": "red"}, {"c": "green"}, {"c": "yellow"}]],
)
@pytest.mark.parametrize("name", [None, "single", ["one", "two", "three"]])
def test_display_from_cv_results_legend_label(
    pyplot, Display, auc_metrics, auc_metric_name, curve_kwargs, name
):
    # [FUNCTION DOC]
    # Verifies the same name/curve_kwargs legend rules as test_display_plot_legend_label
    # but through from_cv_results on a 50-sample problem, with hard-coded per-fold
    #   metric
    # values.
    # KNOWN ISSUE: the line 'X, y = X, y = make_classification(...)' contains a
    #   redundant
    # chained assignment.
    """Check legend label correct with all `curve_kwargs`, `name` combinations.

    This function verifies that the legend labels in a Display object created from
    cross-validation results are correctly formatted based on the provided parameters.
    """
    X, y = X, y = make_classification(n_classes=2, n_samples=50, random_state=0)
    cv_results = cross_validate(
        LogisticRegression(), X, y, cv=3, return_estimator=True, return_indices=True
    )

    if not isinstance(curve_kwargs, list) and isinstance(name, list):
        with pytest.raises(ValueError, match="To avoid labeling individual curves"):
            Display.from_cv_results(
                cv_results, X, y, name=name, curve_kwargs=curve_kwargs
            )
    else:
        display = Display.from_cv_results(
            cv_results, X, y, name=name, curve_kwargs=curve_kwargs
        )

        legend = display.ax_.get_legend()
        legend_labels = [text.get_text() for text in legend.get_texts()]
        if isinstance(curve_kwargs, list):
            # Multiple labels in legend
            assert len(legend_labels) == 3
            for idx, label in enumerate(legend_labels):
                if name is None:
                    assert label == f"{auc_metric_name} = {auc_metrics[idx]:.2f}"
                elif isinstance(name, str):
                    assert (
                        label == f"single ({auc_metric_name} = {auc_metrics[idx]:.2f})"
                    )
                else:
                    # `name` is a list of different strings
                    assert (
                        label
                        == f"{name[idx]} ({auc_metric_name} = {auc_metrics[idx]:.2f})"
                    )
        else:
            # Single label in legend
            assert len(legend_labels) == 1
            if name is None:
                assert legend_labels[0] == (
                    f"{auc_metric_name} = {np.mean(auc_metrics):.2f} +/- "
                    f"{np.std(auc_metrics):.2f}"
                )
            else:
                # name is single string
                assert legend_labels[0] == (
                    f"single ({auc_metric_name} = {np.mean(auc_metrics):.2f} +/- "
                    f"{np.std(auc_metrics):.2f})"
                )
    # Close plots, prevents "more than 20 figures" opened warning
    pyplot.close("all")


@pytest.mark.parametrize("Display", [PrecisionRecallDisplay, RocCurveDisplay])
def test_display_from_cv_results_param_validation(pyplot, data_binary, Display):
    # [FUNCTION DOC]
    # Verifies from_cv_results errors: missing cv_results keys, wrong X length,
    # non-binary y, inconsistent y/sample_weight lengths, a 'No positive ... in y_true'
    # warning when the positive label is absent, list name with non-list curve_kwargs,
    #   wrong
    # curve_kwargs length and conflicting style aliases (TypeError).
    # KNOWN ISSUE: both 'missing key' cv_results are built identically (return_estimator
    # True, return_indices False), so the missing-estimator case is never tested.
    """Check parameter validation is correct."""
    X, y = data_binary

    # `cv_results` missing key
    cv_results_no_est = cross_validate(
        LogisticRegression(), X, y, cv=3, return_estimator=True, return_indices=False
    )
    cv_results_no_indices = cross_validate(
        LogisticRegression(), X, y, cv=3, return_estimator=True, return_indices=False
    )
    for cv_results in (cv_results_no_est, cv_results_no_indices):
        with pytest.raises(
            ValueError,
            match="`cv_results` does not contain one of the following required",
        ):
            Display.from_cv_results(cv_results, X, y)

    cv_results = cross_validate(
        LogisticRegression(), X, y, cv=3, return_estimator=True, return_indices=True
    )

    # `X` wrong length
    with pytest.raises(ValueError, match="`X` does not contain the correct"):
        Display.from_cv_results(cv_results, X[:10, :], y)

    # `y` not binary
    y_multi = y.copy()
    y_multi[0] = 2
    with pytest.raises(ValueError, match="The target `y` is not binary."):
        Display.from_cv_results(cv_results, X, y_multi)

    # input inconsistent length
    with pytest.raises(ValueError, match="Found input variables with inconsistent"):
        Display.from_cv_results(cv_results, X, y[:10])
    with pytest.raises(ValueError, match="Found input variables with inconsistent"):
        Display.from_cv_results(cv_results, X, y, sample_weight=[1, 2])

    # `pos_label` inconsistency
    y_multi[y_multi == 1] = 2
    with suppress(ValueError):  # ignore any `pos_label` side errors
        with pytest.warns(
            # Also captures subclass warnings e.g., `UndefinedMetricWarning`
            UserWarning,
            match="No positive .* in y_true",
        ):
            Display.from_cv_results(cv_results, X, y_multi)

    # `name` is list while `curve_kwargs` is None or dict
    for curve_kwargs in (None, {"alpha": 0.2}):
        with pytest.raises(ValueError, match="To avoid labeling individual curves"):
            Display.from_cv_results(
                cv_results,
                X,
                y,
                name=["one", "two", "three"],
                curve_kwargs=curve_kwargs,
            )

    # `curve_kwargs` incorrect length
    with pytest.raises(ValueError, match="`curve_kwargs` must be None, a dictionary"):
        Display.from_cv_results(cv_results, X, y, curve_kwargs=[{"alpha": 1}])

    # `curve_kwargs` both alias provided
    with pytest.raises(TypeError, match="Got both c and"):
        Display.from_cv_results(
            cv_results, X, y, curve_kwargs={"c": "blue", "color": "red"}
        )


@pytest.mark.parametrize("Display", [PrecisionRecallDisplay, RocCurveDisplay])
def test_display_from_cv_results_pos_label_inferred(pyplot, data_binary, Display):
    # [FUNCTION DOC]
    # Verifies that with pos_label None, from_cv_results stores pos_label equal to the
    # fold estimator's classes_[1] (1).
    """Check `pos_label` inferred correctly by `from_cv_results(pos_label=None)`."""
    X, y = data_binary
    cv_results = cross_validate(
        LogisticRegression(), X, y, cv=3, return_estimator=True, return_indices=True
    )

    disp = Display.from_cv_results(cv_results, X, y, pos_label=None)
    # Should be `estimator.classes_[1]`
    assert disp.pos_label == 1


@pytest.mark.parametrize("Display", [PrecisionRecallDisplay, RocCurveDisplay])
@pytest.mark.parametrize(
    "curve_kwargs",
    [None, {"alpha": 0.2}, [{"alpha": 0.2}, {"alpha": 0.3}, {"alpha": 0.4}]],
)
def test_display_from_cv_results_curve_kwargs(
    pyplot, data_binary, curve_kwargs, Display
):
    # [FUNCTION DOC]
    # Verifies that fold lines use default alpha 0.5, a shared dict alpha, or per-curve
    # alphas from a list.
    """Check `curve_kwargs` correctly passed from `from_cv_results`."""
    X, y = data_binary
    cv_results = cross_validate(
        LogisticRegression(), X, y, cv=3, return_estimator=True, return_indices=True
    )
    display = Display.from_cv_results(
        cv_results,
        X,
        y,
        curve_kwargs=curve_kwargs,
    )
    if curve_kwargs is None:
        # Default `alpha` used
        assert all(line.get_alpha() == 0.5 for line in display.line_)
    elif isinstance(curve_kwargs, Mapping):
        # `alpha` from dict used for all curves
        assert all(line.get_alpha() == 0.2 for line in display.line_)
    else:
        # Different `alpha` used for each curve
        assert all(
            line.get_alpha() == curve_kwargs[i]["alpha"]
            for i, line in enumerate(display.line_)
        )


@pytest.mark.parametrize("Display", [PrecisionRecallDisplay, RocCurveDisplay])
@pytest.mark.parametrize(
    "curve_kwargs",
    [None, {"color": "red"}, [{"c": "red"}, {"c": "green"}, {"c": "yellow"}]],
)
def test_display_from_cv_results_curve_kwargs_default_kwargs(
    pyplot, data_binary, curve_kwargs, Display
):
    # [FUNCTION DOC]
    # Verifies that fold lines default to blue, take a shared dict color, or per-curve
    # colors given with the 'c' alias.
    """Check `curve_kwargs` and default color handled correctly in `from_cv_results`."""

    X, y = data_binary
    cv_results = cross_validate(
        LogisticRegression(), X, y, cv=3, return_estimator=True, return_indices=True
    )
    display = Display.from_cv_results(cv_results, X, y, curve_kwargs=curve_kwargs)

    for idx, line in enumerate(display.line_):
        color = line.get_color()
        if curve_kwargs is None:
            # Default color
            assert color == "blue"
        elif isinstance(curve_kwargs, Mapping):
            # All curves "red"
            assert color == "red"
        else:
            assert color == curve_kwargs[idx]["c"]


@pytest.mark.parametrize(
    "Display, display_kwargs",
    [
        # TODO(1.10): Remove
        (
            PrecisionRecallDisplay,
            {"precision": np.array([1, 0.5, 0]), "recall": np.array([0, 0.5, 1])},
        ),
    ],
)
def test_display_estimator_name_deprecation(pyplot, Display, display_kwargs):
    # [FUNCTION DOC]
    # Verifies that passing estimator_name to the PrecisionRecallDisplay constructor
    # emits a FutureWarning.
    """Check deprecation of `estimator_name`."""
    with pytest.warns(FutureWarning, match="`estimator_name` is deprecated in"):
        Display(**display_kwargs, estimator_name="test")


@pytest.mark.parametrize(
    "Display, display_kwargs",
    [
        # TODO(1.11): Remove
        (
            PrecisionRecallDisplay,
            {"precision": np.array([1, 0.5, 0]), "recall": np.array([0, 0.5, 1])},
        ),
    ],
)
@pytest.mark.parametrize(
    "constructor_name", ["from_estimator", "from_predictions", "plot"]
)
def test_display_kwargs_deprecation(
    pyplot, data_binary, constructor_name, Display, display_kwargs
):
    # [FUNCTION DOC]
    # Verifies for from_estimator, from_predictions and plot that passing both
    # curve_kwargs and extra matplotlib kwargs raises ValueError, and extra kwargs alone
    # emit a FutureWarning.
    """Check **kwargs deprecated correctly in favour of `curve_kwargs`."""
    X, y = data_binary
    lr = LogisticRegression()
    lr.fit(X, y)

    # Error when both `curve_kwargs` and `**kwargs` provided
    with pytest.raises(ValueError, match="Cannot provide both `curve_kwargs`"):
        if constructor_name == "from_estimator":
            Display.from_estimator(lr, X, y, curve_kwargs={"alpha": 1}, label="test")
        elif constructor_name == "from_predictions":
            Display.from_predictions(y, y, curve_kwargs={"alpha": 1}, label="test")
        else:  # constructor_name = "plot"
            Display(**display_kwargs).plot(curve_kwargs={"alpha": 1}, label="test")

    # Warning when `**kwargs`` provided
    with pytest.warns(FutureWarning, match=r"`\*\*kwargs` is deprecated and will be"):
        if constructor_name == "from_estimator":
            Display.from_estimator(lr, X, y, label="test")
        elif constructor_name == "from_predictions":
            Display.from_predictions(y, y, label="test")
        else:  # constructor_name = "plot"
            Display(**display_kwargs).plot(label="test")
