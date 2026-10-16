# [MODULE DOC]
# Test suite for the label utilities in sklearn.preprocessing._label: LabelBinarizer,
#   LabelEncoder, MultiLabelBinarizer, label_binarize and the private inverse helpers
#   _inverse_binarize_multiclass / _inverse_binarize_thresholding, including sparse
#   containers, pandas nullable dtypes and array API (namespace/device) compliance.
# Side effects: loads the iris dataset at import time into the module global iris.
# KNOWN ISSUE: the module-level iris dataset is never used by any test; it only adds
#   import cost.
import numpy as np
import pytest
from scipy.sparse import issparse

from sklearn import config_context, datasets
from sklearn.preprocessing._label import (
    LabelBinarizer,
    LabelEncoder,
    MultiLabelBinarizer,
    _inverse_binarize_multiclass,
    _inverse_binarize_thresholding,
    label_binarize,
)
from sklearn.utils._array_api import (
    _atol_for_type,
    _convert_to_numpy,
    _is_numpy_namespace,
    get_namespace,
    indexing_dtype,
    move_to,
    yield_namespace_device_dtype_combinations,
)
from sklearn.utils._array_api import (
    device as array_api_device,
)
from sklearn.utils._testing import (
    _array_api_for_tests,
    assert_allclose,
    assert_array_equal,
)
from sklearn.utils.fixes import (
    COO_CONTAINERS,
    CSC_CONTAINERS,
    CSR_CONTAINERS,
    DOK_CONTAINERS,
    LIL_CONTAINERS,
)
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _to_object_array

iris = datasets.load_iris()


def toarray(a):
    # [FUNCTION DOC]
    # HELPER: Densifies the input if it exposes toarray (sparse matrix/array), otherwise
    #   returns it unchanged.
    # Inputs: a - array-like or sparse container. Outputs: dense array or the original
    #   object.
    if hasattr(a, "toarray"):
        a = a.toarray()
    return a


def test_label_binarizer():
    # [FUNCTION DOC]
    # Verifies LabelBinarizer for one-class (all zeros column, dense and sparse output),
    #   two-class (single column) and multi-class (one column per sorted class) inputs,
    #   including classes_ and inverse_transform round trips.
    # one-class case defaults to negative label
    # For dense case:
    inp = ["pos", "pos", "pos", "pos"]
    lb = LabelBinarizer(sparse_output=False)
    expected = np.array([[0, 0, 0, 0]]).T
    got = lb.fit_transform(inp)
    assert_array_equal(lb.classes_, ["pos"])
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)

    # For sparse case:
    lb = LabelBinarizer(sparse_output=True)
    got = lb.fit_transform(inp)
    assert issparse(got)
    assert_array_equal(lb.classes_, ["pos"])
    assert_array_equal(expected, got.toarray())
    assert_array_equal(lb.inverse_transform(got.toarray()), inp)

    lb = LabelBinarizer(sparse_output=False)
    # two-class case
    inp = ["neg", "pos", "pos", "neg"]
    expected = np.array([[0, 1, 1, 0]]).T
    got = lb.fit_transform(inp)
    assert_array_equal(lb.classes_, ["neg", "pos"])
    assert_array_equal(expected, got)

    to_invert = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    assert_array_equal(lb.inverse_transform(to_invert), inp)

    # multi-class case
    inp = ["spam", "ham", "eggs", "ham", "0"]
    expected = np.array(
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
    )
    got = lb.fit_transform(inp)
    assert_array_equal(lb.classes_, ["0", "eggs", "ham", "spam"])
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)


def test_label_binarizer_unseen_labels():
    # [FUNCTION DOC]
    # Verifies labels unseen during fit are encoded as all-zero rows by transform.
    lb = LabelBinarizer()

    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    got = lb.fit_transform(["b", "d", "e"])
    assert_array_equal(expected, got)

    expected = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    )
    got = lb.transform(["a", "b", "c", "d", "e", "f"])
    assert_array_equal(expected, got)


def test_label_binarizer_set_label_encoding():
    # [FUNCTION DOC]
    # Verifies custom neg_label/pos_label values (-2/0 for binary, -2/2 for multi-class)
    #   in fit_transform output and inverse_transform round trips.
    lb = LabelBinarizer(neg_label=-2, pos_label=0)

    # two-class case with pos_label=0
    inp = np.array([0, 1, 1, 0])
    expected = np.array([[-2, 0, 0, -2]]).T
    got = lb.fit_transform(inp)
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)

    lb = LabelBinarizer(neg_label=-2, pos_label=2)

    # multi-class case
    inp = np.array([3, 2, 1, 2, 0])
    expected = np.array(
        [
            [-2, -2, -2, +2],
            [-2, -2, +2, -2],
            [-2, +2, -2, -2],
            [-2, -2, +2, -2],
            [+2, -2, -2, -2],
        ]
    )
    got = lb.fit_transform(inp)
    assert_array_equal(expected, got)
    assert_array_equal(lb.inverse_transform(got), inp)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
@pytest.mark.parametrize("unique_first", [True, False])
def test_label_binarizer_pandas_nullable(dtype, unique_first):
    # [FUNCTION DOC]
    # Non-regression for gh-25637: LabelBinarizer fits on pandas nullable dtypes (Int64,
    #   Float64, boolean), either as a Series or as the pandas array returned by unique,
    #   and transforms [1, 0] to [[1], [0]]. Skipped without pandas.
    """Checks that LabelBinarizer works with pandas nullable dtypes.

    Non-regression test for gh-25637.
    """
    pd = pytest.importorskip("pandas")

    y_true = pd.Series([1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=dtype)
    if unique_first:
        # Calling unique creates a pandas array which has a different interface
        # compared to a pandas Series. Specifically, pandas arrays do not have "iloc".
        y_true = y_true.unique()
    lb = LabelBinarizer().fit(y_true)
    y_out = lb.transform([1, 0])

    assert_array_equal(y_out, [[1], [0]])


def test_label_binarizer_errors():
    # [FUNCTION DOC]
    # Verifies ValueErrors: legacy multi-label sequences in transform/fit_transform, use
    #   before fit, neg_label not strictly less than pos_label, sparse_output with non-
    #   zero neg_label, 2-D y with output_type "binary" in
    #   _inverse_binarize_thresholding, and multioutput targets in fit and
    #   label_binarize.
    # Check that invalid arguments yield ValueError
    one_class = np.array([0, 0, 0, 0])
    lb = LabelBinarizer().fit(one_class)

    multi_label = [(2, 3), (0,), (0, 2)]
    err_msg = "You appear to be using a legacy multi-label data representation."
    with pytest.raises(ValueError, match=err_msg):
        lb.transform(multi_label)

    lb = LabelBinarizer()
    err_msg = "This LabelBinarizer instance is not fitted yet"
    with pytest.raises(ValueError, match=err_msg):
        lb.transform([])
    with pytest.raises(ValueError, match=err_msg):
        lb.inverse_transform([])

    input_labels = [0, 1, 0, 1]
    err_msg = "neg_label=2 must be strictly less than pos_label=1."
    lb = LabelBinarizer(neg_label=2, pos_label=1)
    with pytest.raises(ValueError, match=err_msg):
        lb.fit(input_labels)
    err_msg = "neg_label=2 must be strictly less than pos_label=2."
    lb = LabelBinarizer(neg_label=2, pos_label=2)
    with pytest.raises(ValueError, match=err_msg):
        lb.fit(input_labels)
    err_msg = (
        "Sparse binarization is only supported with non zero pos_label and zero "
        "neg_label, got pos_label=2 and neg_label=1"
    )
    lb = LabelBinarizer(neg_label=1, pos_label=2, sparse_output=True)
    with pytest.raises(ValueError, match=err_msg):
        lb.fit(input_labels)

    # Sequence of seq type should raise ValueError
    y_seq_of_seqs = [[], [1, 2], [3], [0, 1, 3], [2]]
    err_msg = "You appear to be using a legacy multi-label data representation"
    with pytest.raises(ValueError, match=err_msg):
        LabelBinarizer().fit_transform(y_seq_of_seqs)

    # Fail on the dimension of 'binary'
    err_msg = "output_type='binary', but y.shape"
    with pytest.raises(ValueError, match=err_msg):
        _inverse_binarize_thresholding(
            y=np.array([[1, 2, 3], [2, 1, 3]]),
            output_type="binary",
            classes=[1, 2, 3],
            threshold=0,
        )

    # Fail on multioutput data
    err_msg = "Multioutput target data is not supported with label binarization"
    with pytest.raises(ValueError, match=err_msg):
        LabelBinarizer().fit(np.array([[1, 3], [2, 1]]))
    with pytest.raises(ValueError, match=err_msg):
        label_binarize(np.array([[1, 3], [2, 1]]), classes=[1, 2, 3])


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_label_binarizer_sparse_errors(csr_container):
    # [FUNCTION DOC]
    # Verifies _inverse_binarize_thresholding on sparse input raises for an unsupported
    #   output_type and for a class count differing from the number of columns.
    # KNOWN ISSUE: the second case also passes output_type "foo", so it only checks the
    #   class-count message wins; it does not isolate that check with a valid
    #   output_type.
    # Fail on y_type
    err_msg = "foo format is not supported"
    with pytest.raises(ValueError, match=err_msg):
        _inverse_binarize_thresholding(
            y=csr_container([[1, 2], [2, 1]]),
            output_type="foo",
            classes=[1, 2],
            threshold=0,
        )

    # Fail on the number of classes
    err_msg = "The number of class is not equal to the number of dimension of y."
    with pytest.raises(ValueError, match=err_msg):
        _inverse_binarize_thresholding(
            y=csr_container([[1, 2], [2, 1]]),
            output_type="foo",
            classes=[1, 2, 3],
            threshold=0,
        )


@pytest.mark.parametrize(
    "y, classes, expected",
    [
        [[1, 0, 0, 1], [0, 1], [[1], [0], [0], [1]]],
        [
            [1, 0, 2, 9],
            [0, 1, 2, 9],
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        ],
    ],
)
@pytest.mark.parametrize(
    "array_namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
def test_label_binarizer_array_api_compliance(
    y, classes, expected, array_namespace, device_name, dtype_name
):
    # [FUNCTION DOC]
    # Verifies LabelBinarizer under array_api_dispatch: for non-NumPy namespaces,
    #   sparse_output=True fit/transform and inverse_transform after a sparse fit raise
    #   ValueError; otherwise fit_transform, classes_ and inverse_transform keep the
    #   namespace, device and an integer dtype and match expected values.
    # Inputs: y, classes, expected - labels, expected classes_ and indicator matrix;
    #   array_namespace, device_name, dtype_name - array API combination.
    # Side effects: sets lb_sparse.sparse_input_ manually to simulate a sparse fit.
    """Test that :class:`LabelBinarizer` works correctly with the array API for binary
    and multi-class inputs for numerical labels and non-sparse outputs.
    """
    xp, device = _array_api_for_tests(array_namespace, device_name, dtype_name)

    y_np = np.asarray(y)

    with config_context(array_api_dispatch=True):
        y = xp.asarray(y, device=device)

        # `sparse_output=True` is not allowed for non-NumPy namespaces.
        # Similarly, if `LabelBinarizer` is fitted on a sparse matrix,
        # then inverse-transforming non-NumPy arrays is not allowed.
        if not _is_numpy_namespace(xp):
            sparse_output_msg = "`sparse_output=True` is not supported for array API"

            with pytest.raises(ValueError, match=sparse_output_msg):
                LabelBinarizer(sparse_output=True).fit(y)

            lb_np = LabelBinarizer(sparse_output=True).fit(y_np)
            with pytest.raises(ValueError, match=sparse_output_msg):
                lb_np.transform(y)

            lb_sparse = LabelBinarizer().fit(y_np)
            lb_sparse.sparse_input_ = True
            sparse_input_msg = (
                "`LabelBinarizer` was fitted on a sparse matrix, and therefore cannot"
            )
            with pytest.raises(ValueError, match=sparse_input_msg):
                lb_sparse.inverse_transform(xp.asarray(expected, device=device))

        # Shouldn't raise error in both `fit` and `transform` when `sparse_output=False`
        lb_xp = LabelBinarizer()

        binarized = lb_xp.fit_transform(y)
        assert get_namespace(binarized)[0].__name__ == xp.__name__
        assert "int" in str(binarized.dtype)
        assert array_api_device(binarized) == array_api_device(y)
        assert_array_equal(
            move_to(binarized, xp=np, device="cpu"), np.asarray(expected)
        )

        fitted_classes = lb_xp.classes_
        assert get_namespace(fitted_classes)[0].__name__ == xp.__name__
        assert array_api_device(fitted_classes) == array_api_device(y)
        assert "int" in str(fitted_classes.dtype)
        assert_array_equal(
            move_to(fitted_classes, xp=np, device="cpu"), np.asarray(classes)
        )

        expected_xp = xp.asarray(expected, device=device)
        binarized_inverse = lb_xp.inverse_transform(expected_xp)
        assert get_namespace(binarized_inverse)[0].__name__ == xp.__name__
        assert "int" in str(binarized_inverse.dtype)
        assert array_api_device(binarized_inverse) == array_api_device(y)
        assert_array_equal(
            move_to(binarized_inverse, xp=np, device="cpu"),
            move_to(y, xp=np, device="cpu"),
        )


@pytest.mark.parametrize(
    "values, classes, unknown",
    [
        (
            np.array([2, 1, 3, 1, 3], dtype="int64"),
            np.array([1, 2, 3], dtype="int64"),
            np.array([4], dtype="int64"),
        ),
        (
            np.array(["b", "a", "c", "a", "c"], dtype=object),
            np.array(["a", "b", "c"], dtype=object),
            np.array(["d"], dtype=object),
        ),
        (
            np.array(["b", "a", "c", "a", "c"]),
            np.array(["a", "b", "c"]),
            np.array(["d"]),
        ),
    ],
    ids=["int64", "object", "str"],
)
def test_label_encoder(values, classes, unknown):
    # [FUNCTION DOC]
    # Verifies LabelEncoder fit/classes_, transform, inverse_transform and fit_transform
    #   on int64, object and str arrays, and that transforming an unseen label raises
    #   ValueError.
    # Test LabelEncoder's transform, fit_transform and
    # inverse_transform methods
    le = LabelEncoder()
    le.fit(values)
    assert_array_equal(le.classes_, classes)
    assert_array_equal(le.transform(values), [1, 0, 2, 0, 2])
    assert_array_equal(le.inverse_transform([1, 0, 2, 0, 2]), values)
    le = LabelEncoder()
    ret = le.fit_transform(values)
    assert_array_equal(ret, [1, 0, 2, 0, 2])

    with pytest.raises(ValueError, match="unseen labels"):
        le.transform(unknown)


def test_label_encoder_negative_ints():
    # [FUNCTION DOC]
    # Verifies LabelEncoder sorts and encodes negative integers correctly, inverts them,
    #   and raises ValueError on an unseen label.
    le = LabelEncoder()
    le.fit([1, 1, 4, 5, -1, 0])
    assert_array_equal(le.classes_, [-1, 0, 1, 4, 5])
    assert_array_equal(le.transform([0, 1, 4, 4, 5, -1, -1]), [1, 2, 3, 3, 4, 0, 0])
    assert_array_equal(
        le.inverse_transform([1, 2, 3, 3, 4, 0, 0]), [0, 1, 4, 4, 5, -1, -1]
    )
    with pytest.raises(ValueError):
        le.transform([0, 6])


@pytest.mark.parametrize("dtype", ["str", "object"])
def test_label_encoder_str_bad_shape(dtype):
    # [FUNCTION DOC]
    # Verifies transform on a scalar string raises "should be a 1d array" for str and
    #   object fitted classes.
    le = LabelEncoder()
    le.fit(np.array(["apple", "orange"], dtype=dtype))
    msg = "should be a 1d array"
    with pytest.raises(ValueError, match=msg):
        le.transform("apple")


def test_label_encoder_errors():
    # [FUNCTION DOC]
    # Verifies ValueErrors for transform/inverse_transform before fit, inverse_transform
    #   with codes outside the fitted range, and inverse_transform on a scalar empty
    #   string.
    # Check that invalid arguments yield ValueError
    le = LabelEncoder()
    with pytest.raises(ValueError):
        le.transform([])
    with pytest.raises(ValueError):
        le.inverse_transform([])

    # Fail on unseen labels
    le = LabelEncoder()
    le.fit([1, 2, 3, -1, 1])
    msg = "contains previously unseen labels"
    with pytest.raises(ValueError, match=msg):
        le.inverse_transform([-2])
    with pytest.raises(ValueError, match=msg):
        le.inverse_transform([-2, -3, -4])

    # Fail on inverse_transform("")
    msg = r"should be a 1d array.+shape \(\)"
    with pytest.raises(ValueError, match=msg):
        le.inverse_transform("")


@pytest.mark.parametrize(
    "values",
    [
        np.array([2, 1, 3, 1, 3], dtype="int64"),
        np.array(["b", "a", "c", "a", "c"], dtype=object),
        np.array(["b", "a", "c", "a", "c"]),
    ],
    ids=["int64", "object", "str"],
)
def test_label_encoder_empty_array(values):
    # [FUNCTION DOC]
    # Verifies transform and inverse_transform of an empty list return empty arrays
    #   after fitting on int64, object or str labels.
    le = LabelEncoder()
    le.fit(values)
    # test empty transform
    transformed = le.transform([])
    assert_array_equal(np.array([]), transformed)
    # test empty inverse transform
    inverse_transformed = le.inverse_transform([])
    assert_array_equal(np.array([]), inverse_transformed)


def test_sparse_output_multilabel_binarizer():
    # [FUNCTION DOC]
    # Verifies MultiLabelBinarizer with lists, sets and iterators of label sets, for
    #   sparse and dense output, via fit_transform and fit+transform: indicator matrix,
    #   classes_, inverse_transform, and that sparse output has matching indices/indptr
    #   dtypes.
    # test input as iterable of iterables
    inputs = [
        lambda: [(2, 3), (1,), (1, 2)],
        lambda: ({2, 3}, {1}, {1, 2}),
        lambda: iter([iter((2, 3)), iter((1,)), {1, 2}]),
    ]
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])

    inverse = inputs[0]()
    for sparse_output in [True, False]:
        for inp in inputs:
            # With fit_transform
            mlb = MultiLabelBinarizer(sparse_output=sparse_output)
            got = mlb.fit_transform(inp())
            assert issparse(got) == sparse_output
            if sparse_output:
                # verify CSR assumption that indices and indptr have same dtype
                assert got.indices.dtype == got.indptr.dtype
                got = got.toarray()
            assert_array_equal(indicator_mat, got)
            assert_array_equal([1, 2, 3], mlb.classes_)
            assert mlb.inverse_transform(got) == inverse

            # With fit
            mlb = MultiLabelBinarizer(sparse_output=sparse_output)
            got = mlb.fit(inp()).transform(inp())
            assert issparse(got) == sparse_output
            if sparse_output:
                # verify CSR assumption that indices and indptr have same dtype
                assert got.indices.dtype == got.indptr.dtype
                got = got.toarray()
            assert_array_equal(indicator_mat, got)
            assert_array_equal([1, 2, 3], mlb.classes_)
            assert mlb.inverse_transform(got) == inverse


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_sparse_output_multilabel_binarizer_errors(csr_container):
    # [FUNCTION DOC]
    # Verifies inverse_transform raises ValueError for a sparse indicator containing a
    #   non-binary value.
    inp = iter([iter((2, 3)), iter((1,)), {1, 2}])
    mlb = MultiLabelBinarizer(sparse_output=False)
    mlb.fit(inp)
    with pytest.raises(ValueError):
        mlb.inverse_transform(
            csr_container(np.array([[0, 1, 1], [2, 0, 0], [1, 1, 0]]))
        )


def test_multilabel_binarizer():
    # [FUNCTION DOC]
    # Verifies MultiLabelBinarizer dense output, classes_ and inverse_transform for
    #   list, set and iterator inputs via fit_transform and fit+transform.
    # test input as iterable of iterables
    inputs = [
        lambda: [(2, 3), (1,), (1, 2)],
        lambda: ({2, 3}, {1}, {1, 2}),
        lambda: iter([iter((2, 3)), iter((1,)), {1, 2}]),
    ]
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])
    inverse = inputs[0]()
    for inp in inputs:
        # With fit_transform
        mlb = MultiLabelBinarizer()
        got = mlb.fit_transform(inp())
        assert_array_equal(indicator_mat, got)
        assert_array_equal([1, 2, 3], mlb.classes_)
        assert mlb.inverse_transform(got) == inverse

        # With fit
        mlb = MultiLabelBinarizer()
        got = mlb.fit(inp()).transform(inp())
        assert_array_equal(indicator_mat, got)
        assert_array_equal([1, 2, 3], mlb.classes_)
        assert mlb.inverse_transform(got) == inverse


def test_multilabel_binarizer_empty_sample():
    # [FUNCTION DOC]
    # Verifies an empty label set yields an all-zero row.
    mlb = MultiLabelBinarizer()
    y = [[1, 2], [1], []]
    Y = np.array([[1, 1], [1, 0], [0, 0]])
    assert_array_equal(mlb.fit_transform(y), Y)


def test_multilabel_binarizer_unknown_class():
    # [FUNCTION DOC]
    # Verifies transform warns that unknown classes are ignored, with inferred classes
    #   and with explicit classes, and that the explicit-classes output ignores the
    #   unknown label.
    # KNOWN ISSUE: the result of the first (inferred classes) transform and the first Y
    #   are never asserted.
    mlb = MultiLabelBinarizer()
    y = [[1, 2]]
    Y = np.array([[1, 0], [0, 1]])
    warning_message = "unknown class.* will be ignored"
    with pytest.warns(UserWarning, match=warning_message):
        matrix = mlb.fit(y).transform([[4, 1], [2, 0]])

    Y = np.array([[1, 0, 0], [0, 1, 0]])
    mlb = MultiLabelBinarizer(classes=[1, 2, 3])
    with pytest.warns(UserWarning, match=warning_message):
        matrix = mlb.fit(y).transform([[4, 1], [2, 0]])
    assert_array_equal(matrix, Y)


def test_multilabel_binarizer_given_classes():
    # [FUNCTION DOC]
    # Verifies explicit classes control column order, extra classes give zero columns,
    #   fit does not consume an iterator when classes are given, and duplicate classes
    #   raise ValueError.
    inp = [(2, 3), (1,), (1, 2)]
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
    # fit_transform()
    mlb = MultiLabelBinarizer(classes=[1, 3, 2])
    assert_array_equal(mlb.fit_transform(inp), indicator_mat)
    assert_array_equal(mlb.classes_, [1, 3, 2])

    # fit().transform()
    mlb = MultiLabelBinarizer(classes=[1, 3, 2])
    assert_array_equal(mlb.fit(inp).transform(inp), indicator_mat)
    assert_array_equal(mlb.classes_, [1, 3, 2])

    # ensure works with extra class
    mlb = MultiLabelBinarizer(classes=[4, 1, 3, 2])
    assert_array_equal(
        mlb.fit_transform(inp), np.hstack(([[0], [0], [0]], indicator_mat))
    )
    assert_array_equal(mlb.classes_, [4, 1, 3, 2])

    # ensure fit is no-op as iterable is not consumed
    inp = iter(inp)
    mlb = MultiLabelBinarizer(classes=[1, 3, 2])
    assert_array_equal(mlb.fit(inp).transform(inp), indicator_mat)

    # ensure a ValueError is thrown if given duplicate classes
    err_msg = (
        "The classes argument contains duplicate classes. Remove "
        "these duplicates before passing them to MultiLabelBinarizer."
    )
    mlb = MultiLabelBinarizer(classes=[1, 3, 2, 3])
    with pytest.raises(ValueError, match=err_msg):
        mlb.fit(inp)


def test_multilabel_binarizer_multiple_calls():
    # [FUNCTION DOC]
    # Verifies changing the classes parameter between fit_transform calls changes the
    #   column order of the output.
    inp = [(2, 3), (1,), (1, 2)]
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])

    indicator_mat2 = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])

    # first call
    mlb = MultiLabelBinarizer(classes=[1, 3, 2])
    assert_array_equal(mlb.fit_transform(inp), indicator_mat)
    # second call change class
    mlb.classes = [1, 2, 3]
    assert_array_equal(mlb.fit_transform(inp), indicator_mat2)


def test_multilabel_binarizer_same_length_sequence():
    # [FUNCTION DOC]
    # Verifies equal-length label sequences are not interpreted as a 2-D array and
    #   round-trip through inverse_transform.
    # Ensure sequences of the same length are not interpreted as a 2-d array
    inp = [[1], [0], [2]]
    indicator_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    # fit_transform()
    mlb = MultiLabelBinarizer()
    assert_array_equal(mlb.fit_transform(inp), indicator_mat)
    assert_array_equal(mlb.inverse_transform(indicator_mat), inp)

    # fit().transform()
    mlb = MultiLabelBinarizer()
    assert_array_equal(mlb.fit(inp).transform(inp), indicator_mat)
    assert_array_equal(mlb.inverse_transform(indicator_mat), inp)


def test_multilabel_binarizer_non_integer_labels():
    # [FUNCTION DOC]
    # Verifies string and tuple labels (object arrays) are binarized and inverted
    #   correctly, and that unhashable dict labels raise TypeError.
    tuple_classes = _to_object_array([(1,), (2,), (3,)])
    inputs = [
        ([("2", "3"), ("1",), ("1", "2")], ["1", "2", "3"]),
        ([("b", "c"), ("a",), ("a", "b")], ["a", "b", "c"]),
        ([((2,), (3,)), ((1,),), ((1,), (2,))], tuple_classes),
    ]
    indicator_mat = np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]])
    for inp, classes in inputs:
        # fit_transform()
        mlb = MultiLabelBinarizer()
        inp = np.array(inp, dtype=object)
        assert_array_equal(mlb.fit_transform(inp), indicator_mat)
        assert_array_equal(mlb.classes_, classes)
        indicator_mat_inv = np.array(mlb.inverse_transform(indicator_mat), dtype=object)
        assert_array_equal(indicator_mat_inv, inp)

        # fit().transform()
        mlb = MultiLabelBinarizer()
        assert_array_equal(mlb.fit(inp).transform(inp), indicator_mat)
        assert_array_equal(mlb.classes_, classes)
        indicator_mat_inv = np.array(mlb.inverse_transform(indicator_mat), dtype=object)
        assert_array_equal(indicator_mat_inv, inp)

    mlb = MultiLabelBinarizer()
    with pytest.raises(TypeError):
        mlb.fit_transform([({}), ({}, {"a": "b"})])


def test_multilabel_binarizer_non_unique():
    # [FUNCTION DOC]
    # Verifies duplicate labels within one sample give a single 1 per class.
    inp = [(1, 1, 1, 0)]
    indicator_mat = np.array([[1, 1]])
    mlb = MultiLabelBinarizer()
    assert_array_equal(mlb.fit_transform(inp), indicator_mat)


def test_multilabel_binarizer_inverse_validation():
    # [FUNCTION DOC]
    # Verifies inverse_transform accepts binary indicator rows but raises ValueError for
    #   non-binary values or a wrong number of columns.
    inp = [(1, 1, 1, 0)]
    mlb = MultiLabelBinarizer()
    mlb.fit_transform(inp)
    # Not binary
    with pytest.raises(ValueError):
        mlb.inverse_transform(np.array([[1, 3]]))
    # The following binary cases are fine, however
    mlb.inverse_transform(np.array([[0, 0]]))
    mlb.inverse_transform(np.array([[1, 1]]))
    mlb.inverse_transform(np.array([[1, 0]]))

    # Wrong shape
    with pytest.raises(ValueError):
        mlb.inverse_transform(np.array([[1]]))
    with pytest.raises(ValueError):
        mlb.inverse_transform(np.array([[1, 1, 1]]))


def test_label_binarize_with_class_order():
    # [FUNCTION DOC]
    # Verifies label_binarize respects the order of the classes argument for columns.
    out = label_binarize([1, 6], classes=[1, 2, 4, 6])
    expected = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])
    assert_array_equal(out, expected)

    # Modified class order
    out = label_binarize([1, 6], classes=[1, 6, 4, 2])
    expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    assert_array_equal(out, expected)

    out = label_binarize([0, 1, 2, 3], classes=[3, 2, 0, 1])
    expected = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]])
    assert_array_equal(out, expected)


def check_binarized_results(y, classes, pos_label, neg_label, expected):
    # [FUNCTION DOC]
    # HELPER: For sparse and dense output, checks label_binarize and LabelBinarizer give
    #   expected (or raise ValueError when sparse output is requested with pos_label 0
    #   or non-zero neg_label), and that the inverse via _inverse_binarize_multiclass or
    #   _inverse_binarize_thresholding (threshold midway between labels) and
    #   LabelBinarizer.inverse_transform reproduce y, preserving sparsity of y.
    # Inputs: y - labels or indicator matrix (dense/sparse); classes - class list;
    #   pos_label, neg_label - ints; expected - dense expected output.
    # Exceptions: assertion errors on mismatch.
    for sparse_output in [True, False]:
        if (pos_label == 0 or neg_label != 0) and sparse_output:
            with pytest.raises(ValueError):
                label_binarize(
                    y,
                    classes=classes,
                    neg_label=neg_label,
                    pos_label=pos_label,
                    sparse_output=sparse_output,
                )
            continue

        # check label_binarize
        binarized = label_binarize(
            y,
            classes=classes,
            neg_label=neg_label,
            pos_label=pos_label,
            sparse_output=sparse_output,
        )
        assert_array_equal(toarray(binarized), expected)
        assert issparse(binarized) == sparse_output

        # check inverse
        y_type = type_of_target(y)
        if y_type == "multiclass":
            inversed = _inverse_binarize_multiclass(binarized, classes=classes)

        else:
            inversed = _inverse_binarize_thresholding(
                binarized,
                output_type=y_type,
                classes=classes,
                threshold=((neg_label + pos_label) / 2.0),
            )

        assert_array_equal(toarray(inversed), toarray(y))

        # Check label binarizer
        lb = LabelBinarizer(
            neg_label=neg_label, pos_label=pos_label, sparse_output=sparse_output
        )
        binarized = lb.fit_transform(y)
        assert_array_equal(toarray(binarized), expected)
        assert issparse(binarized) == sparse_output
        inverse_output = lb.inverse_transform(binarized)
        assert_array_equal(toarray(inverse_output), toarray(y))
        assert issparse(inverse_output) == issparse(y)


def test_label_binarize_binary():
    # [FUNCTION DOC]
    # Verifies binary label_binarize/LabelBinarizer with pos_label=2, neg_label=-1
    #   (sparse rejected) and pos_label=3, neg_label=0 (sparse allowed), using
    #   check_binarized_results.
    y = [0, 1, 0]
    classes = [0, 1]
    pos_label = 2
    neg_label = -1
    expected = np.array([[2, -1], [-1, 2], [2, -1]])[:, 1].reshape((-1, 1))

    check_binarized_results(y, classes, pos_label, neg_label, expected)

    # Binary case where sparse_output = True will not result in a ValueError
    y = [0, 1, 0]
    classes = [0, 1]
    pos_label = 3
    neg_label = 0
    expected = np.array([[3, 0], [0, 3], [3, 0]])[:, 1].reshape((-1, 1))

    check_binarized_results(y, classes, pos_label, neg_label, expected)


def test_label_binarize_multiclass():
    # [FUNCTION DOC]
    # Verifies multi-class binarization with pos_label=2 via check_binarized_results and
    #   that sparse output with neg_label=-1 raises ValueError.
    y = [0, 1, 2]
    classes = [0, 1, 2]
    pos_label = 2
    neg_label = 0
    expected = 2 * np.eye(3)

    check_binarized_results(y, classes, pos_label, neg_label, expected)

    with pytest.raises(ValueError):
        label_binarize(
            y, classes=classes, neg_label=-1, pos_label=pos_label, sparse_output=True
        )


@pytest.mark.parametrize(
    "arr_type",
    [np.array]
    + COO_CONTAINERS
    + CSC_CONTAINERS
    + CSR_CONTAINERS
    + DOK_CONTAINERS
    + LIL_CONTAINERS,
)
def test_label_binarize_multilabel(arr_type):
    # [FUNCTION DOC]
    # Verifies multilabel indicator input in dense and every sparse format is scaled by
    #   pos_label, round-trips, and sparse output with neg_label=-1 raises ValueError.
    y_ind = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]])
    classes = [0, 1, 2]
    pos_label = 2
    neg_label = 0
    expected = pos_label * y_ind
    y = arr_type(y_ind)

    check_binarized_results(y, classes, pos_label, neg_label, expected)

    with pytest.raises(ValueError):
        label_binarize(
            y, classes=classes, neg_label=-1, pos_label=pos_label, sparse_output=True
        )


def test_invalid_input_label_binarize():
    # [FUNCTION DOC]
    # Verifies label_binarize raises ValueError for pos_label 0 with neg_label 1,
    #   continuous targets, and a multilabel row whose width mismatches classes.
    with pytest.raises(ValueError):
        label_binarize([0, 2], classes=[0, 2], pos_label=0, neg_label=1)
    with pytest.raises(ValueError, match="continuous target data is not "):
        label_binarize([1.2, 2.7], classes=[0, 1])
    with pytest.raises(ValueError, match="mismatch with the labels"):
        label_binarize([[1, 3]], classes=[1, 2, 3])


@pytest.mark.parametrize(
    "y, classes, expected",
    [
        [[1, 0, 0, 1], ["yes", "no"], [[0], [0], [0], [0]]],
        [
            [1, 0, 2, 9],
            ["bird", "cat", "dog"],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        [[1, 0, 0, 1], [0, 1], [[1], [0], [0], [1]]],
        [[1, 0, 2, 1], [0, 1, 2], [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]],
    ],
)
@pytest.mark.parametrize(
    "array_namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
def test_label_binarize_array_api_compliance(
    y, classes, expected, array_namespace, device_name, dtype_name
):
    # [FUNCTION DOC]
    # Verifies label_binarize under array_api_dispatch: integer y and classes produce an
    #   integer indicator in the input namespace/device equal to expected (and
    #   sparse_output raises for non-NumPy); string classes raise an unsupported-dtype
    #   ValueError for non-NumPy namespaces. Labels absent from classes give zero rows.
    """Test that :func:`label_binarize` works correctly with the array API for binary
    and multi-class inputs for numerical labels and non-sparse outputs.
    """
    xp, device = _array_api_for_tests(array_namespace, device_name, dtype_name)
    xp_is_numpy = _is_numpy_namespace(xp)
    numeric_dtype = np.issubdtype(np.asarray(y).dtype, np.integer) and np.issubdtype(
        np.asarray(classes).dtype, np.integer
    )

    with config_context(array_api_dispatch=True):
        y = xp.asarray(y, device=device)

        if numeric_dtype:
            # `sparse_output=True` is not allowed for non-NumPy namespaces
            if not xp_is_numpy:
                msg = "`sparse_output=True` is not supported for array API "
                with pytest.raises(ValueError, match=msg):
                    label_binarize(y=y, classes=classes, sparse_output=True)

            # Numeric class labels should not raise any errors for non-NumPy namespaces
            binarized = label_binarize(y, classes=classes)
            expected = np.asarray(expected, dtype=int)

            assert get_namespace(binarized)[0].__name__ == xp.__name__
            assert array_api_device(binarized) == array_api_device(y)
            assert "int" in str(binarized.dtype)
            assert_array_equal(move_to(binarized, xp=np, device="cpu"), expected)

        if not xp_is_numpy and not numeric_dtype:
            msg = "`classes` contains unsupported dtype for array API "
            with pytest.raises(ValueError, match=msg):
                label_binarize(y=y, classes=classes)


@pytest.mark.parametrize(
    "array_namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
@pytest.mark.parametrize("classes", [[0, 1], [0, 1, 2]])
def test_label_binarize_unsigned_integer_overflow(
    array_namespace, device_name, dtype_name, classes
):
    # [FUNCTION DOC]
    # Verifies label_binarize with neg_label=-1 on unsigned integer y (uint8, plus
    #   uint16/32/64 except torch) matches the signed-integer baseline, i.e. -1 does not
    #   wrap around.
    """Ensure label_binarize does not overflow when y has unsigned integer dtype.

    In particular, verify that label_binarize does not wrap -1 to the maximum value
    of unsigned integer dtypes.
    """
    xp, device = _array_api_for_tests(array_namespace, device_name)
    y = classes * 10

    with config_context(array_api_dispatch=True):
        # Stable signed baseline
        signed_dtype = indexing_dtype(xp)
        y_signed = xp.asarray(y, dtype=signed_dtype, device=device)
        desired = label_binarize(y_signed, classes=classes, pos_label=1, neg_label=-1)

        # All namespace support `unit8` dtype
        uint_dtypes = [xp.uint8]

        # PyTorch doesn't fully support `uint16`, `uint32`, `uint64`.
        # See https://github.com/pytorch/pytorch/issues/58734
        if "torch" not in xp.__name__:
            uint_dtypes += [xp.uint16, xp.uint32, xp.uint64]

        for uint_dtype in uint_dtypes:
            y_uint = xp.asarray(y, dtype=uint_dtype, device=device)
            actual = label_binarize(y_uint, classes=classes, pos_label=1, neg_label=-1)

            assert_allclose(
                _convert_to_numpy(actual, xp=xp),
                _convert_to_numpy(desired, xp=xp),
                atol=_atol_for_type(dtype_name),
            )


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_inverse_binarize_multiclass(csr_container):
    # [FUNCTION DOC]
    # Verifies _inverse_binarize_multiclass on a sparse score matrix picks the argmax
    #   column per row (including rows of negatives and all-zero rows).
    got = _inverse_binarize_multiclass(
        csr_container([[0, 1, 0], [-1, 0, -1], [0, 0, 0]]), np.arange(3)
    )
    assert_array_equal(got, np.array([1, 1, 0]))


def test_nan_label_encoder():
    # [FUNCTION DOC]
    # Non-regression for #22628: LabelEncoder fitted with NaN encodes NaN to its sorted
    #   index (last).
    """Check that label encoder encodes nans in transform.

    Non-regression test for #22628.
    """
    le = LabelEncoder()
    le.fit(["a", "a", "b", np.nan])

    y_trans = le.transform([np.nan])
    assert_array_equal(y_trans, [2])


@pytest.mark.parametrize(
    "encoder", [LabelEncoder(), LabelBinarizer(), MultiLabelBinarizer()]
)
def test_label_encoders_do_not_have_set_output(encoder):
    # [FUNCTION DOC]
    # Non-regression for #26854: label encoders do not expose set_output and
    #   fit_transform accepts y as a keyword with the same result as positional.
    """Check that label encoders do not define set_output and work with y as a kwarg.

    Non-regression test for #26854.
    """
    assert not hasattr(encoder, "set_output")
    y_encoded_with_kwarg = encoder.fit_transform(y=["a", "b", "c"])
    y_encoded_positional = encoder.fit_transform(["a", "b", "c"])
    assert_array_equal(y_encoded_with_kwarg, y_encoded_positional)


@pytest.mark.parametrize(
    "array_namespace, device_name, dtype_name",
    yield_namespace_device_dtype_combinations(),
)
@pytest.mark.parametrize(
    "y",
    [
        np.array([2, 1, 3, 1, 3]),
        np.array([1, 1, 4, 5, -1, 0]),
        np.array([3, 5, 9, 5, 9, 3]),
    ],
)
def test_label_encoder_array_api_compliance(
    y, array_namespace, device_name, dtype_name
):
    # [FUNCTION DOC]
    # Verifies LabelEncoder fit/transform/inverse_transform and fit_transform under
    #   array_api_dispatch keep the input namespace for outputs and classes_ and match
    #   NumPy results.
    xp, device = _array_api_for_tests(array_namespace, device_name, dtype_name)
    xp_y = xp.asarray(y, device=device)
    with config_context(array_api_dispatch=True):
        xp_label = LabelEncoder()
        np_label = LabelEncoder()
        xp_label = xp_label.fit(xp_y)
        xp_transformed = xp_label.transform(xp_y)
        xp_inv_transformed = xp_label.inverse_transform(xp_transformed)
        np_label = np_label.fit(y)
        np_transformed = np_label.transform(y)
        assert get_namespace(xp_transformed)[0].__name__ == xp.__name__
        assert get_namespace(xp_inv_transformed)[0].__name__ == xp.__name__
        assert get_namespace(xp_label.classes_)[0].__name__ == xp.__name__
        assert_array_equal(move_to(xp_transformed, xp=np, device="cpu"), np_transformed)
        assert_array_equal(move_to(xp_inv_transformed, xp=np, device="cpu"), y)
        assert_array_equal(
            move_to(xp_label.classes_, xp=np, device="cpu"), np_label.classes_
        )

        xp_label = LabelEncoder()
        np_label = LabelEncoder()
        xp_transformed = xp_label.fit_transform(xp_y)
        np_transformed = np_label.fit_transform(y)
        assert get_namespace(xp_transformed)[0].__name__ == xp.__name__
        assert get_namespace(xp_label.classes_)[0].__name__ == xp.__name__
        assert_array_equal(move_to(xp_transformed, xp=np, device="cpu"), np_transformed)
        assert_array_equal(
            move_to(xp_label.classes_, xp=np, device="cpu"), np_label.classes_
        )
