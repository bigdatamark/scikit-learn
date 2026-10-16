# [MODULE DOC]
# Python front-end for reading and writing the svmlight / libsvm sparse text format.
# Parsing and writing are delegated to the Cython module _svmlight_format_fast; this
# module handles opening files (plain, .gz, .bz2, file descriptors, file-likes), index-
# base detection, feature-count reconciliation across files, CSR construction and
# argument validation. Public: load_svmlight_file, load_svmlight_files,
# dump_svmlight_file. No import-time side effects.
"""This module implements a loader and dumper for the svmlight format

This format is a text-based format, with one sample per line. It does
not store zero valued features hence is suitable for sparse dataset.

The first element of each line can be used to store a target variable to
predict.

This format is used as the default format for both svmlight and the
libsvm command line programs.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import os.path
from contextlib import closing
from numbers import Integral

import numpy as np
import scipy.sparse as sp

from sklearn import __version__
from sklearn.datasets._svmlight_format_fast import (
    _dump_svmlight_file,
    _load_svmlight_file,
)
from sklearn.utils import check_array
from sklearn.utils._param_validation import (
    HasMethods,
    Interval,
    StrOptions,
    validate_params,
)
from sklearn.utils._sparse import _align_api_if_sparse


@validate_params(
    {
        "f": [
            str,
            Interval(Integral, 0, None, closed="left"),
            os.PathLike,
            HasMethods("read"),
        ],
        "n_features": [Interval(Integral, 1, None, closed="left"), None],
        "dtype": "no_validation",  # delegate validation to numpy
        "multilabel": ["boolean"],
        "zero_based": ["boolean", StrOptions({"auto"})],
        "query_id": ["boolean"],
        "offset": [Interval(Integral, 0, None, closed="left")],
        "length": [Integral],
    },
    prefer_skip_nested_validation=True,
)
def load_svmlight_file(
    f,
    *,
    n_features=None,
    dtype=np.float64,
    multilabel=False,
    zero_based="auto",
    query_id=False,
    offset=0,
    length=-1,
):
    # [FUNCTION DOC]
    # Loads a single svmlight file by delegating to load_svmlight_files with a one-
    # element list.
    # Inputs: f - path, path-like, non-negative file descriptor or binary file-like;
    # n_features - int >= 1 or None (inferred); dtype - data dtype; multilabel - bool;
    # zero_based - bool or "auto"; query_id - bool; offset - byte offset to start
    # reading; length - max bytes to read (negative means all).
    # Outputs: tuple (X sparse CSR, y) or (X, y, query_id) when query_id is True; y is a
    # float64 array or a list of tuples when multilabel.
    # Exceptions: as load_svmlight_files.
    """Load datasets in the svmlight / libsvm format into sparse CSR matrix.

    This format is a text-based format, with one sample per line. It does
    not store zero valued features hence is suitable for sparse dataset.

    The first element of each line can be used to store a target variable
    to predict.

    This format is used as the default format for both svmlight and the
    libsvm command line programs.

    Parsing a text based source can be expensive. When repeatedly
    working on the same dataset, it is recommended to wrap this
    loader with joblib.Memory.cache to store a memmapped backup of the
    CSR results of the first call and benefit from the near instantaneous
    loading of memmapped structures for the subsequent calls.

    In case the file contains a pairwise preference constraint (known
    as "qid" in the svmlight format) these are ignored unless the
    query_id parameter is set to True. These pairwise preference
    constraints can be used to constraint the combination of samples
    when using pairwise loss functions (as is the case in some
    learning to rank problems) so that only pairs with the same
    query_id value are considered.

    This implementation is written in Cython and is reasonably fast.
    However, a faster API-compatible loader is also available at:
    https://github.com/mblondel/svmlight-loader

    Parameters
    ----------
    f : str, path-like, file-like or int
        (Path to) a file to load. If a path ends in ".gz" or ".bz2", it will
        be uncompressed on the fly. If an integer is passed, it is assumed to
        be a file descriptor. A file-like or file descriptor will not be closed
        by this function. A file-like object must be opened in binary mode.

        .. versionchanged:: 1.2
           Path-like objects are now accepted.

    n_features : int, default=None
        The number of features to use. If None, it will be inferred. This
        argument is useful to load several files that are subsets of a
        bigger sliced dataset: each subset might not have examples of
        every feature, hence the inferred shape might vary from one
        slice to another.
        n_features is only required if ``offset`` or ``length`` are passed a
        non-default value.

    dtype : numpy data type, default=np.float64
        Data type of dataset to be loaded. This will be the data type of the
        output numpy arrays ``X`` and ``y``.

    multilabel : bool, default=False
        Samples may have several labels each (see
        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html).

    zero_based : bool or "auto", default="auto"
        Whether column indices in f are zero-based (True) or one-based
        (False). If column indices are one-based, they are transformed to
        zero-based to match Python/NumPy conventions.
        If set to "auto", a heuristic check is applied to determine this from
        the file contents. Both kinds of files occur "in the wild", but they
        are unfortunately not self-identifying. Using "auto" or True should
        always be safe when no ``offset`` or ``length`` is passed.
        If ``offset`` or ``length`` are passed, the "auto" mode falls back
        to ``zero_based=True`` to avoid having the heuristic check yield
        inconsistent results on different segments of the file.

    query_id : bool, default=False
        If True, will return the query_id array for each file.

    offset : int, default=0
        Ignore the offset first bytes by seeking forward, then
        discarding the following bytes up until the next new line
        character.

    length : int, default=-1
        If strictly positive, stop reading any new line of data once the
        position in the file has reached the (offset + length) bytes threshold.

    Returns
    -------
    X : scipy.sparse matrix of shape (n_samples, n_features)
        The data matrix.

    y : ndarray of shape (n_samples,), or a list of tuples of length n_samples
        The target. It is a list of tuples when ``multilabel=True``, else a
        ndarray.

    query_id : array of shape (n_samples,)
       The query_id for each sample. Only returned when query_id is set to
       True.

    See Also
    --------
    load_svmlight_files : Similar function for loading multiple files in this
        format, enforcing the same number of features/columns on all of them.

    Examples
    --------
    To use joblib.Memory to cache the svmlight file::

        from joblib import Memory
        from sklearn.datasets import load_svmlight_file
        mem = Memory("./mycache")

        @mem.cache
        def get_data():
            data = load_svmlight_file("mysvmlightfile")
            return data[0], data[1]

        X, y = get_data()
    """
    return tuple(
        load_svmlight_files(
            [f],
            n_features=n_features,
            dtype=dtype,
            multilabel=multilabel,
            zero_based=zero_based,
            query_id=query_id,
            offset=offset,
            length=length,
        )
    )


def _gen_open(f):
    # [FUNCTION DOC]
    # HELPER: Opens a path or file descriptor for binary reading, choosing the
    # decompressor from the extension.
    # Inputs: f - int file descriptor (opened without closing the fd), os.PathLike or
    # str path.
    # Outputs: binary file object (gzip for .gz, BZ2File for .bz2, plain otherwise).
    # Exceptions: TypeError for any other input type; OSError from opening.
    # Side effects: opens a file handle the caller must close.
    if isinstance(f, int):  # file descriptor
        return open(f, "rb", closefd=False)
    elif isinstance(f, os.PathLike):
        f = os.fspath(f)
    elif not isinstance(f, str):
        raise TypeError("expected {str, int, path-like, file-like}, got %s" % type(f))

    _, ext = os.path.splitext(f)
    if ext == ".gz":
        import gzip

        return gzip.open(f, "rb")
    elif ext == ".bz2":
        from bz2 import BZ2File

        return BZ2File(f, "rb")
    else:
        return open(f, "rb")


def _open_and_load(f, dtype, multilabel, zero_based, query_id, offset=0, length=-1):
    # [FUNCTION DOC]
    # HELPER: Parses one svmlight source into raw CSR components.
    # Steps: if f has read, parse it directly; otherwise open it with _gen_open (closed
    # afterwards) and parse via the Cython _load_svmlight_file. Convert the returned
    # array.array buffers to numpy arrays: labels to float64 (unless multilabel, then
    # left as list), data to the dtype reported by the parser then cast to dtype,
    # indices and indptr to int64, query ids to int64.
    # Inputs: f - source; dtype; multilabel - bool; zero_based - bool; query_id - bool;
    # offset, length - byte window.
    # Outputs: tuple (data, indices, indptr, labels, query).
    # Exceptions: ValueError from the parser on malformed input; TypeError from
    # _gen_open.
    # Side effects: reads from (and possibly opens/closes) the file.
    if hasattr(f, "read"):
        actual_dtype, data, ind, indptr, labels, query = _load_svmlight_file(
            f, dtype, multilabel, zero_based, query_id, offset, length
        )
    else:
        with closing(_gen_open(f)) as f:
            actual_dtype, data, ind, indptr, labels, query = _load_svmlight_file(
                f, dtype, multilabel, zero_based, query_id, offset, length
            )

    # convert from array.array, give data the right dtype
    if not multilabel:
        labels = np.frombuffer(labels, np.float64)
    data = np.frombuffer(data, actual_dtype)
    indices = np.frombuffer(ind, np.longlong)
    indptr = np.frombuffer(indptr, dtype=np.longlong)  # never empty
    query = np.frombuffer(query, np.int64)

    data = np.asarray(data, dtype=dtype)  # no-op for float{32,64}
    return data, indices, indptr, labels, query


@validate_params(
    {
        "files": [
            "array-like",
            str,
            os.PathLike,
            HasMethods("read"),
            Interval(Integral, 0, None, closed="left"),
        ],
        "n_features": [Interval(Integral, 1, None, closed="left"), None],
        "dtype": "no_validation",  # delegate validation to numpy
        "multilabel": ["boolean"],
        "zero_based": ["boolean", StrOptions({"auto"})],
        "query_id": ["boolean"],
        "offset": [Interval(Integral, 0, None, closed="left")],
        "length": [Integral],
    },
    prefer_skip_nested_validation=True,
)
def load_svmlight_files(
    files,
    *,
    n_features=None,
    dtype=np.float64,
    multilabel=False,
    zero_based="auto",
    query_id=False,
    offset=0,
    length=-1,
):
    # [FUNCTION DOC]
    # Loads several svmlight files and returns aligned CSR matrices with a common number
    # of columns.
    # Steps: when offset or length restrict the read, force zero_based to True if it was
    # "auto" and require n_features. Parse every file with _open_and_load. Shift all
    # indices down by one when zero_based is False, or when "auto" and every file is
    # non-empty with minimum index > 0 (one-based heuristic). Compute the needed number
    # of features as max index + 1 over all files; use it when n_features is None, else
    # check n_features is not smaller. For each file build a CSR array of shape (n_rows,
    # n_features), sort its indices and append X, y and optionally query ids to a flat
    # result list.
    # Inputs: files - iterable of sources (as load_svmlight_file's f); other parameters
    # as in load_svmlight_file.
    # Outputs: list [X1, y1, X2, y2, ...] (with q_i after each y_i when query_id is
    # True).
    # Exceptions: ValueError when n_features is missing with offset/length, or smaller
    # than the features found; parser errors.
    # Side effects: mutates the parsed index arrays in place when shifting.
    # KNOWN ISSUE: the validator accepts str, path-like, file-like or int for files, but
    # the code iterates over files, so a single path string would be iterated character
    # by character.
    """Load dataset from multiple files in SVMlight format.

    This function is equivalent to mapping load_svmlight_file over a list of
    files, except that the results are concatenated into a single, flat list
    and the samples vectors are constrained to all have the same number of
    features.

    In case the file contains a pairwise preference constraint (known
    as "qid" in the svmlight format) these are ignored unless the
    query_id parameter is set to True. These pairwise preference
    constraints can be used to constraint the combination of samples
    when using pairwise loss functions (as is the case in some
    learning to rank problems) so that only pairs with the same
    query_id value are considered.

    Parameters
    ----------
    files : array-like, dtype=str, path-like, file-like or int
        (Paths of) files to load. If a path ends in ".gz" or ".bz2", it will
        be uncompressed on the fly. If an integer is passed, it is assumed to
        be a file descriptor. File-likes and file descriptors will not be
        closed by this function. File-like objects must be opened in binary
        mode.

        .. versionchanged:: 1.2
           Path-like objects are now accepted.

    n_features : int, default=None
        The number of features to use. If None, it will be inferred from the
        maximum column index occurring in any of the files.

        This can be set to a higher value than the actual number of features
        in any of the input files, but setting it to a lower value will cause
        an exception to be raised.

    dtype : numpy data type, default=np.float64
        Data type of dataset to be loaded. This will be the data type of the
        output numpy arrays ``X`` and ``y``.

    multilabel : bool, default=False
        Samples may have several labels each (see
        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html).

    zero_based : bool or "auto", default="auto"
        Whether column indices in f are zero-based (True) or one-based
        (False). If column indices are one-based, they are transformed to
        zero-based to match Python/NumPy conventions.
        If set to "auto", a heuristic check is applied to determine this from
        the file contents. Both kinds of files occur "in the wild", but they
        are unfortunately not self-identifying. Using "auto" or True should
        always be safe when no offset or length is passed.
        If offset or length are passed, the "auto" mode falls back
        to zero_based=True to avoid having the heuristic check yield
        inconsistent results on different segments of the file.

    query_id : bool, default=False
        If True, will return the query_id array for each file.

    offset : int, default=0
        Ignore the offset first bytes by seeking forward, then
        discarding the following bytes up until the next new line
        character.

    length : int, default=-1
        If strictly positive, stop reading any new line of data once the
        position in the file has reached the (offset + length) bytes threshold.

    Returns
    -------
    [X1, y1, ..., Xn, yn] or [X1, y1, q1, ..., Xn, yn, qn]: list of arrays
        Each (Xi, yi) pair is the result from load_svmlight_file(files[i]).
        If query_id is set to True, this will return instead (Xi, yi, qi)
        triplets.

    See Also
    --------
    load_svmlight_file: Similar function for loading a single file in this
        format.

    Notes
    -----
    When fitting a model to a matrix X_train and evaluating it against a
    matrix X_test, it is essential that X_train and X_test have the same
    number of features (X_train.shape[1] == X_test.shape[1]). This may not
    be the case if you load the files individually with load_svmlight_file.

    Examples
    --------
    To use joblib.Memory to cache the svmlight file::

        from joblib import Memory
        from sklearn.datasets import load_svmlight_file
        mem = Memory("./mycache")

        @mem.cache
        def get_data():
            data_train, target_train, data_test, target_test = load_svmlight_files(
                ["svmlight_file_train", "svmlight_file_test"]
            )
            return data_train, target_train, data_test, target_test

        X_train, y_train, X_test, y_test = get_data()
    """
    if (offset != 0 or length > 0) and zero_based == "auto":
        # disable heuristic search to avoid getting inconsistent results on
        # different segments of the file
        zero_based = True

    if (offset != 0 or length > 0) and n_features is None:
        raise ValueError("n_features is required when offset or length is specified.")

    r = [
        _open_and_load(
            f,
            dtype,
            multilabel,
            bool(zero_based),
            bool(query_id),
            offset=offset,
            length=length,
        )
        for f in files
    ]

    if zero_based is False or (
        zero_based == "auto" and all(len(tmp[1]) and np.min(tmp[1]) > 0 for tmp in r)
    ):
        for _, indices, _, _, _ in r:
            indices -= 1

    n_f = max(ind[1].max() if len(ind[1]) else 0 for ind in r) + 1

    if n_features is None:
        n_features = n_f
    elif n_features < n_f:
        raise ValueError(
            "n_features was set to {}, but input file contains {} features".format(
                n_features, n_f
            )
        )

    result = []
    for data, indices, indptr, y, query_values in r:
        shape = (indptr.shape[0] - 1, n_features)
        X = sp.csr_array((data, indices, indptr), shape)
        X.sort_indices()
        result += _align_api_if_sparse(X), y
        if query_id:
            result.append(query_values)

    return result


def _dump_svmlight(X, y, f, multilabel, one_based, comment, query_id):
    # [FUNCTION DOC]
    # HELPER: Writes the optional header comment and delegates the body to the Cython
    # writer.
    # Steps: if comment is non-empty, write a generator line with the sklearn version, a
    # line stating zero/one-based indices, a blank "#" line, then each comment line
    # prefixed with "# ". Reshape a dense 1-D y to a column when not multilabel. Call
    # _dump_svmlight_file with sparsity flags.
    # Inputs: X - dense or CSR; y - dense or CSR; f - binary writable file; multilabel -
    # bool; one_based - bool; comment - bytes or None; query_id - int array (possibly
    # empty).
    # Outputs: None. Side effects: writes to f.
    if comment:
        f.write(
            (
                "# Generated by dump_svmlight_file from scikit-learn %s\n" % __version__
            ).encode()
        )
        f.write(
            ("# Column indices are %s-based\n" % ["zero", "one"][one_based]).encode()
        )

        f.write(b"#\n")
        f.writelines(b"# %s\n" % line for line in comment.splitlines())
    X_is_sp = sp.issparse(X)
    y_is_sp = sp.issparse(y)
    if not multilabel and not y_is_sp:
        y = y[:, np.newaxis]
    _dump_svmlight_file(
        X,
        y,
        f,
        multilabel,
        one_based,
        query_id,
        X_is_sp,
        y_is_sp,
    )


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "y": ["array-like", "sparse matrix"],
        "f": [str, HasMethods(["write"])],
        "zero_based": ["boolean"],
        "comment": [str, bytes, None],
        "query_id": ["array-like", None],
        "multilabel": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def dump_svmlight_file(
    X,
    y,
    f,
    *,
    zero_based=True,
    comment=None,
    query_id=None,
    multilabel=False,
):
    # [FUNCTION DOC]
    # Writes X, y (and optional query ids) to a file in svmlight format.
    # Steps: validate comment (bytes must be ASCII; str encoded as UTF-8; NUL byte
    # rejected). Validate y with check_array allowing CSR and 1-D; require single-column
    # sparse or 1-D dense unless multilabel. Validate X; check row counts match. Sort
    # CSR indices, copying when the validated object is the user's own array so user
    # input is not modified. Replace query_id None by an empty int32 array, else check
    # its length. Write via _dump_svmlight, opening f in binary mode if a path.
    # Inputs: X - array-like or sparse (n_samples, n_features); y - array-like or sparse
    # (n_samples,) or (n_samples, n_labels) for multilabel; f - path string or writable
    # binary file; zero_based - bool, default True; comment - str, bytes or None;
    # query_id - array-like (n_samples,) or None; multilabel - bool.
    # Outputs: None.
    # Exceptions: UnicodeDecodeError for non-ASCII bytes comment; ValueError for NUL
    # bytes, wrong y shape, mismatched row counts or query_id length.
    # Side effects: writes/creates the file; may sort indices in place on arrays created
    # by check_array (not on user arrays).
    """Dump the dataset in svmlight / libsvm file format.

    This format is a text-based format, with one sample per line. It does
    not store zero valued features hence is suitable for sparse dataset.

    The first element of each line can be used to store a target variable
    to predict.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vectors, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : {array-like, sparse matrix}, shape = (n_samples,) or (n_samples, n_labels)
        Target values. Class labels must be an
        integer or float, or array-like objects of integer or float for
        multilabel classifications.

    f : str or file-like in binary mode
        If string, specifies the path that will contain the data.
        If file-like, data will be written to f. f should be opened in binary
        mode.

    zero_based : bool, default=True
        Whether column indices should be written zero-based (True) or one-based
        (False).

    comment : str or bytes, default=None
        Comment to insert at the top of the file. This should be either a
        Unicode string, which will be encoded as UTF-8, or an ASCII byte
        string.
        If a comment is given, then it will be preceded by one that identifies
        the file as having been dumped by scikit-learn. Note that not all
        tools grok comments in SVMlight files.

    query_id : array-like of shape (n_samples,), default=None
        Array containing pairwise preference constraints (qid in svmlight
        format).

    multilabel : bool, default=False
        Samples may have several labels each (see
        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html).

        .. versionadded:: 0.17
           parameter `multilabel` to support multilabel datasets.

    Examples
    --------
    >>> from sklearn.datasets import dump_svmlight_file, make_classification
    >>> X, y = make_classification(random_state=0)
    >>> output_file = "my_dataset.svmlight"
    >>> dump_svmlight_file(X, y, output_file)  # doctest: +SKIP
    """
    if comment is not None:
        # Convert comment string to list of lines in UTF-8.
        # If a byte string is passed, then check whether it's ASCII;
        # if a user wants to get fancy, they'll have to decode themselves.
        if isinstance(comment, bytes):
            comment.decode("ascii")  # just for the exception
        else:
            comment = comment.encode("utf-8")
        if b"\0" in comment:
            raise ValueError("comment string contains NUL byte")

    yval = check_array(y, accept_sparse="csr", ensure_2d=False)
    if sp.issparse(yval):
        if yval.shape[1] != 1 and not multilabel:
            raise ValueError(
                "expected y of shape (n_samples, 1), got %r" % (yval.shape,)
            )
    else:
        if yval.ndim != 1 and not multilabel:
            raise ValueError("expected y of shape (n_samples,), got %r" % (yval.shape,))

    Xval = check_array(X, accept_sparse="csr")
    if Xval.shape[0] != yval.shape[0]:
        raise ValueError(
            "X.shape[0] and y.shape[0] should be the same, got %r and %r instead."
            % (Xval.shape[0], yval.shape[0])
        )

    # We had some issues with CSR matrices with unsorted indices (e.g. #1501),
    # so sort them here, but first make sure we don't modify the user's X.
    # TODO We can do this cheaper; sorted_indices copies the whole matrix.
    if yval is y and hasattr(yval, "sorted_indices"):
        y = yval.sorted_indices()
    else:
        y = yval
        if hasattr(y, "sort_indices"):
            y.sort_indices()

    if Xval is X and hasattr(Xval, "sorted_indices"):
        X = Xval.sorted_indices()
    else:
        X = Xval
        if hasattr(X, "sort_indices"):
            X.sort_indices()

    if query_id is None:
        # NOTE: query_id is passed to Cython functions using a fused type on query_id.
        # Yet as of Cython>=3.0, memory views can't be None otherwise the runtime
        # would not known which concrete implementation to dispatch the Python call to.
        # TODO: simplify interfaces and implementations in _svmlight_format_fast.pyx.
        query_id = np.array([], dtype=np.int32)
    else:
        query_id = np.asarray(query_id)
        if query_id.shape[0] != y.shape[0]:
            raise ValueError(
                "expected query_id of shape (n_samples,), got %r" % (query_id.shape,)
            )

    one_based = not zero_based

    if hasattr(f, "write"):
        _dump_svmlight(X, y, f, multilabel, one_based, comment, query_id)
    else:
        with open(f, "wb") as f:
            _dump_svmlight(X, y, f, multilabel, one_based, comment, query_id)
