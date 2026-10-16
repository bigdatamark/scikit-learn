# [MODULE DOC]
# Loader for the KDD Cup 1999 network-intrusion dataset, used mainly for anomaly
# detection benchmarks. Defines remote metadata for the full and 10 percent archives,
# the public fetch_kddcup99 (with the classical SA/SF/http/smtp subsets), a private raw
# loader/cacher and a mkdir -p helper. Cache dirs: <data_home>/kddcup99-py3 or
# kddcup99_10-py3.
# Side effects at import: creates a module-level logger only.
"""KDDCUP 99 dataset.

A classic dataset for anomaly detection.

The dataset page is available from UCI Machine Learning Repository

https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data.gz

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import errno
import logging
import os
from gzip import GzipFile
from numbers import Integral, Real
from os.path import exists, join

import joblib
import numpy as np

from sklearn.datasets import get_data_home
from sklearn.datasets._base import (
    RemoteFileMetadata,
    _convert_data_dataframe,
    _fetch_remote,
    load_descr,
)
from sklearn.utils import Bunch, check_random_state
from sklearn.utils import shuffle as shuffle_method
from sklearn.utils._param_validation import Interval, StrOptions, validate_params

# The original data can be found at:
# https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data.gz
ARCHIVE = RemoteFileMetadata(
    filename="kddcup99_data",
    url="https://ndownloader.figshare.com/files/5976045",
    checksum="3b6c942aa0356c0ca35b7b595a26c89d343652c9db428893e7494f837b274292",
)

# The original data can be found at:
# https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data_10_percent.gz
ARCHIVE_10_PERCENT = RemoteFileMetadata(
    filename="kddcup99_10_data",
    url="https://ndownloader.figshare.com/files/5976042",
    checksum="8045aca0d84e70e622d1148d7df782496f6333bf6eb979a1b0837c42a9fd9561",
)

logger = logging.getLogger(__name__)


@validate_params(
    {
        "subset": [StrOptions({"SA", "SF", "http", "smtp"}), None],
        "data_home": [str, os.PathLike, None],
        "shuffle": ["boolean"],
        "random_state": ["random_state"],
        "percent10": ["boolean"],
        "download_if_missing": ["boolean"],
        "return_X_y": ["boolean"],
        "as_frame": ["boolean"],
        "n_retries": [Interval(Integral, 1, None, closed="left")],
        "delay": [Interval(Real, 0.0, None, closed="neither")],
    },
    prefer_skip_nested_validation=True,
)
def fetch_kddcup99(
    *,
    subset=None,
    data_home=None,
    shuffle=False,
    random_state=None,
    percent10=True,
    download_if_missing=True,
    return_X_y=False,
    as_frame=False,
    n_retries=3,
    delay=1.0,
):
    # [FUNCTION DOC]
    # Returns the KDD Cup 99 data, optionally reduced to a classical subset.
    # Steps: load raw object arrays via _fetch_brute_kddcup99. Subset SA: keep all rows
    # labelled b"normal." and append 3377 abnormal rows drawn uniformly WITH replacement
    # (randint) using random_state. Subsets SF/http/smtp: keep rows where logged_in
    # (column 11) equals 1, drop that column, and replace duration, src_bytes and
    # dst_bytes by log(x + 0.1); then http/smtp keep rows whose service equals
    # b"http"/b"smtp" and only the 3 log columns; SF keeps those 3 plus service (4
    # columns). Optionally shuffle data and target jointly; load DESCR; optionally
    # convert to pandas.
    # Inputs: subset - one of SA, SF, http, smtp or None; data_home; shuffle - bool;
    # random_state - used for SA sampling and shuffling; percent10 - bool, use 10
    # percent archive; download_if_missing; return_X_y; as_frame; n_retries; delay.
    # Outputs: (data, target) or Bunch(data, target, frame, target_names, feature_names,
    # DESCR). data is an object ndarray (bytes for categorical columns, numbers
    # otherwise); target holds byte-string labels.
    # Exceptions: OSError via the raw loader when data is missing or cache corrupt;
    # validation errors from validate_params.
    # Side effects: may download and write cache files.
    # KNOWN ISSUE: the docstring calls target the "regression target"; it is a
    # categorical label array.
    # KNOWN ISSUE: shape values in the docstring (494021) are only valid for
    # percent10=True and subset=None.
    """Load the kddcup99 dataset (classification).

    Download it if necessary.

    =================   ====================================
    Classes                                               23
    Samples total                                    4898431
    Dimensionality                                        41
    Features            discrete (int) or continuous (float)
    =================   ====================================

    Read more in the :ref:`User Guide <kddcup99_dataset>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    subset : {'SA', 'SF', 'http', 'smtp'}, default=None
        To return the corresponding classical subsets of kddcup 99.
        If None, return the entire kddcup 99 dataset.

    data_home : str or path-like, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

        .. versionadded:: 0.19

    shuffle : bool, default=False
        Whether to shuffle dataset.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset shuffling and for
        selection of abnormal samples if `subset='SA'`. Pass an int for
        reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    percent10 : bool, default=True
        Whether to load only 10 percent of the data.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` objects.

        .. versionadded:: 0.20

    as_frame : bool, default=False
        If `True`, returns a pandas Dataframe for the ``data`` and ``target``
        objects in the `Bunch` returned object; `Bunch` return object will also
        have a ``frame`` member.

        .. versionadded:: 0.24

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.

        .. versionadded:: 1.5

    delay : float, default=1.0
        Number of seconds between retries.

        .. versionadded:: 1.5

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (494021, 41)
            The data matrix to learn. If `as_frame=True`, `data` will be a
            pandas DataFrame.
        target : {ndarray, series} of shape (494021,)
            The regression target for each sample. If `as_frame=True`, `target`
            will be a pandas Series.
        frame : dataframe of shape (494021, 42)
            Only present when `as_frame=True`. Contains `data` and `target`.
        DESCR : str
            The full description of the dataset.
        feature_names : list
            The names of the dataset columns
        target_names: list
            The names of the target columns

    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarray. The first containing a 2D array of
        shape (n_samples, n_features) with each row representing one
        sample and each column representing the features. The second
        ndarray of shape (n_samples,) containing the target samples.

        .. versionadded:: 0.20
    """
    data_home = get_data_home(data_home=data_home)
    kddcup99 = _fetch_brute_kddcup99(
        data_home=data_home,
        percent10=percent10,
        download_if_missing=download_if_missing,
        n_retries=n_retries,
        delay=delay,
    )

    data = kddcup99.data
    target = kddcup99.target
    feature_names = kddcup99.feature_names
    target_names = kddcup99.target_names

    if subset == "SA":
        s = target == b"normal."
        t = np.logical_not(s)
        normal_samples = data[s, :]
        normal_targets = target[s]
        abnormal_samples = data[t, :]
        abnormal_targets = target[t]

        n_samples_abnormal = abnormal_samples.shape[0]
        # selected abnormal samples:
        random_state = check_random_state(random_state)
        r = random_state.randint(0, n_samples_abnormal, 3377)
        abnormal_samples = abnormal_samples[r]
        abnormal_targets = abnormal_targets[r]

        data = np.r_[normal_samples, abnormal_samples]
        target = np.r_[normal_targets, abnormal_targets]

    if subset == "SF" or subset == "http" or subset == "smtp":
        # select all samples with positive logged_in attribute:
        s = data[:, 11] == 1
        data = np.c_[data[s, :11], data[s, 12:]]
        feature_names = feature_names[:11] + feature_names[12:]
        target = target[s]

        data[:, 0] = np.log((data[:, 0] + 0.1).astype(float, copy=False))
        data[:, 4] = np.log((data[:, 4] + 0.1).astype(float, copy=False))
        data[:, 5] = np.log((data[:, 5] + 0.1).astype(float, copy=False))

        if subset == "http":
            s = data[:, 2] == b"http"
            data = data[s]
            target = target[s]
            data = np.c_[data[:, 0], data[:, 4], data[:, 5]]
            feature_names = [feature_names[0], feature_names[4], feature_names[5]]

        if subset == "smtp":
            s = data[:, 2] == b"smtp"
            data = data[s]
            target = target[s]
            data = np.c_[data[:, 0], data[:, 4], data[:, 5]]
            feature_names = [feature_names[0], feature_names[4], feature_names[5]]

        if subset == "SF":
            data = np.c_[data[:, 0], data[:, 2], data[:, 4], data[:, 5]]
            feature_names = [
                feature_names[0],
                feature_names[2],
                feature_names[4],
                feature_names[5],
            ]

    if shuffle:
        data, target = shuffle_method(data, target, random_state=random_state)

    fdescr = load_descr("kddcup99.rst")

    frame = None
    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "fetch_kddcup99", data, target, feature_names, target_names
        )

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        feature_names=feature_names,
        DESCR=fdescr,
    )


def _fetch_brute_kddcup99(
    data_home=None, download_if_missing=True, percent10=True, n_retries=3, delay=1.0
):
    # [FUNCTION DOC]
    # HELPER: Loads (or downloads and caches) the raw KDD Cup 99 arrays without
    # subsetting.
    # Steps: choose cache directory and archive by percent10; define the 42-column
    # structured dtype (ints, floats, fixed-size byte strings). If the samples pickle
    # exists, joblib-load samples and targets. Else if download_if_missing: create dir,
    # download archive, read the gzip line by line, split each line on commas, build an
    # object array, cast each column to its dtype, split off last column as y, delete
    # the archive and dump X and y with joblib (compress=3). Else raise.
    # Inputs: data_home - cache root or None; download_if_missing - bool; percent10 -
    # bool; n_retries - int; delay - float.
    # Outputs: Bunch with data (object ndarray n x 41), target (object ndarray of
    # bytes), feature_names (41 names), target_names (single-element list ["labels"]).
    # Exceptions: OSError if cache loading fails (chained) or data missing with
    # downloads disabled.
    # Side effects: network download, file writes, removal of the downloaded archive,
    # logging.
    # KNOWN ISSUE: the docstring lists a DESCR attribute in the returned Bunch, but none
    # is set.
    # KNOWN ISSUE: availability is judged only by the samples file; a missing targets
    # file surfaces as the generic invalid-cache OSError.
    """Load the kddcup99 dataset, downloading it if necessary.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    percent10 : bool, default=True
        Whether to load only 10 percent of the data.

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.

    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray of shape (494021, 41)
            Each row corresponds to the 41 features in the dataset.
        target : ndarray of shape (494021,)
            Each value corresponds to one of the 21 attack types or to the
            label 'normal.'.
        feature_names : list
            The names of the dataset columns
        target_names: list
            The names of the target columns
        DESCR : str
            Description of the kddcup99 dataset.

    """

    data_home = get_data_home(data_home=data_home)
    dir_suffix = "-py3"

    if percent10:
        kddcup_dir = join(data_home, "kddcup99_10" + dir_suffix)
        archive = ARCHIVE_10_PERCENT
    else:
        kddcup_dir = join(data_home, "kddcup99" + dir_suffix)
        archive = ARCHIVE

    samples_path = join(kddcup_dir, "samples")
    targets_path = join(kddcup_dir, "targets")
    available = exists(samples_path)

    dt = [
        ("duration", int),
        ("protocol_type", "S4"),
        ("service", "S11"),
        ("flag", "S6"),
        ("src_bytes", int),
        ("dst_bytes", int),
        ("land", int),
        ("wrong_fragment", int),
        ("urgent", int),
        ("hot", int),
        ("num_failed_logins", int),
        ("logged_in", int),
        ("num_compromised", int),
        ("root_shell", int),
        ("su_attempted", int),
        ("num_root", int),
        ("num_file_creations", int),
        ("num_shells", int),
        ("num_access_files", int),
        ("num_outbound_cmds", int),
        ("is_host_login", int),
        ("is_guest_login", int),
        ("count", int),
        ("srv_count", int),
        ("serror_rate", float),
        ("srv_serror_rate", float),
        ("rerror_rate", float),
        ("srv_rerror_rate", float),
        ("same_srv_rate", float),
        ("diff_srv_rate", float),
        ("srv_diff_host_rate", float),
        ("dst_host_count", int),
        ("dst_host_srv_count", int),
        ("dst_host_same_srv_rate", float),
        ("dst_host_diff_srv_rate", float),
        ("dst_host_same_src_port_rate", float),
        ("dst_host_srv_diff_host_rate", float),
        ("dst_host_serror_rate", float),
        ("dst_host_srv_serror_rate", float),
        ("dst_host_rerror_rate", float),
        ("dst_host_srv_rerror_rate", float),
        ("labels", "S16"),
    ]

    column_names = [c[0] for c in dt]
    target_names = column_names[-1]
    feature_names = column_names[:-1]

    if available:
        try:
            X = joblib.load(samples_path)
            y = joblib.load(targets_path)
        except Exception as e:
            raise OSError(
                "The cache for fetch_kddcup99 is invalid, please delete "
                f"{kddcup_dir} and run the fetch_kddcup99 again"
            ) from e

    elif download_if_missing:
        _mkdirp(kddcup_dir)
        logger.info("Downloading %s" % archive.url)
        _fetch_remote(archive, dirname=kddcup_dir, n_retries=n_retries, delay=delay)
        DT = np.dtype(dt)
        logger.debug("extracting archive")
        archive_path = join(kddcup_dir, archive.filename)
        Xy = []

        with GzipFile(filename=archive_path, mode="r") as file_:
            for line in file_.readlines():
                line = line.decode()
                Xy.append(line.replace("\n", "").split(","))

        logger.debug("extraction done")
        os.remove(archive_path)

        Xy = np.asarray(Xy, dtype=object)
        for j in range(42):
            Xy[:, j] = Xy[:, j].astype(DT[j])

        X = Xy[:, :-1]
        y = Xy[:, -1]
        joblib.dump(X, samples_path, compress=3)
        joblib.dump(y, targets_path, compress=3)
    else:
        raise OSError("Data not found and `download_if_missing` is False")

    return Bunch(
        data=X,
        target=y,
        feature_names=feature_names,
        target_names=[target_names],
    )


def _mkdirp(d):
    # [FUNCTION DOC]
    # HELPER: Creates directory d and any parents, ignoring the error if it already
    # exists.
    # Inputs: d - path. Outputs: None.
    # Exceptions: re-raises any OSError other than EEXIST.
    # Side effects: creates directories on disk.
    """Ensure directory d exists (like mkdir -p on Unix)
    No guarantee that the directory is writable.
    """
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
