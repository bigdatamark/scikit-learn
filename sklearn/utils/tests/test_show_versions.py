# [MODULE DOC]
# Tests for sklearn.utils._show_versions: system info, dependency info and the printed
# show_versions report.
from threadpoolctl import threadpool_info

from sklearn.utils._show_versions import _get_deps_info, _get_sys_info, show_versions
from sklearn.utils._testing import ignore_warnings


def test_get_sys_info():
    # [FUNCTION DOC]
    # Verifies that _get_sys_info returns a mapping containing 'python', 'executable'
    # and 'machine' keys.
    sys_info = _get_sys_info()

    assert "python" in sys_info
    assert "executable" in sys_info
    assert "machine" in sys_info


def test_get_deps_info():
    # [FUNCTION DOC]
    # Verifies that _get_deps_info (with warnings ignored) lists the expected
    # dependencies: pip, setuptools, sklearn, numpy, scipy, Cython, pandas, matplotlib,
    # joblib and narwhals.
    with ignore_warnings():
        deps_info = _get_deps_info()

    assert "pip" in deps_info
    assert "setuptools" in deps_info
    assert "sklearn" in deps_info
    assert "numpy" in deps_info
    assert "scipy" in deps_info
    assert "Cython" in deps_info
    assert "pandas" in deps_info
    assert "matplotlib" in deps_info
    assert "joblib" in deps_info
    assert "narwhals" in deps_info


def test_show_versions(capsys):
    # [FUNCTION DOC]
    # Verifies that show_versions prints to stdout mentioning python and numpy, and a
    # 'threadpoolctl info:' section whenever threadpool_info reports any thread pools.
    # Inputs: capsys - pytest fixture capturing output.
    with ignore_warnings():
        show_versions()
        out, err = capsys.readouterr()

    assert "python" in out
    assert "numpy" in out

    info = threadpool_info()
    if info:
        assert "threadpoolctl info:" in out
