# [MODULE DOC]
# Maintenance script that suggests new minimum dependency versions for an upcoming
# scikit-learn release. Usage: run from the repo root with an optional release date
# argument (YYYY-MM-DD, default "today").
# Import-time side effects: performs an HTTP GET to devguide.python.org, parses its
# HTML tables with pandas and builds the global python_version_info mapping
# (Python branch string -> first release Timestamp). release_dates is an unused
# global.
# Policy encoded: min Python = newest branch released more than 3 years before the
# release date; compiled deps = oldest X.Y.0 release with a CPython wheel for that
# Python; pure-Python/example deps = newest X.Y.0 release whose wheel was uploaded
# more than 2 years before the release date.
# Output: prints lines "- name: current -> proposed" for every dependency whose
# proposed version differs from the current one.
# Exceptions: network/HTTP parsing errors propagate (requests, pandas).
import io
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from packaging import version

req = requests.get("https://devguide.python.org/versions/")
df_list = pd.read_html(io.StringIO(req.content.decode("utf-8")))
df = pd.concat(df_list).astype({"Branch": str})
release_dates = {}
python_version_info = {
    version: release_date
    for version, release_date in zip(df["Branch"], df["First release"])
}
python_version_info = {
    version: pd.to_datetime(release_date)
    for version, release_date in python_version_info.items()
}


def get_min_version_with_wheel(package_name, python_version):
    # [FUNCTION DOC]
    # Queries the PyPI JSON API for package_name and returns the oldest minor release
    # (version string matching X.Y.0) that has a non-yanked bdist_wheel file whose name
    # contains the cpXY tag of python_version.
    # Inputs: package_name - PyPI name (str); python_version - "X.Y" string.
    # Outputs: version string, or None if the HTTP status is not 200 or no release matches.
    # Side effects: one network request.
    # KNOWN ISSUE: the substring test for "cpXY" can match unintended tags (e.g. "cp31"
    # inside "cp310"), so a wheel for a different Python could be accepted.
    # For compiled dependencies we want the oldest minor version that has
    # wheels for 'python_version'
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    if response.status_code != 200:
        return None

    data = response.json()
    releases = data["releases"]

    compatible_versions = []
    # We want only minor X.Y.0 and not bugfix X.Y.Z
    minor_releases = [
        (ver, release_info)
        for ver, release_info in releases.items()
        if re.match(r"^\d+\.\d+\.0$", ver)
    ]
    for ver, release_info in minor_releases:
        for file_info in release_info:
            if (
                file_info["packagetype"] == "bdist_wheel"
                and f"cp{python_version.replace('.', '')}" in file_info["filename"]
                and not file_info["yanked"]
            ):
                compatible_versions.append(ver)
                break

    if not compatible_versions:
        return None

    return min(compatible_versions, key=version.parse)


def get_min_python_version(scikit_learn_release_date_str="today"):
    # [FUNCTION DOC]
    # Determines the minimum supported Python version: among Python branches in the
    # global python_version_info whose first release is more than 3*365 days before the
    # release date, returns the one with the latest release date.
    # Inputs: scikit_learn_release_date_str - "today" (default; uses the current date)
    # or a "%Y-%m-%d" string.
    # Outputs: Python branch string such as "3.10".
    # Exceptions: ValueError from strptime on malformed dates; ValueError from max if
    # no branch qualifies.
    # min Python version is the most recent Python release at least 3 years old
    # at the time of the scikit-learn release
    if scikit_learn_release_date_str == "today":
        scikit_learn_release_date = pd.to_datetime(datetime.now().date())
    else:
        scikit_learn_release_date = datetime.strptime(
            scikit_learn_release_date_str, "%Y-%m-%d"
        )
    version_and_releases = [
        {"python_version": python_version, "python_release_date": python_release_date}
        for python_version, python_release_date in python_version_info.items()
        if (scikit_learn_release_date - python_release_date).days > 365 * 3
    ]
    return max(version_and_releases, key=lambda each: each["python_release_date"])[
        "python_version"
    ]


def get_min_version_pure_python_or_example_dependency(
    package_name, scikit_learn_release_date_str="today"
):
    # [FUNCTION DOC]
    # For a pure-Python (or example-only) dependency, fetches PyPI JSON and returns the
    # newest X.Y.0 release that has a non-yanked bdist_wheel uploaded more than
    # 2*365 days before the release date.
    # Inputs: package_name - PyPI name; scikit_learn_release_date_str - "today" or
    # "%Y-%m-%d" string.
    # Outputs: version string, or None on non-200 response or when nothing qualifies.
    # Side effects: one network request.
    # KNOWN ISSUE: packages that only publish sdists (no wheels) never qualify and
    # return None.
    # for pure Python dependencies we want the most recent minor release that
    # is at least 2 years old
    if scikit_learn_release_date_str == "today":
        scikit_learn_release_date = pd.to_datetime(datetime.now().date())
    else:
        scikit_learn_release_date = datetime.strptime(
            scikit_learn_release_date_str, "%Y-%m-%d"
        )

    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    if response.status_code != 200:
        return None

    data = response.json()
    releases = data["releases"]

    compatible_versions = []
    # We want only minor X.Y.0 and not bugfix X.Y.Z
    releases = [
        (ver, release_info)
        for ver, release_info in releases.items()
        if re.match(r"^\d+\.\d+\.0$", ver)
    ]
    for ver, release_info in releases:
        for file_info in release_info:
            if (
                file_info["packagetype"] == "bdist_wheel"
                and not file_info["yanked"]
                and (
                    scikit_learn_release_date - pd.to_datetime(file_info["upload_time"])
                ).days
                > 365 * 2
            ):
                compatible_versions.append(ver)
                break

    if not compatible_versions:
        return None

    return max(compatible_versions, key=version.parse)


def get_current_dependencies_version(dep):
    # [FUNCTION DOC]
    # Returns the current minimum version of dependency dep by running
    # sklearn/_min_dependencies.py as a subprocess with dep as argument and decoding
    # its stripped stdout.
    # Inputs: dep - dependency name. Outputs: version string.
    # Exceptions: CalledProcessError if the subprocess fails; requires cwd = repo root.
    return (
        subprocess.check_output([sys.executable, "sklearn/_min_dependencies.py", dep])
        .decode()
        .strip()
    )


def get_current_min_python_version():
    # [FUNCTION DOC]
    # Reads pyproject.toml in the current directory and extracts the X.Y version from
    # the requires-python ">=X.Y" entry.
    # Outputs: version string. Exceptions: IndexError if the pattern is absent,
    # FileNotFoundError if the file is missing.
    content = Path("pyproject.toml").read_text()
    min_python = re.findall(r'requires-python\s*=\s*">=(\d+\.\d+)"', content)[0]

    return min_python


def show_versions_update(scikit_learn_release_date="today"):
    # [FUNCTION DOC]
    # Builds proposed versions (Python, compiled deps via get_min_version_with_wheel,
    # pure-Python/example deps via get_min_version_pure_python_or_example_dependency)
    # and current versions (pyproject.toml and _min_dependencies.py), then prints a
    # header and one line per key whose proposed and current versions differ.
    # Inputs: scikit_learn_release_date - "today" or "%Y-%m-%d" string.
    # Outputs: None. Side effects: network requests, subprocesses, stdout printing.
    # KNOWN ISSUE: "polars" appears in both dependency lists, so its compiled-style
    # result is overwritten by the pure-Python result and it is queried twice.
    future_versions = {"python": get_min_python_version(scikit_learn_release_date)}

    compiled_dependencies = [
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "pyamg",
        "polars",
        "pyarrow",
    ]
    future_versions.update(
        {
            dep: get_min_version_with_wheel(dep, future_versions["python"])
            for dep in compiled_dependencies
        }
    )

    pure_python_or_example_dependencies = [
        "joblib",
        "narwhals",
        "threadpoolctl",
        "scikit-image",
        "seaborn",
        "polars",
        "Pillow",
        "pooch",
        "plotly",
    ]
    future_versions.update(
        {
            dep: get_min_version_pure_python_or_example_dependency(
                dep, scikit_learn_release_date
            )
            for dep in pure_python_or_example_dependencies
        }
    )

    current_versions = {"python": get_current_min_python_version()}
    current_versions.update(
        {
            dep: get_current_dependencies_version(dep)
            for dep in compiled_dependencies + pure_python_or_example_dependencies
        }
    )

    print(f"For future release at date {scikit_learn_release_date}")
    for k in future_versions:
        if future_versions[k] != current_versions[k]:
            print(f"- {k}: {current_versions[k]} -> {future_versions[k]}")


if __name__ == "__main__":
    scikit_learn_release_date = sys.argv[1] if len(sys.argv) > 1 else "today"
    show_versions_update(scikit_learn_release_date)
