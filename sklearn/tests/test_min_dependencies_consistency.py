# [MODULE DOC]
# Tests that the minimum dependency versions declared in sklearn/_min_dependencies.py
# agree with README.rst and with pyproject.toml. Defines module-level toy pyproject
# TOML snippets and a toy dependency table (joblib install, scipy build+install) used
# to exercise the checker on matching and deliberately broken inputs.
"""Tests for the minimum dependencies in README.rst and pyproject.toml"""

import os
import re
import tomllib
from collections import defaultdict
from pathlib import Path

import pytest

import sklearn
from sklearn._min_dependencies import dependent_packages
from sklearn.utils.fixes import parse_version

# minimal dependencies and pyproject definitions for testing the pyproject tests

TOY_MIN_DEPENDENCIES_PY_INFO = {
    "joblib": ("1.3.0", "install"),
    "scipy": ("1.10.0", "build, install"),
}

TOY_MATCHING_PYPROJECT_SECTIONS = """
[project]
dependencies = ["joblib>=1.3.0", "scipy>=1.10.0"]
[build-system]
requires = ["scipy>=1.10.0"]
"""

TOY_MATCHING_PYPROJECT_SECTIONS_WITH_UPPER_BOUND = """
[project]
dependencies = ["joblib>=1.3.0,<2.0", "scipy>=1.10.0"]
[build-system]
requires = ["scipy>=1.10.0,<1.19.0"]
"""

TOY_WRONG_SYMBOL_PYPROJECT_SECTIONS = """
[project]
dependencies = ["scipy<1.10.0"]
[build-system]
requires = ["scipy>=1.10.0"]
"""

TOY_MISSING_PACKAGE_PYPROJECT_SECTIONS = """
[project]
dependencies = ["scipy>=1.10.0"]
[build-system]
requires = ["scipy>=1.10.0"]
"""

TOY_ADDITIONAL_PACKAGE_PYPROJECT_SECTIONS = """
[project]
dependencies = ["joblib>=1.3.0", "scipy>=1.10.0"]
[build-system]
requires = ["scipy>=1.10.0", "package_not_in_min_dependencies_py_file>=4.2"]
"""

TOY_NON_MATCHING_VERSION_PYPROJECT_SECTIONS = """
[project]
dependencies = ["joblib>=1.42.0", "scipy>=1.10.0"]
[build-system]
requires = ["scipy>=1.10.0"]
"""


def test_min_dependencies_readme():
    # [FUNCTION DOC]
    # Parses README.rst (located next to the installed sklearn package directory) for
    # lines of the form ".. |<Name>MinVersion| replace:: X.Y[.Z]"; for each lower-cased
    # package also listed in dependent_packages, asserts the parsed version equals the
    # declared minimum version. Skips when README.rst is absent (e.g. wheel install).
    # Test that the minimum dependencies in the README.rst file are
    # consistent with the minimum dependencies defined at the file:
    # sklearn/_min_dependencies.py

    pattern = re.compile(
        r"\.\. \|"
        r"([A-Za-z-]+)"
        r"MinVersion\| replace::"
        r"( [0-9]+\.[0-9]+(\.[0-9]+)?)"
    )

    readme_path = Path(sklearn.__file__).parent.parent
    readme_file = readme_path / "README.rst"

    if not os.path.exists(readme_file):
        # Skip the test if the README.rst file is not available.
        # For instance, when installing scikit-learn from wheels
        pytest.skip("The README.rst file is not available.")

    with readme_file.open("r") as f:
        for line in f:
            matched = pattern.match(line)

            if not matched:
                continue

            package, version = matched.group(1), matched.group(2)
            package = package.lower()

            if package in dependent_packages:
                version = parse_version(version)
                min_version = parse_version(dependent_packages[package][0])

                message = (
                    f"{package} has inconsistent minimum versions in README.rst and"
                    f" _min_dependencies.py: {version} != {min_version}"
                )
                assert version == min_version, message


def extract_packages_and_pyproject_tags(dependencies):
    # [FUNCTION DOC]
    # HELPER: Groups package names by their min-dependencies tag and returns the mapping
    # from pyproject section path to tag.
    # Inputs: dependencies - dict package -> (min_version, comma-separated tag string).
    # Outputs: tuple (defaultdict tag -> list of packages, restricted to tags "build" and
    # "install"; dict {"build-system.requires": "build", "project.dependencies":
    # "install"}).
    # Side effects: none.
    pyproject_section_to_min_dependencies_tag = {
        "build-system.requires": "build",
        "project.dependencies": "install",
    }

    min_dependencies_tag_to_packages_without_version = defaultdict(list)
    for package, (min_version, tags) in dependencies.items():
        for t in tags.split(", "):
            if t in pyproject_section_to_min_dependencies_tag.values():
                min_dependencies_tag_to_packages_without_version[t].append(package)

    return (
        min_dependencies_tag_to_packages_without_version,
        pyproject_section_to_min_dependencies_tag,
    )


def check_pyproject_sections(pyproject_toml, min_dependencies):
    # [FUNCTION DOC]
    # HELPER: Checks one parsed pyproject against a min-dependencies table. For each of
    # the two sections, walks the dotted key path into the TOML dict, regex-parses each
    # requirement string into name and version (only ">=" or "==" accepted), asserts the
    # set of package names equals the packages tagged for that section, then asserts each
    # version equals the declared minimum (version check skipped for numpy in
    # build-system.requires).
    # Inputs: pyproject_toml - dict from tomllib; min_dependencies - dict package ->
    # (version, tags).
    # Outputs: None.
    # Exceptions: NotImplementedError when a requirement does not match the pattern;
    # AssertionError on package set or version mismatch; KeyError if a section is
    # missing.
    # KNOWN ISSUE: the regex only reads the first ">=" or "==" constraint; upper
    # bounds after a comma are silently ignored rather than validated.
    packages, pyproject_tags = extract_packages_and_pyproject_tags(min_dependencies)

    for pyproject_section, min_dependencies_tag in pyproject_tags.items():
        # Special situation for numpy: we have numpy>=2 in
        # build-system.requires to make sure we build wheels against numpy>=2.
        # TODO remove this when our minimum supported numpy version is >=2.
        skip_version_check_for = (
            ["numpy"] if pyproject_section == "build-system.requires" else []
        )

        expected_packages = packages[min_dependencies_tag]

        pyproject_section_keys = pyproject_section.split(".")
        info = pyproject_toml
        # iterate through nested keys to get packages and version
        for key in pyproject_section_keys:
            info = info[key]

        pyproject_build_min_versions = {}
        # Assuming pyproject.toml build section has something like "my-package>=2.3.0"
        pattern = r"([\w-]+)\s*[>=]=\s*([\d\w.]+)"
        for requirement in info:
            match = re.search(pattern, requirement)
            if match is None:
                raise NotImplementedError(
                    f"{requirement} does not match expected regex {pattern!r}. "
                    "Only >= and == are supported for version requirements"
                )

            package, version = match.group(1), match.group(2)

            pyproject_build_min_versions[package] = version

        msg = f"Packages in {pyproject_section} differ from _min_dependencies.py"

        assert sorted(pyproject_build_min_versions) == sorted(expected_packages), msg

        for package, version in pyproject_build_min_versions.items():
            version = parse_version(version)
            expected_min_version = parse_version(min_dependencies[package][0])
            if package in skip_version_check_for:
                continue

            message = (
                f"{package} has inconsistent minimum versions in pyproject.toml and"
                f" _min_dependencies.py: {version} != {expected_min_version}"
            )
            assert version == expected_min_version, message


def test_min_dependencies_pyproject_toml():
    # [FUNCTION DOC]
    # Loads the real pyproject.toml next to the sklearn package directory and runs
    # check_pyproject_sections against dependent_packages. Skips when the file is absent.
    """Check versions in pyproject.toml is consistent with _min_dependencies."""

    root_directory = Path(sklearn.__file__).parent.parent
    pyproject_toml_path = root_directory / "pyproject.toml"

    if not pyproject_toml_path.exists():
        # Skip the test if the pyproject.toml file is not available.
        # For instance, when installing scikit-learn from wheels
        pytest.skip("pyproject.toml is not available.")

    with pyproject_toml_path.open("rb") as f:
        pyproject_toml = tomllib.load(f)

    check_pyproject_sections(pyproject_toml, dependent_packages)


@pytest.mark.parametrize(
    "example_pyproject",
    [
        TOY_MATCHING_PYPROJECT_SECTIONS,
        TOY_MATCHING_PYPROJECT_SECTIONS_WITH_UPPER_BOUND,
    ],
)
def test_check_matching_pyproject_section(example_pyproject):
    # [FUNCTION DOC]
    # Verifies that toy pyprojects with matching versions (with and without upper bounds)
    # pass check_pyproject_sections without error.
    # Inputs: example_pyproject - parametrized TOML text.
    """Test the version check for matching packages."""

    pyproject_toml = tomllib.loads(example_pyproject)

    check_pyproject_sections(pyproject_toml, TOY_MIN_DEPENDENCIES_PY_INFO)


@pytest.mark.parametrize(
    "example_non_matching_pyproject, error_msg",
    [
        (
            TOY_WRONG_SYMBOL_PYPROJECT_SECTIONS,
            ".* does not match expected regex .*. "
            "Only >= and == are supported for version requirements",
        ),
        (
            TOY_MISSING_PACKAGE_PYPROJECT_SECTIONS,
            "Packages in .* differ from _min_dependencies.py",
        ),
        (
            TOY_ADDITIONAL_PACKAGE_PYPROJECT_SECTIONS,
            "Packages in .* differ from _min_dependencies.py",
        ),
        (
            TOY_NON_MATCHING_VERSION_PYPROJECT_SECTIONS,
            ".* has inconsistent minimum versions in pyproject.toml and"
            " _min_dependencies.py: .* != .*",
        ),
    ],
)
def test_check_non_matching_pyproject_section(
    example_non_matching_pyproject, error_msg
):
    # [FUNCTION DOC]
    # Verifies that each broken toy pyproject (wrong operator, missing package, extra
    # package, wrong version) makes check_pyproject_sections raise with a message
    # matching the expected regex.
    # Inputs: example_non_matching_pyproject - TOML text; error_msg - regex to match.
    """Test the version check for non-matching packages and versions."""

    pyproject_toml = tomllib.loads(example_non_matching_pyproject)

    with pytest.raises(Exception, match=error_msg):
        check_pyproject_sections(pyproject_toml, TOY_MIN_DEPENDENCIES_PY_INFO)
