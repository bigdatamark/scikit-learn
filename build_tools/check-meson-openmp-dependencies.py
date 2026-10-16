# [MODULE DOC]
# CI check script: compares the set of Cython extensions that use OpenMP according
# to source code (git grep for cython parallel or _openmp_helpers in .pyx files)
# with the set built with OpenMP flags according to meson introspection. Raises if
# they differ, listing extensions missing or wrongly having openmp_dep.
# Side effects when run: runs "meson setup build/introspect --reconfigure",
# "meson introspect" and "git grep" subprocesses.
"""
Check that OpenMP dependencies are correctly defined in meson.build files.

This is based on trying to make sure the following two things match:
- the Cython files using OpenMP (based on a git grep regex)
- the Cython extension modules that are built with OpenMP compiler flags (based
  on meson introspect json output)
"""

import json
import re
import subprocess
from pathlib import Path


def has_source_openmp_flags(target_source):
    # [FUNCTION DOC]
    # HELPER: True if any compiler/linker argument of a meson target source contains
    # "openmp". Input: dict with a "parameters" list of strings.
    return any("openmp" in arg for arg in target_source["parameters"])


def has_openmp_flags(target):
    # [FUNCTION DOC]
    # HELPER: Returns whether a meson target is built with OpenMP flags. If no source
    # has OpenMP flags returns False; otherwise asserts there are exactly two sources
    # (compiler then linker) and that both agree on OpenMP use.
    # Input: meson introspect target dict. Output: bool.
    # Exceptions: AssertionError on unexpected source layout or compiler/linker
    # mismatch.
    """Return whether target sources use OpenMP flags.

    Make sure that both compiler and linker source use OpenMP.
    Look at `get_meson_info` docstring to see what `target` looks like.
    """
    target_sources = target["target_sources"]

    target_use_openmp_flags = any(
        has_source_openmp_flags(target_source) for target_source in target_sources
    )

    if not target_use_openmp_flags:
        return False

    # When the target use OpenMP we expect a compiler + linker source and we
    # want to make sure that both the compiler and the linker use OpenMP
    assert len(target_sources) == 2
    compiler_source, linker_source = target_sources
    assert "compiler" in compiler_source
    assert "linker" in linker_source

    compiler_use_openmp_flags = any(
        "openmp" in arg for arg in compiler_source["parameters"]
    )
    linker_use_openmp_flags = any(
        "openmp" in arg for arg in linker_source["parameters"]
    )

    assert compiler_use_openmp_flags == linker_use_openmp_flags
    return compiler_use_openmp_flags


def get_canonical_name_meson(target, build_path):
    # [FUNCTION DOC]
    # HELPER: Converts a target's shared library path into a path relative to the build
    # directory with the Python ABI suffix (".cpython-..." or ".cp312-...") stripped,
    # using forward slashes. Inputs: target dict; build_path Path.
    # Output: string such as "sklearn/cluster/_k_means_elkan".
    # Exceptions: AssertionError unless exactly one filename; ValueError if not under
    # build_path.
    """Return a name based on generated shared library.

    The goal is to return a name that can be easily matched with the output
    from `git_grep_info`.

    Look at `get_meson_info` docstring to see what `target` looks like.
    """
    # Expect a list with one element with the name of the shared library
    assert len(target["filename"]) == 1
    shared_library_path = Path(target["filename"][0])
    shared_library_relative_path = shared_library_path.relative_to(
        build_path.absolute()
    )
    # Needed on Windows to match git grep output
    rel_path = shared_library_relative_path.as_posix()
    # OS-specific naming of the shared library .cpython- on POSIX and
    # something like .cp312- on Windows
    pattern = r"\.(cpython|cp\d+)-.+"
    return re.sub(pattern, "", str(rel_path))


def get_canonical_name_git_grep(filename):
    # [FUNCTION DOC]
    # HELPER: Strips ".pyx" or ".pyx.tp" from a repository file path to match the
    # meson canonical name. Output: string.
    """Return name based on filename.

    The goal is to return a name that can easily be matched with the output
    from `get_meson_info`.
    """
    return re.sub(r"\.pyx(\.tp)?", "", filename)


def get_meson_info():
    # [FUNCTION DOC]
    # Configures meson in build/introspect, reads the targets JSON, keeps those with
    # OpenMP flags, and returns their canonical names. Output: list of strings.
    # Exceptions: CalledProcessError if meson fails.
    # Side effects: subprocesses, writes the build directory.
    """Return names of extension that use OpenMP based on meson introspect output.

    The meson introspect json info is a list of targets where a target is a dict
    that looks like this (parts not used in this script are not shown for simplicity):
    {
      'name': '_k_means_elkan.cpython-312-x86_64-linux-gnu',
      'filename': [
        '<meson_build_dir>/sklearn/cluster/_k_means_elkan.cpython-312-x86_64-linux-gnu.so'
      ],
      'target_sources': [
        {
          'compiler': ['ccache', 'cc'],
          'parameters': [
            '-Wall',
            '-std=c11',
            '-fopenmp',
            ...
          ],
          ...
        },
        {
          'linker': ['cc'],
          'parameters': [
            '-shared',
            '-fPIC',
            '-fopenmp',
            ...
          ]
        }
      ]
    }
    """
    build_path = Path("build/introspect")
    subprocess.check_call(["meson", "setup", build_path, "--reconfigure"])

    json_out = subprocess.check_output(
        ["meson", "introspect", build_path, "--targets"], text=True
    )
    target_list = json.loads(json_out)
    meson_targets = [target for target in target_list if has_openmp_flags(target)]

    return [get_canonical_name_meson(each, build_path) for each in meson_targets]


def get_git_grep_info():
    # [FUNCTION DOC]
    # Lists tracked files matching the OpenMP regex via git grep (Perl regex), keeps
    # those containing ".pyx", and returns their canonical names.
    # Output: list of strings. Exceptions: CalledProcessError if git grep fails
    # (including when nothing matches).
    """Return names of extensions that use OpenMP based on git grep regex."""
    git_grep_filenames = subprocess.check_output(
        ["git", "grep", "-lP", "cython.*parallel|_openmp_helpers"], text=True
    ).splitlines()
    git_grep_filenames = [f for f in git_grep_filenames if ".pyx" in f]

    return [get_canonical_name_git_grep(each) for each in git_grep_filenames]


def main():
    # [FUNCTION DOC]
    # Computes both name sets, builds a message describing names only in git grep
    # (missing openmp_dep) and only in meson (unnecessary openmp_dep).
    # Exceptions: ValueError with that message when the sets differ.
    from_meson = set(get_meson_info())
    from_git_grep = set(get_git_grep_info())

    only_in_git_grep = from_git_grep - from_meson
    only_in_meson = from_meson - from_git_grep

    msg = ""
    if only_in_git_grep:
        only_in_git_grep_msg = "\n".join(
            [f"  {each}" for each in sorted(only_in_git_grep)]
        )
        msg += (
            "Some Cython files use OpenMP,"
            " but their meson.build is missing the openmp_dep dependency:\n"
            f"{only_in_git_grep_msg}\n\n"
        )

    if only_in_meson:
        only_in_meson_msg = "\n".join([f"  {each}" for each in sorted(only_in_meson)])
        msg += (
            "Some Cython files do not use OpenMP,"
            " you should remove openmp_dep from their meson.build:\n"
            f"{only_in_meson_msg}\n\n"
        )

    if from_meson != from_git_grep:
        raise ValueError(
            f"Some issues have been found in Meson OpenMP dependencies:\n\n{msg}"
        )


if __name__ == "__main__":
    main()
