# [MODULE DOC]
# Helpers for sphinx.ext.linkcode: builds a function that maps documented Python objects
# to URLs of their source lines on an online repository at the current git revision.
import inspect
import os
import subprocess
import sys
from functools import partial
from operator import attrgetter

REVISION_CMD = "git rev-parse --short HEAD"


def _get_git_revision():
    # [FUNCTION DOC]
    # HELPER: runs "git rev-parse --short HEAD" in the current working directory.
    # Outputs: short commit hash as str, or None if git fails or is not found.
    # Side effects: spawns a subprocess; prints a message to stdout on failure.
    try:
        revision = subprocess.check_output(REVISION_CMD.split()).strip()
    except (subprocess.CalledProcessError, OSError):
        print("Failed to execute git to get revision")
        return None
    return revision.decode("utf-8")


def _linkcode_resolve(domain, info, package, url_fmt, revision):
    # [FUNCTION DOC]
    # HELPER: resolves the source URL for one documented object. Returns None when
    # revision is
    # None, domain is not "py"/"pyx", or info lacks module/fullname. Otherwise imports
    # the
    # module, resolves the dotted fullname by attribute lookup, unwraps decorators,
    # finds
    # the source file (falling back to the file of the object's __module__), makes the
    # path
    # relative to the package root directory, gets the starting line number (empty
    # string
    # on failure) and formats url_fmt with revision, package, path and lineno.
    # Inputs: domain - str; info - dict with "module" and "fullname"; package - root
    # package name; url_fmt - format string; revision - str or None.
    # Outputs: URL str or None.
    # Exceptions: import errors or AttributeError from the attribute lookup propagate.
    # Side effects: imports modules.
    # KNOWN ISSUE: mixes bare "return" and "return None"; both mean None (cosmetic).
    """Determine a link to online source for a class/method/function

    This is called by sphinx.ext.linkcode

    An example with a long-untouched module that everyone has
    >>> _linkcode_resolve('py', {'module': 'tty',
    ...                          'fullname': 'setraw'},
    ...                   package='tty',
    ...                   url_fmt='https://hg.python.org/cpython/file/'
    ...                           '{revision}/Lib/{package}/{path}#L{lineno}',
    ...                   revision='xxxx')
    'https://hg.python.org/cpython/file/xxxx/Lib/tty/tty.py#L18'
    """

    if revision is None:
        return
    if domain not in ("py", "pyx"):
        return
    if not info.get("module") or not info.get("fullname"):
        return

    class_name = info["fullname"].split(".")[0]
    module = __import__(info["module"], fromlist=[class_name])
    obj = attrgetter(info["fullname"])(module)

    # Unwrap the object to get the correct source
    # file in case that is wrapped by a decorator
    obj = inspect.unwrap(obj)

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return
    try:
        fn = os.path.relpath(fn, start=os.path.dirname(__import__(package).__file__))
    except ValueError:
        return None

    try:
        lineno = inspect.getsourcelines(obj)[1]
    except Exception:
        lineno = ""
    return url_fmt.format(revision=revision, package=package, path=fn, lineno=lineno)


def make_linkcode_resolve(package, url_fmt):
    # [FUNCTION DOC]
    # Gets the git revision once (at call time) and returns _linkcode_resolve partially
    # bound with revision, package and url_fmt; the result takes (domain, info).
    # Side effects: runs git via _get_git_revision.
    """Returns a linkcode_resolve function for the given URL format

    revision is a git commit reference (hash or name)

    package is the name of the root module of the package

    url_fmt is along the lines of ('https://github.com/USER/PROJECT/'
                                   'blob/{revision}/{package}/'
                                   '{path}#L{lineno}')
    """
    revision = _get_git_revision()
    return partial(
        _linkcode_resolve, revision=revision, package=package, url_fmt=url_fmt
    )
