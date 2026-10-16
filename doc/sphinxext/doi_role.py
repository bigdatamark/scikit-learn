# [MODULE DOC]
# Sphinx extension adding the inline roles doi/DOI and arxiv/arXiv that render as
# external links to https://doi.org/ and https://arxiv.org/abs/ respectively.
"""
doilinks
~~~~~~~~
Extension to add links to DOIs. With this extension you can use e.g.
:doi:`10.1016/S0022-2836(05)80360-2` in your documents. This will
create a link to a DOI resolver
(``https://doi.org/10.1016/S0022-2836(05)80360-2``).
The link caption will be the raw DOI.
You can also give an explicit caption, e.g.
:doi:`Basic local alignment search tool <10.1016/S0022-2836(05)80360-2>`.

:copyright: Copyright 2015  Jon Lund Steffensen. Based on extlinks by
    the Sphinx team.
:license: BSD.
"""

from docutils import nodes, utils
from sphinx.util.nodes import split_explicit_title


def reference_role(typ, rawtext, text, lineno, inliner, options={}, content=[]):
    # [FUNCTION DOC]
    # Role function shared by all four roles. Unescapes the text, splits an optional
    # explicit
    # title "Caption <target>", builds the full URL by prefixing the resolver base to
    # the
    # target and, without an explicit title, uses "arXiv:<target>" or "DOI:<target>" as
    # caption. Inputs: standard docutils role signature (role name typ, raw text, role
    # text,
    # line number, inliner, options, content); options/content unused.
    # Outputs: tuple (list with one reference node, empty list of system messages).
    # KNOWN ISSUE: options and content use mutable default arguments (unused, so
    # harmless).
    # KNOWN ISSUE: for any role name other than the four handled ones the function falls
    # through and returns None instead of a (nodes, messages) tuple; unreachable in
    # practice
    # because only those four names are registered.
    text = utils.unescape(text)
    has_explicit_title, title, part = split_explicit_title(text)
    if typ in ["arXiv", "arxiv"]:
        full_url = "https://arxiv.org/abs/" + part
        if not has_explicit_title:
            title = "arXiv:" + part
        pnode = nodes.reference(title, title, internal=False, refuri=full_url)
        return [pnode], []
    if typ in ["doi", "DOI"]:
        full_url = "https://doi.org/" + part
        if not has_explicit_title:
            title = "DOI:" + part
        pnode = nodes.reference(title, title, internal=False, refuri=full_url)
        return [pnode], []


def setup_link_role(app):
    # [FUNCTION DOC]
    # HELPER: registers reference_role under "arxiv", "arXiv", "doi" and "DOI",
    # overriding
    # any existing roles with those names. Input: Sphinx app. Side effects: mutates the
    # global role registry of the app.
    app.add_role("arxiv", reference_role, override=True)
    app.add_role("arXiv", reference_role, override=True)
    app.add_role("doi", reference_role, override=True)
    app.add_role("DOI", reference_role, override=True)


def setup(app):
    # [FUNCTION DOC]
    # Extension entry point: connects setup_link_role to the "builder-inited" event.
    # Outputs: metadata dict with version "0.1" and parallel_read_safe True.
    app.connect("builder-inited", setup_link_role)
    return {"version": "0.1", "parallel_read_safe": True}
