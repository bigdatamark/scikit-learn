# [MODULE DOC]
# Docstring helpers shared by the HTML parameter/attribute tables: building
# text-fragment links into the online docs and extracting per-item numpydoc
# descriptions as HTML-escaped tooltip text.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import html
import inspect
import re
from functools import lru_cache
from urllib.parse import quote

from sklearn.externals._numpydoc import docscrape


def generate_link_to_param_doc(estimator_class, param_name, doc_link):
    # [FUNCTION DOC]
    # Builds a URL that scrolls to a parameter's entry in the online docs. Searches the
    # class docstring for the first line 'param_name : <type>', strips backticks from
    # the
    # type, URL-quotes name and type and returns
    # 'doc_link#:~:text=<name>,-<type>'.
    # Inputs: estimator_class - class; param_name - str; doc_link - base URL str.
    # Outputs: str URL, or None if the docstring is missing or has no match.
    # KNOWN ISSUE: param_name is inserted into the regex unescaped, so names with regex
    # metacharacters could mis-match; the pattern also matches a longer name ending with
    # param_name (e.g. 'alpha' inside 'l1_alpha') since there is no left anchor.
    """URL to the relevant section of the docstring using a Text Fragment

    https://developer.mozilla.org/en-US/docs/Web/URI/Reference/Fragment/Text_fragments
    """
    docstring = estimator_class.__doc__

    m = re.search(f"{param_name} : (.+)\\n", docstring or "")

    if m is None:
        # No match found in the docstring, return None to indicate that we
        # cannot link.
        return None

    # Extract the whole line of the type information, up to the line break as
    # disambiguation suffix to build the fragment
    param_type = m.group(1).replace("`", "")
    text_fragment = f"{quote(param_name)},-{quote(param_type)}"

    return f"{doc_link}#:~:text={text_fragment}"


@lru_cache
def scrape_estimator_docstring(docstring):
    # [FUNCTION DOC]
    # HELPER: Parses a docstring string into a numpydoc NumpyDocString.
    # Inputs: docstring - str. Outputs: NumpyDocString.
    # Side effects: results cached without bound by lru_cache keyed on the string.
    return docscrape.NumpyDocString(docstring)


def get_docstring(estimator_class, section_name, item):
    # [FUNCTION DOC]
    # Looks up item in the given numpydoc section of the class docstring (via
    # inspect.getdoc, so inherited docs count) and returns 'name: type<br><br>' followed
    # by
    # the description lines joined by '<br>', each piece HTML-escaped.
    # Outputs: str or None when no docstring or item not found.
    # Exceptions: KeyError-like errors from numpydoc if section_name is not a valid
    # section; parse warnings/errors from numpydoc propagate.
    """Extract and format docstring information for a specific item.

    Parses the estimator's docstring to retrieve documentation for a
    specific parameter or attribute, formatting it as HTML-escaped text.

    Parameters
    ----------
    estimator_class : type
        The estimator class whose docstring will be parsed.

    section_name : str
        The numpydoc section to search in (e.g., "Parameters", "Attributes").

    item : str
        The name of the parameter or attribute to retrieve documentation for.

    Returns
    -------
    item_description : str or None
        HTML-formatted docstring to be used as a tooltip. Returns None if the
        estimator has no docstring or if the item is not found in the
        specified section.
    """
    estimator_class_docs = inspect.getdoc(estimator_class)
    if estimator_class_docs and (
        structured_docstring := scrape_estimator_docstring(estimator_class_docs)
    ):
        docstring_map = {
            item_docstring.name: item_docstring
            for item_docstring in structured_docstring[section_name]
        }
    else:
        docstring_map = {}
    if item_numpydoc := docstring_map.get(item, None):
        item_description = (
            f"{html.escape(item_numpydoc.name)}: "
            f"{html.escape(item_numpydoc.type)}<br><br>"
            f"{'<br>'.join(html.escape(line) for line in item_numpydoc.desc)}"
        )
    else:
        item_description = None
    return item_description
