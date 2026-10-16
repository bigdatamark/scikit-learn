# [MODULE DOC]
# HTML table of an estimator's fitted attributes (names ending in '_'): value
# formatting helper, table renderer and the AttrsDict container whose Jupyter repr
# is that table.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import html
import reprlib
from collections import UserDict

import numpy as np

from sklearn.utils._repr_html.base import ReprHTMLMixin
from sklearn.utils._repr_html.common import (
    generate_link_to_param_doc,
    get_docstring,
)


def _read_fitted_attr(value):
    # [FUNCTION DOC]
    # HELPER: Formats one fitted attribute value for display. ndarrays use numpy
    # array2string with precision 2, comma separator, small values suppressed and
    # summarisation beyond 4 elements, then HTML-escaped. Floats are formatted with 4
    # significant digits (not escaped; not needed). Other values use a reprlib.Repr
    # limited to 4 items for containers and 9 chars for strings, HTML-escaped.
    # Inputs: value - any. Outputs: str.
    if isinstance(value, np.ndarray):
        value = np.array2string(
            value, precision=2, separator=",", suppress_small=True, threshold=4
        )
        return html.escape(value)

    r = reprlib.Repr()

    for attr in (
        "maxlist",
        "maxdict",
        "maxtuple",
        "maxset",
        "maxfrozenset",
        "maxdeque",
        "maxarray",
    ):
        setattr(r, attr, 4)
    r.maxstring = 9

    if isinstance(value, float):
        return f"{value:.4g}"

    return html.escape(r.repr(value))


def _fitted_attr_html_repr(fitted_attributes):
    # [FUNCTION DOC]
    # HELPER: Renders an AttrsDict as a collapsible 'Fitted attributes' table with
    # Name/Type/Value columns. For each attribute: builds a doc link and Attributes
    # docstring tooltip; if both exist and doc_link is set, the name becomes a link
    # with hover description, else a plain styled anchor. When the info dict has 2
    # entries (type_name, value) the type is the escaped type name; otherwise it is
    # treated as array-like and displayed as type_name[dtype]shape. The value column
    # comes from _read_fitted_attr.
    # Inputs: fitted_attributes - AttrsDict mapping name to info dict with keys
    # type_name, value and optionally dtype, shape. Outputs: str HTML.
    # Exceptions: KeyError if info dicts lack expected keys.
    """Generate HTML representation of estimator fitted attributes.

    Creates an HTML table with fitted attribute names and values
    wrapped in a collapsible details element. When attributes are arrays,
    shape and dtype are shown.
    """

    FITTED_ATTR_TEMPLATE = """
        <div class="estimator-table">
            <details>
                <summary>Fitted attributes</summary>
                <table class="parameters-table">
                    <tbody>
                        <tr>
                        <th>Name</th>
                        <th>Type</th>
                        <th>Value</th>
                        </tr>
                        {rows}
                    </tbody>
                </table>
            </details>
        </div>
    """
    FITTED_ATTR_ROW_TEMPLATE = """
       <tr class="default">
           <td class="param">{fitted_attr_display}</td>
           <td class="fitted-att-type">{0}</td>
           <td>{1}</td>


       </tr>
    """

    FITTED_ATTR_AVAILABLE_DOC_LINK_TEMPLATE = """
        <a class="param-doc-link"
            style="anchor-name: --doc-link-{fitted_attr_name};"
            rel="noreferrer" target="_blank" href="{link}">
            {fitted_attr_name}
            <span class="param-doc-description"
            style="position-anchor: --doc-link-{fitted_attr_name};">
            {fitted_attr_description}</span>
        </a>
    """

    rows = []

    for name, attr_info in fitted_attributes.items():
        link = generate_link_to_param_doc(
            fitted_attributes.estimator_class,
            name,
            fitted_attributes.doc_link,
        )
        fitted_attr_description = get_docstring(
            fitted_attributes.estimator_class, "Attributes", name
        )

        if fitted_attributes.doc_link and link and fitted_attr_description:
            # Create clickable attribute name with documentation link
            fitted_attr_display = FITTED_ATTR_AVAILABLE_DOC_LINK_TEMPLATE.format(
                link=link,
                fitted_attr_name=html.escape(name),
                fitted_attr_description=fitted_attr_description,
            )
        else:
            # Just show the attribute name without link
            fitted_attr_display = (
                f'<a class="param-doc-link" style="text-decoration:none;">'
                f"{html.escape(name)}</a>"
            )

        if len(attr_info) == 2:
            html_row_values = (
                html.escape(str(attr_info["type_name"])),
                _read_fitted_attr(attr_info["value"]),  # All the attr info comes here
            )
        else:  # fitted attribute type is array-like
            type_name = html.escape(attr_info["type_name"])
            dtype = html.escape(str(attr_info["dtype"]))
            shape = html.escape(str(attr_info["shape"]))

            html_row_values = (
                f"{type_name}[{dtype}]{shape}",
                _read_fitted_attr(attr_info["value"]),
            )
        rows.append(
            FITTED_ATTR_ROW_TEMPLATE.format(
                *html_row_values,
                fitted_attr_display=fitted_attr_display,
            )
        )

    return FITTED_ATTR_TEMPLATE.format(rows="\n".join(rows))


class AttrsDict(ReprHTMLMixin, UserDict):
    # [CLASS DOC]
    # UserDict of fitted attribute info with an HTML repr (via ReprHTMLMixin and
    # _html_repr set to _fitted_attr_html_repr), carrying estimator_class and doc_link
    # metadata used for documentation links.
    """Dictionary-like class to store and provide an HTML representation.

    It builds an HTML structure to be used with Jupyter notebooks or similar
    environments.

    Parameters
    ----------
    fitted_attrs : dict, default=None
        Dictionary of fitted attributes and their values.

    estimator_class : type, default=None
        The class of the estimator. It allows to find the online documentation
        link for each parameter.

    doc_link : str, default=""
        The base URL to the online documentation for the estimator class.
        Used to generate parameter-specific documentation links in the HTML
        representation. If empty, documentation links will not be generated.
    """

    _html_repr = _fitted_attr_html_repr

    def __init__(self, *, fitted_attrs=None, estimator_class=None, doc_link=""):
        # [FUNCTION DOC]
        # Inputs (keyword-only): fitted_attrs - dict or None (empty dict if falsy);
        # estimator_class - class or None; doc_link - str, default ''.
        # Side effects: fills the dict and sets estimator_class and doc_link.
        super().__init__(fitted_attrs or {})
        self.estimator_class = estimator_class
        self.doc_link = doc_link
