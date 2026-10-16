# [MODULE DOC]
# Tests for the parameter HTML table (params module) and the doc-link helper
# generate_link_to_param_doc from common.
import re

import pytest

from sklearn import config_context
from sklearn.utils._repr_html.common import generate_link_to_param_doc
from sklearn.utils._repr_html.params import ParamsDict, _params_html_repr, _read_params


def test_params_dict_content():
    # [FUNCTION DOC]
    # Verifies ParamsDict stores values and the non_default tuple (default empty).
    """Check the behavior of the ParamsDict class."""
    params = ParamsDict(params={"a": 1, "b": 2})
    assert params["a"] == 1
    assert params["b"] == 2
    assert params.non_default == ()

    params = ParamsDict(params={"a": 1, "b": 2}, non_default=("a",))
    assert params["a"] == 1
    assert params["b"] == 2
    assert params.non_default == ("a",)


def test_params_dict_repr_html_():
    # [FUNCTION DOC]
    # Verifies _repr_html_ yields the parameters table and raises AttributeError under
    # display='text'.
    params = ParamsDict(params={"a": 1, "b": 2}, non_default=("a",), estimator_class="")
    out = params._repr_html_()
    assert "<summary>Parameters</summary>" in out

    with config_context(display="text"):
        msg = "_repr_html_ is only defined when"
        with pytest.raises(AttributeError, match=msg):
            params._repr_html_()


def test_params_dict_repr_mimebundle():
    # [FUNCTION DOC]
    # Verifies mime bundle content (plain dict repr and HTML) and that HTML is omitted
    # under display='text'.
    params = ParamsDict(params={"a": 1, "b": 2}, non_default=("a",), estimator_class="")
    out = params._repr_mimebundle_()

    assert "text/plain" in out
    assert "text/html" in out
    assert "<summary>Parameters</summary>" in out["text/html"]
    assert out["text/plain"] == "{'a': 1, 'b': 2}"

    with config_context(display="text"):
        out = params._repr_mimebundle_()
        assert "text/plain" in out
        assert "text/html" not in out


def test_read_params():
    # [FUNCTION DOC]
    # Verifies _read_params classifies default vs user-set and HTML-escapes a script
    # tag injection in the value repr.
    """Check the behavior of the `_read_params` function."""
    out = _read_params("a", 1, tuple())
    assert out["param_type"] == "default"
    assert out["param_name"] == "a"
    assert out["param_value"] == "1"

    # check non-default parameters
    out = _read_params("a", 1, ("a",))
    assert out["param_type"] == "user-set"
    assert out["param_name"] == "a"
    assert out["param_value"] == "1"

    # check that we escape html tags
    tag_injection = "<script>alert('xss')</script>"
    out = _read_params("a", tag_injection, tuple())
    assert (
        out["param_value"]
        == "&quot;&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;&quot;"
    )
    assert out["param_name"] == "a"
    assert out["param_type"] == "default"


def test_params_html_repr():
    # [FUNCTION DOC]
    # Smoke test: renderer output contains both table CSS classes.
    """Check returned HTML template"""
    params = ParamsDict(params={"a": 1, "b": 2}, estimator_class="")
    assert "parameters-table" in _params_html_repr(params)
    assert "estimator-table" in _params_html_repr(params)


def test_params_html_repr_copy_button():
    # [FUNCTION DOC]
    # Verifies each row's copy control is a button with an accessible aria-label.
    """Copy control renders as an accessible <button> with an aria-label."""
    params = ParamsDict(params={"alpha": 1}, estimator_class="")
    html_output = _params_html_repr(params)

    copy_button = (
        r'<button type="button" class="copy-paste-icon"'
        r'\s*aria-label="Copy alpha to clipboard"'
    )
    assert re.search(copy_button, html_output, flags=re.DOTALL)


def test_params_html_repr_with_doc_links():
    # [FUNCTION DOC]
    # Using a mock class with a numpydoc Parameters section and a doc_link, verifies
    # both parameters render as anchors with text-fragment hrefs and escaped hover
    # descriptions (angle brackets escaped, description lines joined by <br>).
    """Test `_params_html_repr` with valid and invalid doc links."""

    class MockEstimator:
        # [CLASS DOC]
        # HELPER: Fake estimator class whose docstring documents a and b; module and
        # qualname are overridden to look like a scikit-learn class.
        """A fake estimator class with a docstring used for testing.

        Parameters
        ----------
        a : int
            Description of a which can include `<formatted text
            https://example.com>`_ that should not be confused with HTML tags.
        b : str
        """

        __module__ = "sklearn.mock_module"
        __qualname__ = "MockEstimator"

    params = ParamsDict(
        params={"a": 1, "b": "value"},
        non_default=("a",),
        estimator_class=MockEstimator,
        doc_link="mock_module.MockEstimator.html",
    )
    html_output = _params_html_repr(params)

    html_param_a = (
        r'<td class="param">'
        r'\s*<a class="param-doc-link"'
        r'\s*style="anchor-name: --doc-link-a;"'
        r'\s*rel="noreferrer" target="_blank"'
        r'\shref="mock_module\.MockEstimator\.html#:~:text=a,-int">'
        r"\s*a"
        r'\s*<span class="param-doc-description"'
        r'\s*style="position-anchor: --doc-link-a;">\s*a:'
        r"\sint<br><br>"
        r"Description of a which can include `&lt;formatted text<br>"
        r"https://example.com&gt;`_ that should not be confused with HTML tags.</span>"
        r"\s*</a>"
        r"\s*</td>"
    )
    assert re.search(html_param_a, html_output, flags=re.DOTALL)
    html_param_b = (
        r'<td class="param">'
        r'.*<a class="param-doc-link"'
        r'\s*style="anchor-name: --doc-link-b;"'
        r'\s*rel="noreferrer" target="_blank"'
        r'\shref="mock_module\.MockEstimator\.html#:~:text=b,-str">'
        r"\s*b"
        r'\s*<span class="param-doc-description"'
        r'\s*style="position-anchor: --doc-link-b;">\s*b:'
        r"\sstr<br><br></span>"
        r"\s*</a>"
        r"\s*</td>"
    )
    assert re.search(html_param_b, html_output, flags=re.DOTALL)


def test_params_html_repr_without_doc_links():
    # [FUNCTION DOC]
    # Verifies that with an undocumented class and no doc_link, names render as plain
    # text cells with no '?' link and values are escaped.
    """Test `_params_html_repr` when `link_to_param_doc` returns None."""

    class MockEstimatorWithoutDoc:
        # [CLASS DOC]
        # HELPER: Fake estimator class without docstring.
        __module__ = "sklearn.mock_module"
        __qualname__ = "MockEstimatorWithoutDoc"
        # No docstring defined on this test class.

    params = ParamsDict(
        params={"a": 1, "b": "value"},
        non_default=("a",),
        estimator_class=MockEstimatorWithoutDoc,
    )
    html_output = _params_html_repr(params)
    # Check that no doc links are generated
    assert "?" not in html_output
    assert "Click to access" not in html_output
    html_param_a = (
        r'<td class="param">a</td>'
        r'\s*<td class="value">1</td>'
    )
    assert re.search(html_param_a, html_output, flags=re.DOTALL)
    html_param_b = (
        r'<td class="param">b</td>'
        r'\s*<td class="value">&#x27;value&#x27;</td>'
    )
    assert re.search(html_param_b, html_output, flags=re.DOTALL)


def test_generate_link_to_param_doc_basic():
    # [FUNCTION DOC]
    # Verifies the URL fragment is 'name,-type' for two documented parameters.
    """Return anchor URLs for documented parameters in the estimator."""

    class MockEstimator:
        # [CLASS DOC]
        # HELPER: Mock class documenting alpha (float) and beta (int).
        """Mock class.

        Parameters
        ----------
        alpha : float
            Regularization strength.
        beta : int
            Some integer parameter.
        """

    doc_link = "mock_module.MockEstimator.html"
    url = generate_link_to_param_doc(MockEstimator, "alpha", doc_link)
    assert url == "mock_module.MockEstimator.html#:~:text=alpha,-float"

    url = generate_link_to_param_doc(MockEstimator, "beta", doc_link)
    assert url == "mock_module.MockEstimator.html#:~:text=beta,-int"


def test_generate_link_to_param_doc_param_not_found():
    # [FUNCTION DOC]
    # Verifies None is returned for an undocumented parameter.
    """Ensure None is returned when the parameter is not documented."""

    class MockEstimator:
        # [CLASS DOC]
        # HELPER: Mock class documenting only alpha.
        """Mock class

        Parameters
        ----------
        alpha : float
            Regularization strength.
        """

    doc_link = "mock_module.MockEstimator.html"
    url = generate_link_to_param_doc(MockEstimator, "gamma", doc_link)

    assert url is None


def test_generate_link_to_param_doc_empty_docstring():
    # [FUNCTION DOC]
    # Verifies None is returned when the class has no docstring.
    """Ensure None is returned when the estimator has no docstring."""

    class MockEstimator:
        # [CLASS DOC]
        # HELPER: Mock class without docstring.
        pass

    doc_link = "mock_module.MockEstimator.html"
    url = generate_link_to_param_doc(MockEstimator, "alpha", doc_link)
    assert url is None


def test_generate_link_to_param_doc_special_char():
    # [FUNCTION DOC]
    # Regression test: a type line containing backticks, parentheses and commas is
    # stripped of backticks and percent-encoded in the fragment.
    """Non-regression test for
    https://github.com/scikit-learn/scikit-learn/issues/33830
    """

    class MockEstimator:
        # [CLASS DOC]
        # HELPER: Mock class with an Attributes entry whose type contains special
        # characters.
        """Mock class.

        Attributes
        ----------
        weird_attr_ : ndarray of shape (`n_features_in_`,)
        """

    doc_link = "mock_module.MockEstimator.html"
    url = generate_link_to_param_doc(MockEstimator, "weird_attr_", doc_link)
    expected_url = (
        "mock_module.MockEstimator.html#:~:text=weird_attr_,"
        "-ndarray%20of%20shape%20%28n_features_in_%2C%29"
    )
    assert url == expected_url
