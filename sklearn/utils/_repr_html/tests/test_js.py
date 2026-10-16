# [MODULE DOC]
# Browser tests (playwright, via the pytest 'page' fixture) of the JavaScript in
# estimator.js: copying parameter names and feature names to the clipboard and the
# theme detection forceTheme. Pages are served from a throwaway local HTTP server.
# Skipped entirely when playwright is not installed.
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def check_playwright():
    # [FUNCTION DOC]
    # HELPER: Session-scoped autouse fixture that skips all tests in the module when
    # playwright cannot be imported; returns the module otherwise.
    # KNOWN ISSUE: the docstring says it is used by 'the next fixture', but it is
    # itself autouse and local_server does not depend on it.
    """Skip tests if playwright is not installed.

    This fixture is used by the next fixture (which is autouse) to skip all tests
    if playwright is not installed."""
    return pytest.importorskip("playwright")


@pytest.fixture
def local_server(request):
    # [FUNCTION DOC]
    # HELPER: Fixture starting an HTTPServer on a free 127.0.0.1 port in a daemon
    # thread. Yields (base URL, setter) where the setter replaces the HTML returned for
    # every GET. Shuts the server down after the test.
    # KNOWN ISSUE: the free port is found by binding and closing a temporary socket, so
    # another process could grab it before the server binds (small race).
    """Start a simple HTTP server that serves custom HTML per test.

    Usage :

    ```python
    def test_something(page, local_server):
        url, set_html_response = local_server
        set_html_response("<html>...</html>")
        page.goto(url)
        ...
    ```
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        PORT = s.getsockname()[1]

    html_content = "<html><body>Default</body></html>"

    def set_html_response(content):
        # [FUNCTION DOC]
        # HELPER: Replaces the served HTML string (closure variable). Inputs: content -
        # str.
        nonlocal html_content
        html_content = content

    class Handler(BaseHTTPRequestHandler):
        # [CLASS DOC]
        # HELPER: Request handler serving the current HTML content for GET requests.
        def do_GET(self):
            # [FUNCTION DOC]
            # HELPER: Responds 200 with text/html and the current content UTF-8 encoded.
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))

        # suppress logging
        def log_message(self, format, *args):
            # [FUNCTION DOC]
            # HELPER: No-op override suppressing request logging to stderr.
            return

    httpd = HTTPServer(("127.0.0.1", PORT), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{PORT}", set_html_response

    httpd.shutdown()


def _make_page(body):
    # [FUNCTION DOC]
    # HELPER: Reads estimator.js (UTF-8) from the parent package and returns a full
    # HTML page with it inlined in the head and body inserted in the body.
    # Inputs: body - str HTML. Outputs: str. Side effects: file read.
    """Helper to create an HTML page that includes `estimator.js` and the given body."""

    js_path = Path(__file__).parent.parent / "estimator.js"
    with open(js_path, "r", encoding="utf-8") as f:
        script = f.read()

    return f"""
    <html>
      <head>
      <script>{script}</script>
      </head>
      <body>
        {body}
      </body>
    </html>
    """


def test_copy_paste(page, local_server):
    # [FUNCTION DOC]
    # Serves a content div with data-param-prefix, calls copyToClipboard in the
    # browser with clipboard permissions granted, and verifies the clipboard holds the
    # prefix concatenated with the name.
    """Test that copyToClipboard copies the right text to the clipboard.

    Test requires clipboard permissions, which are granted through page's context.
    Assertion is done by reading back the clipboard content from the browser.
    This is easier than writing a cross platform clipboard reader.
    """
    url, set_html_response = local_server

    copy_paste_html = _make_page(
        '<div class="sk-toggleable__content" data-param-prefix="prefix"/>'
    )

    set_html_response(copy_paste_html)
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    page.goto(url)
    page.evaluate(
        "copyToClipboard('test', document.querySelector('.sk-toggleable__content'))"
    )
    clipboard_content = page.evaluate("navigator.clipboard.readText()")

    # `copyToClipboard` function concatenates the `data-param-prefix` attribute
    #  with the first argument. Hence we expect "prefixtest" and not just test.
    assert clipboard_content == "prefixtest"


@pytest.mark.parametrize(
    "color,expected_theme",
    [
        (
            "black",
            "light",
        ),
        (
            "white",
            "dark",
        ),
        (
            "#828282",
            "light",
        ),
    ],
)
def test_force_theme(page, local_server, color, expected_theme):
    # [FUNCTION DOC]
    # Parametrized over text colours. Verifies forceTheme adds the 'light' class for
    # dark text and 'dark' for light text on the target element.
    """Test that forceTheme applies the right theme class to the element.

    A light color must lead to a dark theme and vice-versa.
    """
    url, set_html_response = local_server

    html = _make_page('<div style="color: ${color};"><div id="test"></div></div>')
    set_html_response(html.replace("${color}", color))
    page.goto(url)
    page.evaluate("forceTheme('test')")
    assert page.locator("#test").evaluate(
        f"el => el.classList.contains('{expected_theme}')"
    )


FEATURE_NAMES_HTML = """
                        <div class="features">
                            <details>
                                <summary>
                                    <div class="image-container">
                                        <button type="button" class="copy-paste-icon"
                                            title="Copy output features (max 100)"
                                            aria-label="Copy output features (max 100)">
                                        </button>
                                    </div>
                                </summary>
                                <div class="features-container">
                                    <table class="features-table">
                                        <tbody>
                                            <tr><td>feature1</td></tr>
                                            <tr><td>feature2</td></tr>
                                        </tbody>
                                    </table>
                                </div>
                            </details>
                        </div>
                     """


def test_copy_paste_feature_names(page, local_server):
    # [FUNCTION DOC]
    # Serves a static features box and verifies copyFeatureNamesToClipboard places a
    # formatted list of the table's feature names on the clipboard.
    """Test that copyFeatureNamesToClipboard copies the right text to the clipboard.

    Test requires clipboard permissions, which are granted through page's context.
    Assertion is done by reading back the clipboard content from the browser.
    This is easier than writing a cross platform clipboard reader.

    Test adapted from test_copy_paste
    """
    url, set_html_response = local_server

    copy_paste_html = _make_page(FEATURE_NAMES_HTML)

    set_html_response(copy_paste_html)
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    page.goto(url)
    page.evaluate(
        "copyFeatureNamesToClipboard(document.querySelector('.copy-paste-icon'))"
    )
    clipboard_content = page.evaluate("navigator.clipboard.readText()")

    assert clipboard_content == '[\n    "feature1",\n    "feature2",\n]'
