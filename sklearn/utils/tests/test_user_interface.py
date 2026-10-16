# [MODULE DOC]
# Tests for sklearn.utils._user_interface: _message_with_time formatting and the
# _print_elapsed_time context manager.
import string
import timeit

import pytest

from sklearn.utils._user_interface import _message_with_time, _print_elapsed_time


@pytest.mark.parametrize(
    ["source", "message", "is_long"],
    [
        ("ABC", string.ascii_lowercase, False),
        ("ABCDEF", string.ascii_lowercase, False),
        ("ABC", string.ascii_lowercase * 3, True),
        ("ABC" * 10, string.ascii_lowercase, True),
        ("ABC", string.ascii_lowercase + "\u1048", False),
    ],
)
@pytest.mark.parametrize(
    ["time", "time_str"],
    [
        (0.2, "   0.2s"),
        (20, "  20.0s"),
        (2000, "33.3min"),
        (20000, "333.3min"),
    ],
)
def test_message_with_time(source, message, is_long, time, time_str):
    # [FUNCTION DOC]
    # Verifies the layout of _message_with_time: '[source] ' prefix, dot padding, a
    # space, the message, ', total=' and a right-aligned time string (seconds below 60
    # s, otherwise minutes); total length exactly 70 for short content (non-ASCII
    # included), longer with no dots when content overflows.
    # Inputs: source, message, is_long - paired; time, time_str - paired elapsed seconds
    # and expected formatting.
    out = _message_with_time(source, message, time)
    if is_long:
        assert len(out) > 70
    else:
        assert len(out) == 70

    assert out.startswith("[" + source + "] ")
    out = out[len(source) + 3 :]

    assert out.endswith(time_str)
    out = out[: -len(time_str)]
    assert out.endswith(", total=")
    out = out[: -len(", total=")]
    assert out.endswith(message)
    out = out[: -len(message)]
    assert out.endswith(" ")
    out = out[:-1]

    if is_long:
        assert not out
    else:
        assert list(set(out)) == ["."]


@pytest.mark.parametrize(
    ["message", "expected"],
    [
        ("hello", _message_with_time("ABC", "hello", 0.1) + "\n"),
        ("", _message_with_time("ABC", "", 0.1) + "\n"),
        (None, ""),
    ],
)
def test_print_elapsed_time(message, expected, capsys, monkeypatch):
    # [FUNCTION DOC]
    # Verifies _print_elapsed_time prints the formatted message with 0.1 s elapsed
    # (timer monkeypatched from 0 to 0.1) followed by a newline, and prints nothing when
    # message is None.
    # Inputs: message; expected - output string; capsys, monkeypatch - pytest fixtures.
    monkeypatch.setattr(timeit, "default_timer", lambda: 0)
    with _print_elapsed_time("ABC", message):
        monkeypatch.setattr(timeit, "default_timer", lambda: 0.1)
    assert capsys.readouterr().out == expected
