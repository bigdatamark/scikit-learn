# [MODULE DOC]
# HELPER: Small user-interface utilities for verbose output: formatting a fixed-width
# timing log line and a context manager that prints elapsed time on exit.
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import timeit
from contextlib import contextmanager


def _message_with_time(source, message, time):
    # [FUNCTION DOC]
    # HELPER: Builds a 70-character log line "[source] ....... message, total=T". Time
    # is formatted as minutes with one decimal ("%4.1fmin") when above 60 seconds,
    # otherwise seconds ("%5.1fs" with a leading space). The gap between the prefix
    # and the suffix is filled with dots so the total is 70 characters (no dots if the
    # parts already exceed 70).
    # Inputs: source - str; message - str; time - number of seconds (int or float).
    # Outputs: the formatted str.
    # Exceptions/side effects: none.
    """Create one line message for logging purposes.

    Parameters
    ----------
    source : str
        String indicating the source or the reference of the message.

    message : str
        Short message.

    time : int
        Time in seconds.
    """
    start_message = "[%s] " % source

    # adapted from joblib.logger.short_format_time without the Windows -.1s
    # adjustment
    if time > 60:
        time_str = "%4.1fmin" % (time / 60)
    else:
        time_str = " %5.1fs" % time
    end_message = " %s, total=%s" % (message, time_str)
    dots_len = 70 - len(start_message) - len(end_message)
    return "%s%s%s" % (start_message, dots_len * ".", end_message)


@contextmanager
def _print_elapsed_time(source, message=None):
    # [FUNCTION DOC]
    # HELPER: Generator-based context manager. When message is None it simply yields
    # and prints nothing. Otherwise it records a timeit.default_timer timestamp,
    # yields, and after the block completes prints _message_with_time(source, message,
    # elapsed) to stdout.
    # Inputs: source - str; message - str or None (default None).
    # Outputs: yields None.
    # Side effects: prints to stdout.
    # KNOWN ISSUE: the yield is not wrapped in try/finally, so nothing is printed if
    # the block raises.
    """Log elapsed time to stdout when the context is exited.

    Parameters
    ----------
    source : str
        String indicating the source or the reference of the message.

    message : str, default=None
        Short message. If None, nothing will be printed.

    Returns
    -------
    context_manager
        Prints elapsed time upon exit if verbose.
    """
    if message is None:
        yield
    else:
        start = timeit.default_timer()
        yield
        print(_message_with_time(source, message, timeit.default_timer() - start))
