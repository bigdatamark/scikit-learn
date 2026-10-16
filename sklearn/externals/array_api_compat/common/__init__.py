# [PACKAGE DOC]
# Subpackage of array_api_compat holding backend-agnostic implementations shared by the
#   per-library wrappers (_aliases, _linalg, _fft, _helpers, _typing).
# Public contents: re-exports every public name of _helpers (namespace detection, device
#   handling, array-type predicates). Side effect: imports _helpers at import time.
from ._helpers import *  # noqa: F403
