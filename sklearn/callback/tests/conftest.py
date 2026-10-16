# [MODULE DOC]
# HELPER: pytest configuration for the callback test package.
from sklearn.utils.fixes import _IS_WASM


def pytest_ignore_collect(collection_path, config):
    # [FUNCTION DOC]
    # HELPER: pytest hook deciding whether to skip collecting a path.
    # Inputs: collection_path, config - provided by pytest (unused).
    # Outputs: True when running under WASM/Pyodide, which skips all callback tests
    # (multiprocessing/threads are unavailable there); otherwise False.
    # Skip tests for the whole callback module on WASM/Pyodide.
    return _IS_WASM
