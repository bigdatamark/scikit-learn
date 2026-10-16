# [MODULE DOC]
# Pytest configuration for the vendored externals package. Its only role is to stop
# pytest from collecting any tests inside sklearn/externals.
# Do not collect any tests in externals. This is more robust than using
# --ignore because --ignore needs a path and it is not convenient to pass in
# the externals path (very long install-dependent path in site-packages) when
# using --pyargs
def pytest_ignore_collect(collection_path, config):
    # [FUNCTION DOC]
    # Pytest hook that unconditionally excludes every path under this directory from
    # test collection.
    # Inputs: collection_path - path pytest is considering; config - pytest config
    # (both unused).
    # Outputs: always True, meaning "ignore this path".
    # Side effects: none.
    return True
