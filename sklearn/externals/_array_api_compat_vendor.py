# [MODULE DOC]
# HELPER: Hook module looked up by the vendored array_api_extra (_lib/_compat.py) so
#   that it uses scikit-learn's co-vendored copy of array_api_compat.
# Behavior: star-imports every public name of sklearn.externals.array_api_compat into
#   this namespace; the file name must not change because array_api_extra imports it by
#   name.
# Side effects: importing it imports the vendored array_api_compat package.
# DO NOT RENAME THIS FILE
# This is a hook for array_api_extra/_lib/_compat.py
# to co-vendor array_api_compat and potentially override its functions.

from .array_api_compat import *  # noqa: F403
