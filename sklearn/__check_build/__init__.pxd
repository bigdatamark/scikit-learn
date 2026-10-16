# [PACKAGE DOC]
# Package: sklearn.__check_build
# Cython component: _check_build, an extension exposing a no-op check_build().
# This __init__ imports it; failure to import means the compiled extensions were
# not built, and a descriptive ImportError with build instructions is raised.
# (This __init__.pxd holds only documentation; it declares nothing.)
