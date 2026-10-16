# [PACKAGE DOC]
# Package: sklearn
# Cython components directly in this top-level package directory:
# - _isotonic (_isotonic.pyx): weighted pool-adjacent-violators solver and a
#   duplicate-X merging routine, both used by sklearn.isotonic.
# - __check_build/_check_build.pyx: trivial extension imported at package import
#   time to detect an unbuilt/broken source checkout (see __check_build).
# Subpackages carry their own [PACKAGE DOC] blocks for their compiled modules.
# (This __init__.pxd holds only documentation; it declares nothing.)
