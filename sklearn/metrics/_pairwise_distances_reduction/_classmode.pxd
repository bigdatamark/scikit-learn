# [MODULE DOC]
# Declares the WeightingStrategy enumeration shared by the "classmode" reductions
# (ArgKminClassMode and RadiusNeighborsClassMode). It selects how each neighbor's
# vote is weighted when accumulating per-class scores: uniform (0) adds 1 per
# neighbor, distance (1) adds the inverse of the neighbor's stored distance,
# callable (2) is reserved. Being a cpdef enum it is usable from Cython and
# Python. The classmode constructors map the string "uniform" to uniform,
# "distance" to distance and anything else to callable; callable currently
# behaves like uniform since only distance changes the increment.
cpdef enum WeightingStrategy:
    uniform = 0
    # TODO: Implement the following options in weighted_histogram_mode
    distance = 1
    callable = 2
