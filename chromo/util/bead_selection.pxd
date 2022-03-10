"""Function declarations for bead_selection.pyx

Joseph Wakim
July 18, 2021
"""

cdef long capped_exponential(long window, long cap)
cpdef long from_left(long window, long N_beads)
cpdef long from_right(long window, long N_beads)
cpdef long from_point(long window, long N_beads, long ind0)
cdef (long, long) check_bead_bounds(long bound_0, long bound_1, long num_beads)

