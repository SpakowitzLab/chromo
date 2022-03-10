"""Class and function declarations for mc_sim.pyx

Joseph Wakim
December 9, 2021
"""
from chromo.polymers cimport PolymerBase
from chromo.mc.moves cimport MCAdapter
from chromo.fields cimport UniformDensityField as UDF

cpdef void mc_sim(
    list polymers, epigenmarks, long num_mc_steps,
    list mc_move_controllers, UDF field, long random_seed
)

cdef void mc_step(
    MCAdapter adaptible_move, PolymerBase poly, epigenmarks,
    UDF field, bint active_field
)