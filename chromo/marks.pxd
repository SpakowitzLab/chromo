"""Class and function declarations for marks.pxd

Joseph Wakim
September 20, 2021
"""

cdef class Mark:
    cdef public str name
    cdef public long sites_per_bead
    cdef public long[:] binding_seq

cdef class Epigenmark(Mark):
    cdef public double bind_energy_mod, bind_energy_no_mod,\
        interaction_energy, chemical_potential, interaction_radius,\
        interaction_volume, field_energy_prefactor,\
        interaction_energy_intranucleosome
