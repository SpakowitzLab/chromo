"""Class and function declarations for binders.pyx

Joseph Wakim
September 20, 2021
"""

cdef class Binder:
    cdef public str name
    cdef public long sites_per_bead
    cdef public long[:] binding_seq

cdef class ReaderProtein(Binder):
    cdef public double bind_energy_mod, bind_energy_no_mod,\
        interaction_energy, chemical_potential, interaction_radius,\
        interaction_volume, field_energy_prefactor,\
        interaction_energy_intranucleosome
    cdef public dict cross_talk_interaction_energy, \
        cross_talk_field_energy_prefactor
