"""Class and function declarations for `polymers.pyx`.

Joseph Wakim
July 21, 2021
"""

cimport numpy as np

cdef class TransformedObject:
    cdef public double[:, ::1] transformation_mat

cdef class PolymerBase(TransformedObject):
    cdef public str name, log_path
    cdef public beads
    cdef public configuration_tracker
    cdef public double lp
    cdef public long num_marks, num_beads, n_marks_p1
    cdef public long[:] all_inds
    cdef public double[:, ::1] r, t3, t2, r_trial, t3_trial, t2_trial
    cdef public long[:, ::1] states, states_trial
    cdef public long[:, ::1] chemical_mods
    cdef public np.ndarray chemical_mod_names
    cdef public double[:] direction, point
    cdef public double last_amp_move
    cdef public long last_amp_bead
    cdef public np.ndarray mark_names, required_attrs, _arrays, _3d_arrays
    cdef public double[:] dr, dr_test, dr_perp, dr_perp_test, bend, bend_test
    cdef public double[:, :, ::1] densities_temp
    cdef double compute_dE(
        self,
        str move_name,
        long[:] inds,
        long n_inds
    )
    cpdef void update_log_path(self, str log_path)
    cdef void construct_beads(self)
    cdef void check_marks(self, long[:, ::1] states, np.ndarray mark_names)
    cdef void check_chemical_mods(
        self, long[:, ::1] chemical_mods,
        np.ndarray chemical_mod_names
    )
    cpdef np.ndarray get_prop(self, long[:] inds, str prop)
    cpdef np.ndarray get_all(self, str prop)
    cpdef long get_num_marks(self)
    cpdef long get_num_beads(self)
    cpdef bint is_field_active(self)

cdef class Rouse(PolymerBase):
    cdef void construct_beads(self)

cdef class SSWLC(PolymerBase):
    cdef public double delta, eps_bend, eps_par, eps_perp, gamma, eta
    cdef public double bead_length, bead_rad
    cdef void construct_beads(self)
    cpdef double compute_E(self)
    cdef double compute_dE(
        self,
        str move_name,
        long[:] inds,
        long n_inds
    )
    cdef double continuous_dE_poly(
        self,
        long ind0,
        long indf,
    )
    cdef double E_pair(self, double[:] bend, double dr_par, double[:] dr_perp)
    cdef double bead_pair_dE_poly_forward(
        self,
        double[:] r_0,
        double[:] r_1,
        double[:] test_r_1,
        double[:] t3_0,
        double[:] t3_1,
        double[:] test_t3_1
    )
    cdef double bead_pair_dE_poly_reverse(
        self,
        double[:] r_0,
        double[:] test_r_0,
        double[:] r_1,
        double[:] t3_0,
        double[:] test_t3_0,
        double[:] t3_1
    )
    cdef double binding_dE(self, long ind0, long indf, long n_inds)
    cdef double bead_binding_dE(self, long ind, long[:] states_trial_ind)
    cpdef void _find_parameters(self, double length_bead)

cdef class Chromatin(SSWLC):
    cdef double compute_dE(
        self,
        str move_name,
        long[:] inds,
        long n_inds
    )

cdef class SSTWLC(SSWLC):
    cdef public double lt, eps_twist
    cdef double E_pair_with_twist(
        self, double[:] bend, double dr_par, double[:] dr_perp, double omega
    )
    cdef double compute_dE(
        self,
        str move_name,
        long[:] inds,
        long n_inds
    )
    cdef double bead_pair_dE_poly_forward_with_twist(
        self,
        double[:] r_0,
        double[:] r_1,
        double[:] test_r_1,
        double[:] t3_0,
        double[:] t3_1,
        double[:] test_t3_1,
        double[:] t2_0,
        double[:] t2_1,
        double[:] test_t2_1
    )
    cdef double bead_pair_dE_poly_reverse_with_twist(
        self,
        double[:] r_0,
        double[:] test_r_0,
        double[:] r_1,
        double[:] t3_0,
        double[:] test_t3_0,
        double[:] t3_1,
        double[:] t2_0,
        double[:] test_t2_0,
        double[:] t2_1
    )

cdef class LoopedSSTWLC(SSTWLC):
    pass

cpdef double sin_func(double x)
cpdef double helix_parametric_x(double t)
cpdef double helix_parametric_y(double t)
cpdef double helix_parametric_z(double t)