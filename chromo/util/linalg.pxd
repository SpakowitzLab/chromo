"""Function declarations for linalg.pyx

Joseph Wakim
July 18, 2021
"""
cimport chromo.polymers as poly
cimport numpy as np

cdef np.ndarray identify_4
cdef double[:] uniform_sample_unit_sphere()
cdef void uniform_sample_unit_sphere_inplace(double[:]& arr)
cpdef void arbitrary_axis_rotation(
    double[:] axis,
    double[:] point,
    double rot_angle,
    poly.TransformedObject poly
)
cpdef void generate_translation_mat(
    double[:]& delta, poly.TransformedObject polymer
)
cpdef np.ndarray[double, ndim=2] get_prism_verticies(
    long num_sides,
    double width,
    double height
)
cdef double[:] vec_add3(double[:] v1, double[:] v2)
cdef void inplace_add3(double[:]& v1, double[:]& v2)
cdef double[:, ::1] col_add3(double[:, ::1] c1, double[:, ::1] c2)
cdef double[:] vec_sub3(double[:] v1, double[:] v2)
cdef void inplace_sub3(double[:]& v1, double[:]& v2)
cdef double vec_dot3(double[:] v1, double[:] v2)
cdef double[:] vec_scale3(double[:] v, double factor)
cdef void inplace_scale3(double[:]& v, double factor)
cdef long alt_mod(long x, long n)
