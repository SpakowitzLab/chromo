"""Utility functions for linear algebra calculations.
"""

import pyximport
pyximport.install()

# Built-in Modules
from libc.math cimport sin, cos, acos, pi, floor
from libc.stdlib cimport rand, RAND_MAX
# External Modules
import numpy as np
#cimport numpy as np

# Custom Modules
import chromo.polymers as poly
cimport chromo.polymers as poly
#cimport pandas as pd
#import pandas as pd
cdef np.ndarray identity_4 = np.identity(4, dtype=np.double)




cdef double[:] uniform_sample_unit_sphere():
    """Randomly sample a vector on the unit sphere.

    Returns
    -------
    array_like (3,) of double
        Random (x, y, z) vector on unit sphere
    """
    cdef double phi, theta
    cdef double[:] sample_vec

    phi = <double>rand() / RAND_MAX * (2.0 * pi)
    theta = acos((<double>rand() / RAND_MAX) * 2 - 1)
    sample_vec = np.array([
        cos(phi) * sin(theta),    # x
        sin(phi) * sin(theta),    # y
        cos(theta)                # z
    ])

    return sample_vec


cdef void uniform_sample_unit_sphere_inplace(double[:]& arr):
    """Randomly sample a vector on the unit sphere and update vector in-place.

    Parameters
    ----------
    array_like (3,) of double by reference
        Random (x, y, z) vector on unit sphere to be updated
    """
    cdef double phi, theta

    phi = <double>rand() / RAND_MAX * (2.0 * pi)
    theta = acos((<double>rand() / RAND_MAX) * 2 - 1)
    arr[0] = cos(phi) * sin(theta)  # x
    arr[1] = sin(phi) * sin(theta)  # y
    arr[2] = cos(theta)             # z


cpdef void arbitrary_axis_rotation(
    double[:] axis, double[:] point, double rot_angle,
    poly.TransformedObject poly
):
    """
    Rotate about an axis defined by two points.
    
    Notes
    -----
    Generate a transformation matrix for counterclockwise (right handed
    convention) rotation of angle `rot_angle` about an arbitrary axis from
    points `r0` to `r1`.

    TODO: Store `rot_vec` as attribute of `poly` (to avoid recreating vector)

    Parameters
    ----------
    axis : array_like (3,) of double
        Rotation axis
    point : array_like (3,) of double
        Rotation fulcrum
    rot_angle : double
        Angle of rotation. Positive rotation is counterclockwise when
        rotation axis is pointing directly out of the screen
    poly : poly.TransformedObject
        Container holding the transformation matrix
    """
    cdef double[:] rot_vec
    cdef long i

    for i in range(4):
        for j in range(4):
            poly.transformation_mat[i, j] = 0
    for i in range(4):
        poly.transformation_mat[i, i] = 1

    poly.transformation_mat[0, 0] = axis[0]**2 + (axis[1]**2 + axis[2]**2) *\
        cos(rot_angle)
    poly.transformation_mat[0, 1] = axis[0] * axis[1] * (1 - cos(rot_angle)) -\
        axis[2]*sin(rot_angle)
    poly.transformation_mat[0, 2] = axis[0] * axis[2] * (1 - cos(rot_angle)) +\
        axis[1]*sin(rot_angle)

    poly.transformation_mat[1, 0] = axis[0] * axis[1] * (1 - cos(rot_angle)) +\
        axis[2]*sin(rot_angle)
    poly.transformation_mat[1, 1] = axis[1]**2 + (axis[0]**2 + axis[2]**2) *\
        cos(rot_angle)
    poly.transformation_mat[1, 2] = axis[1] * axis[2]*(1 - cos(rot_angle)) -\
        axis[0]*sin(rot_angle)

    poly.transformation_mat[2, 0] = axis[0] * axis[2] * (1 - cos(rot_angle)) -\
        axis[1]*sin(rot_angle)
    poly.transformation_mat[2, 1] = axis[1] * axis[2] * (1 - cos(rot_angle)) +\
        axis[0]*sin(rot_angle)
    poly.transformation_mat[2, 2] = axis[2]**2 + (axis[0]**2 + axis[1]**2) *\
        cos(rot_angle)

    rot_vec = np.zeros(3)
    # This is slower: rot_vec = np.cross(point, axis) * sin(rot_angle)
    rot_vec[0] = (point[1] * axis[2] - point[2] * axis[1]) * sin(rot_angle)
    rot_vec[1] = (point[2] * axis[0] - point[0] * axis[2]) * sin(rot_angle)
    rot_vec[2] = (point[0] * axis[1] - point[1] * axis[0]) * sin(rot_angle)

    rot_vec[0] += (
        point[0] * (1 - axis[0]**2) - axis[0] *
        (point[1] * axis[1] + point[2] * axis[2])
        ) * (1 - cos(rot_angle))
    rot_vec[1] += (
        point[1] * (1 - axis[1]**2) - axis[1] *
        (point[0] * axis[0] + point[2] * axis[2])
        ) * (1 - cos(rot_angle))
    rot_vec[2] += (
        point[2]*(1 - axis[2]**2) - axis[2] *
        (point[0] * axis[0] + point[1] * axis[1])
        ) * (1 - cos(rot_angle))

    for i in range(3):
        poly.transformation_mat[i, 3] = rot_vec[i]


def return_arbitrary_axis_rotation(axis, point, rot_angle):
    """Generate & return homogeneous transform matrix of arbitrary axis rot.
    
    Notes
    -----
    Use this function when you want to return a transformation matrix, not just
    update the `transformation_mat` attribute of a polymer.
    
    This function is not yet tested.
    
    Parameters
    ----------
    axis : array_like (3,) of double
        Rotation axis
    point : array_like (3,) of double
        Rotation fulcrum
    rot_angle : double
        Angle of rotation. Positive rotation is counterclockwise when
        rotation axis is pointing directly out of the screen
        
    Returns
    -------
    double[:, ::1]
        Homogeneous transformation matrix encoding arbitrary axis rotation.
    """
    transformation_mat_obj = poly.TransformedObject()
    arbitrary_axis_rotation(axis, point, rot_angle, transformation_mat_obj)
    return transformation_mat_obj.transformation_mat


cpdef void generate_translation_mat(
    double[:]& delta, poly.TransformedObject polymer
):
    """Generate translation matrix.

    Notes
    -----
    Generate the homogeneous transformation matrix for a translation
    of distance delta_x, delta_y, delta_z in the x, y, and z directions,
    respectively.

    Parameters
    ----------
    delta : array_like (3,) of double
        Array of translation magnitudes in x, y, z directions, in form
        [dx, dy, dz]
    polymer : poly.TransformedObject
        Container holding the transformation matrix
    """
    cdef long i, j

    for i in range(4):
        for j in range(4):
            polymer.transformation_mat[i, j] = 0
    for i in range(4):
        polymer.transformation_mat[i, i] = 1
    for i in range(3):
        polymer.transformation_mat[i, 3] = delta[i]


cpdef np.ndarray[double, ndim=2] get_prism_verticies(
    long num_sides, double width, double height
):
    """Get the cartesian coordinates of verticies for specified prism geometry.

    Parameters
    ----------
    num_sides : int
        Number of sides on the face of the prism used to represent the
        nucleosome's geometry; this determines the locations of verticies
        of the `Prism`
    width : float
        Determines the shape of the prism defining the location of the
        nucleosome's verticies. The `width` gives the diameter of the
        circle circumscribing the base of the prism in the simulation's
        distance units.
    height : float
        Determines the shape of the prism defining the location of the
        nucleosome's verticies. The `height` gives the height of the prism
        in the simulation's distance units.

    Returns
    -------
    np.ndarray (M, 3) of double
        Cartesian coordinates of the M verticies of the prism geometry.
    """
    cdef double ang
    cdef np.ndarray[double, ndim=1] base_coords_2D, new_coords
    cdef np.ndarray[double, ndim=2] rot_mat, verticies

    ang = 2 * pi / num_sides
    base_coords_2D = np.array([1, 0])
    rot_mat = np.array(
        [[cos(ang), -sin(ang)],
         [sin(ang), cos(ang)]]
    )
    for i in range(1, num_sides):
        new_coords = np.matmul(rot_mat, base_coords_2D[i-1])
        base_coords_2D = np.stack(base_coords_2D, new_coords)
    base_coords_2D *= width / 2
    verticies = np.zeros((num_sides * 2, 3))
    for i in range(num_sides):
        verticies[i, 0:2] = base_coords_2D[i]
        verticies[i, 2] = -height / 2
        verticies[2 * i, 0:2] = base_coords_2D[i]
        verticies[2 * i, 2] = height / 2
    return verticies


cdef double[:] vec_add3(double[:] v1, double[:] v2):
    """Element-wise addition of two size-three memoryviews of doubles.
    
    Notes
    -----
    A new vector equal to the element-wise sum is created and returned by this
    method. This results in a slower runtime than `inplace_add3`.

    Parameters
    ----------
    v1 : array_like (3,) of double
        First size-three memoryview of doubles to be added
    v2 : array_like (3,) of double
        Second size-three memoryview of doubles to be added

    Returns
    -------
    array_like (3,) of double
        Element-wise sum of vectors
    """
    cdef double[:] v3
    cdef int i
    v3 = v1.copy()
    for i in range(3):
        v3[i] += v2[i]
    return v3


cdef void inplace_add3(double[:]& v1, double[:]& v2):
    """Element-wise addition of two size-three memoryviews of doubles in place.

    Elements of v2 are added to respective elements of v1 IN PLACE, producing
    an  updated v1.

    Parameters
    ----------
    v1 : array_like (3,) of double
        First size-three memoryview of doubles to be added in-place
    v2 : array_like (3,) of double
        Second size-three memoryview of doubles to be added in-place
    """
    cdef int i
    for i in range(3):
        v1[i] += v2[i]


cdef double[:, ::1] col_add3(double[:, ::1] c1, double[:, ::1] c2):
    """Element-wise addition of two size (1 x 3) memoryviews of doubles.

    Parameters
    ----------
    c1 : array_like (1, 3) of double
        First size-three memoryview (column format) of doubles to be added
    c2 : array_like (1, 3) of double
        Second size-three memoryview (column format) of doubles to be added

    Returns
    -------
    array_like (1, 3) of double
        Element-wise sum of columns
    """
    cdef double[:, ::1] c3
    cdef long i

    c3 = c1.copy()
    for i in range(3):
        c3[0, i] += c2[0, i]
    
    return c3


cdef double[:] vec_sub3(double[:] v1, double[:] v2):
    """Element-wise difference of two size-three memoryviews of doubles.

    Notes
    -----
    A new vector equal to the element-wise difference is created and returned
    by this method. This results in a slower runtime than `inplace_sub3`.

    Parameters
    ----------
    v1 : array_like (3,) of double
        First size-three memoryview of doubles in subtraction
    v2 : array_like (3,) of double
        Second size-three memoryview of doubles in subtraction

    Returns
    -------
    array_like (3,) of double
        Element-wise difference of vectors (v1 - v2)
    """
    cdef double[:] v3
    cdef long i

    v3 = v1.copy()
    for i in range(3):
        v3[i] -= v2[i]
    
    return v3


cdef void inplace_sub3(double[:]& v1, double[:]& v2):
    """Element-wise subtraction of two size-three doubles memoryviews in place.

    Notes
    -----
    Elements of v2 are subtracted from respective elements of v1 IN PLACE,
    producing an  updated v1.

    Parameters
    ----------
    v1 : array_like (3,) of double
        First size-three memoryview of doubles for in-place subtraction of
        (`v1` - `v2`); this is the vector that gets updated during the in-place
        operation
    v2 : array_like (3,) of double
        Second size-three memoryview of doubles for in-place subtraction
    """
    cdef int i
    for i in range(3):
        v1[i] -= v2[i]


cdef double vec_dot3(double[:] v1, double[:] v2):
    """Dot product of two size-three memoryview of doubles.

    Parameters
    ----------
    v1 : array_like (3,) of double
        First size-three memoryview of doubles in dot product
    v2 : array_like (3,) of double
        Second size-three memoryview of doubles in dot product

    Returns
    -------
    array_like (3,) of double
        Dot product of two vectors
    """
    cdef double dot
    cdef long i

    dot = 0
    for i in range(3):
        dot += v1[i] * v2[i]

    return dot


cdef double[:] vec_scale3(double[:] v, double factor):
    """Scale a size-three memoryview of doubles by a constant factor.

    Parameters
    ----------
    v : array_like (3,) of double
        Vector to be scaled
    factor : double
        Factor by which to scale memoryview

    Return
    ------
    array_like (3,) of double
        Scaled vector
    """
    cdef double[:] v_scaled = v.copy()

    for i in range(3):
        v_scaled[i] *= factor

    return v_scaled


cdef void inplace_scale3(double[:]& v, double factor):
    """Scale a size-three memoryview of doubles by a constant factor in-place.

    Parameters
    ----------
    v : array_like (3,) of double
        Vector to be scaled in-place
    factor : double
        Factor by which to scale memoryview
    """
    cdef int i
    for i in range(3):
        v[i] *= factor


cdef long alt_mod(long x, long n):
    """Arithmetic solution to (x % n).
    
    Notes
    -----
    I have not yet determined whether or not this operation is in fact more
    efficient.

    Parameters
    ----------
    x : long
        Dividend of modulo operation
    n : long
        Divisor of modulo operation
    
    Returns
    -------
    long
        x % n calculated using less "expensive" arithmetic
    """
    cdef long filled, remainder
    filled = <long>floor(x / n)
    remainder = x - filled * n
    return remainder
