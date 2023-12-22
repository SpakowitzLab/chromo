"""Beads represent monomeric units forming the polymer.
"""

# Built-in Modules
from abc import ABC, abstractmethod
from typing import (Sequence, Optional, List)

# External Modules
import numpy as np

# Custom Modules
from .binders import Binder, get_by_name
from .util import linalg as la
from .util.gjk import gjk_collision


# Length of a base pair of DNA in nanometers
LENGTH_BP = 0.332

# Rise per helical turn of DNA around the nucleosome
# Modeled after the thickness of DNA, which is 2nm
RISE_PER_LAP = 2.0

# Natural twist of DNA wrapped around the nucleosome (in radians / bp)
NATURAL_TWIST_NUCLEOSOME = 2 * np.pi / 10.17

class Bead(ABC):
    """Abstract class representation of a bead of a polymer.

    Attributes
    ----------
    id : int
        Bead identifier; typically the beads index position in the chain
    r : array_like (3,) of double
        Position of the bead
    t3 : array_like (3,) of double
        Tangential orientation vector defining the bead
    t2 : array_like (3,) of double
        Tangential orientation vector defining the bead; orthogonal to `t3`
    states : array_like (M,) of int
        State of each bound protein on the bead
    binder_names : array_like (M,) of str
        Name of each bound protein on the bead
    """

    def __init__(
        self, id_: int, r: np.ndarray, t3: Optional[np.ndarray] = None,
        t2: Optional[np.ndarray] = None, states: Optional[np.ndarray] = None,
        binder_names: Optional[Sequence[str]] = None
    ):
        """Initialize the bead object.

        Notes
        -----
        These are the required attributes that any type of bead must have for
        proper compatibility with the rest of the simulator.

        Parameters
        ----------
        id_ : int
            Bead identifier
        r : np.ndarray (3,)
            Coordinates of the nucleosome in form (x, y, z)
        t3 : Optional np.ndarray (3,)
            Tangent vector defining orientation of nucleosome, default = `None`
        t2 : Optional np.ndarray (3,)
            Tangent vector orthogonal to `t3` defining orientation of
            nucleosome; a third orthogonal tangent vector can be obtained
            from the cross product of `t2` and `t3`, default = `None`
        states : Optional[np.ndarray] (M, ) of int
            State of each of the M binders being tracked, default = None
        binder_names : Optional[Sequence[str]] (M, )
            The name of each bound component tracked in `states`; binder names
            are how the properties of the binders are loaded, default = None
        """
        self.id = id_
        self.r = r
        self.t3 = t3
        self.t2 = t2
        self.states = states
        self.binder_names = binder_names
        if binder_names is not None:
            self.binders: Optional[List[Binder]] =\
                [get_by_name(name) for name in binder_names]
        else:
            self.binders = None

    @abstractmethod
    def test_collision(self, **kwargs):
        """Test collisions with another bead or a confinement.
        """
        pass

    @abstractmethod
    def print_properties(self):
        """Print properties of the bead.
        """
        pass


class GhostBead(Bead):
    """Class representation of a "ghost bead" for which collisions are ignored.

    Notes
    -----
    See documentation for `Bead` class for additional attribute definitions.

    Attributes
    ----------
    rad : double
        Radius of the spherical ghost bead
    vol : double
        Volume of the spherical ghost bead
    kwargs : Dict
        Additional named properties of the ghost bead (for later use)
    """

    def __init__(
        self, id_: int, r: np.ndarray, *, t3: Optional[np.ndarray] = None,
        t2: Optional[np.ndarray] = None, states: Optional[np.ndarray] = None,
        binder_names: Optional[Sequence[str]] = None, rad: Optional[float] = 5,
        **kwargs
    ):
        """Initialize the non-interacting ghost bead object.

        Parameters
        ----------
        id_ : int
            Bead identifier
        r : np.ndarray (3,)
            Coordinates of the nucleosome in form (x, y, z)
        t3 : Optional np.ndarray (3,)
            Tangent vector defining orientation of nucleosome, default = `None`
        t2 : Optional np.ndarray (3,)
            Tangent vector orthogonal to `t3` defining orientation of
            nucleosome; a third orthogonal tangent vector can be obtained
            from the cross product of `t2` and `t3`, default = `None`
        states : Optional[np.ndarray] (M, ) of int
            State of each of the M binders being tracked, default = None
        binder_names : Optional[Sequence[str]] (M, )
            The name of each bound component tracked in `states`; binder names
            are how the properties of the binders are loaded, default = None
        rad : Optional[float]
            Radius of the spherical ghost particle.
        """
        super(GhostBead, self).__init__(
            id_, r, t3, t2, states, binder_names
        )
        self.rad = rad
        self.vol = (4/3) * np.pi * rad ** 3
        self.kwargs = kwargs

    def test_collision(self):
        """Collisions are neglected for a ghost bead.
        """
        return False

    def print_properties(self):
        """Print properties of the ghost bead.
        """
        print("Ghost Bead ID: ", self.id)
        print("Central Position: ")
        print(self.r)
        print("t3 Orientation: ")
        print(self.t3)
        print("t2 Orientation: ")
        print(self.t2)


def get_verticies_rot_mat(current_vec, target_vec, fulcrum):
    """Rotate verticies defined relative to `current_vec`.

    Parameters
    ----------
    current_vec : array_like (3,) of double
        Current vector relative to which verticies are defined
    target_vec : array_like (3,) of double
        Target vector relative to which verticies should be defined
    fulcrum : array_like (3,) of double
        Point about which rotation will take place

    Returns
    -------
    array_like (4, 4) of double
        Homogeneous rotation matrix with which to rotate verticies
    """
    axis = np.cross(current_vec, target_vec)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(
        np.dot(current_vec, target_vec) /
        (np.linalg.norm(current_vec) * np.linalg.norm(target_vec))
    )
    return la.return_arbitrary_axis_rotation(axis, fulcrum, angle)


class Prism(Bead):
    """Class representation of prism-shaped monomer, enabling sterics.

    Notes
    -----
    The `Prism` objects allow for careful evaluation of collisions between
    nucleosome beads.

    This class is not yet tested.

    See documentation for `Bead` class for additional attribute definitions.

    Attributes
    ----------
    vertices : array_like (M, 3)
        Vertices representing a mesh of the nucleosome bead. The verticies
        are defined around the origin, with orientations such that t3
        coincides with the positive x axis and t2 coincides with the
        positive z axis.
    """

    def __init__(
        self, id_: int, r: np.ndarray, *, t3: np.ndarray, t2: np.ndarray,
        vertices: np.ndarray, states: Optional[np.ndarray] = None,
        binder_names: Optional[Sequence[str]] = None
    ):
        """Initialize prism object.

        Parameters
        ----------
        id_ : int
            Identifier for the nucleosome
        r : np.ndarray (3,)
            Coordinates of the nucleosome in form (x, y, z)
        t3 : np.ndarray (3,)
            Tangent vector defining orientation of nucleosome
        t2 : np.ndarray (3,)
            Tangent vector orthogonal to `t3` defining orientation of
            nucleosome; a third orthogonal tangent vector can be obtained
            from the cross product of `t2` and `t3`
        vertices : np.ndarray (M, 3)
            Vertices representing a mesh of the nucleosome bead. The verticies
            are defined around the origin, with orientations such that t3
            coincides with the positive x-axis and t2 coincides with the
            positive z-axis.
        states : Optional[np.ndarray] (M, ) of int
            State of each of the M epigenetic binders being tracked
        binder_names : Optional[Sequence[str]] (M, )
            The name of each bound component tracked in `states`; binder names
            are how the properties of the binders are loaded, default = None
        """
        super(Prism, self).__init__(
            id_, r, t3, t2, states, binder_names
        )
        self.vertices = vertices

    @classmethod
    def construct_nucleosome(
        cls, id_: int, r: np.ndarray, *, t3: np.ndarray, t2: np.ndarray,
        num_sides: int, width: float, height: float,
        states: Optional[np.ndarray] = None,
        binder_names: Optional[Sequence[str]] = None
    ):
        """Construct nucleosome as a prism w/ specified position & orientation.

        Parameters
        ----------
        id_ : int
            Identifier for the nucleosome
        r : np.ndarray (3,)
            Coordinates of the nucleosome in form (x, y, z)
        t3 : np.ndarray (3,)
            Tangent vector defining orientation of nucleosome
        t2 : np.ndarray (3,)
            Tangent vector orthogonal to `t3` defining orientation of
            nucleosome; a third orthogonal tangent vector can be obtained
            from the cross product of `t2` and `t3`
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
        states : Optional[np.ndarray] (M,) of int
            State of each of the M epigenetic binders being tracked
        binder_names : Optional[Sequence[str]] (M, )
            The name of each bound component tracked in `states`; binder names
            are how the properties of the binders are loaded, default = None

        Returns
        -------
        Prism
            Instance of a prism matching included specifications
        """
        verticies = la.get_prism_verticies(num_sides, width, height)
        return cls(
            id_, r, t3=t3, t2=t2, vertices=verticies, states=states,
            binder_names=binder_names
        )

    def test_collision(self, vertices: np.ndarray, max_iters: int) -> bool:
        """Test collision with the current nucleosome.

        Parameters
        ----------
        vertices : np.ndarray (M, 3)
            Vertices representing a mesh of the nucleosome bead
        max_iters : int
            Maximum iterations of the GJK algorithm to evaluate when testing
            for collision

        Returns
        -------
        bool
            Flag for collision with the nucleosome (True = collision, False =
            no collision)
        """
        return gjk_collision(self.transform_vertices(), vertices, max_iters)

    def print_properties(self):
        """Print properties of the current nucleosome.
        """
        print("Nucleosome ID: ", self.id)
        print("Central Position: ")
        print(self.r)
        print("t3 Orientation: ")
        print(self.t3)
        print("t2 Orientation: ")
        print(self.t2)

    def transform_vertices(self) -> np.ndarray:
        """Transform the verticies of the nucleosome based on `r`, `t2`, & `t3`.

        Notes
        -----
        Begin by translating the position of the verticies to match the
        position of the nucleosome in space.

        Then rotate the nucleosome so that the x-axis on which the verticies
        are defined aligns with the t3 tangent of the nucleosome.

        Finally, rotate the nucleosome so that the z-axis on which the vertices
        are defined aligns with the t2 tangent of the nucleosome.

        Returns
        -------
        np.ndarray (M, 3)
            Transformed vertices representing a mesh of the nucleosome bead
            positioned and oriented in space.
        """
        num_verticies = len(self.vertices)
        verticies = np.ones((num_verticies, 4))
        verticies[:, 0:3] = self.vertices
        verticies = verticies.T

        translate_mat = np.identity(4)
        for i in range(3):
            translate_mat[i, 3] = self.r[i]

        verticies = translate_mat @ verticies

        x_axis = np.array([1, 0, 0])
        if not np.allclose(x_axis, self.t3):
            rot_mat = get_verticies_rot_mat(x_axis, self.t3, self.r)
            verticies = rot_mat @ verticies

        z_axis = np.array([0, 0, 1])
        if not np.allclose(z_axis, self.t2):
            rot_mat = get_verticies_rot_mat(z_axis, self.t2, self.r)
            verticies = rot_mat @ verticies

        return verticies.T[:, 0:3]


class Nucleosome(Bead):
    """Class representation of a nucleosome bead.

    Notes
    -----
    See documentation for `Bead` class for additional attribute definitions.

    Attributes
    ----------
    bead_length : double
        Spacing between the nucleosome and its neighbor
    rad : double
        Radius of spherical excluded volume around nucleosome
    vol : double
        Volume of spherical excluded volume around nucleosome
    """

    def __init__(
        self, id_: int, r: np.ndarray, *, t3: np.ndarray, t2: np.ndarray,
        bead_length: float, rad: Optional[float] = 5,
        states: Optional[np.ndarray] = None,
        binder_names: Optional[Sequence[str]] = None
    ):
        """Initialize nucleosome object.

        Parameters
        ----------
        id_ : int
            Identifier for the nucleosome
        r : array_like (3,) of double
            Coordinates of the nucleosome in form (x, y, z)
        t3 : array_like (3,) of double
            Tangent vector defining orientation of nucleosome
        t2 : array_like (3,) of double
            Tangent vector orthogonal to `t3` defining orientation of
            nucleosome; a third orthogonal tangent vector can be obtained
            from the cross product of `t2` and `t3`
        bead_length : float
            Spacing between the nucleosome and its neighbor
        rad : Optional[float]
            Radius of spherical excluded volume around nucleosome in simulation
            units of distance; default is 5
        states : Optional array_like (M,) of int
            State of each of the M epigenetic binders being tracked; default is
            None
        binder_names : Optional[Sequence[str]] (M, )
            The name of each bound component tracked in `states`; binder names
            are how the properties of the binders are loaded, default = None
        """
        super(Nucleosome, self).__init__(id_, r, t3, t2, states, binder_names)
        self.bead_length = bead_length
        self.rad = rad
        self.vol = (4/3) * np.pi * rad ** 3

    def test_collision(self, point: np.ndarray) -> bool:
        """Test collision with the nucleosome bead.

        Parameters
        ----------
        point : array_like (3,) of double
            Point at which to test for collision with the nucleosomes

        Returns
        -------
        bool
            Flag for collision with the nucleosome (True = collision, False =
            no collision)
        """
        if self.rad is not None:
            return np.linalg.norm(point - self.r) <= self.rad
        raise ValueError("Nucleosome radius is not specified.")

    def print_properties(self):
        """Print properties of the current nucleosome.
        """
        print("Nucleosome ID: ", self.id)
        print("Radius: ", self.rad)
        print("Position: ")
        print(self.r)
        print("t3 Orientation: ")
        print(self.t3)
        print("t2 Orientation: ")
        print(self.t2)


class DetailedNucleosome(Nucleosome):
    """A nucleosome with fixed entry/exit positions and orientations.

    Notes
    -----
    The entry and exit positions and orientations are defined relative to the
    position and t3/t2 vectors of the nucleosome. We assume at this stage that
    the relative entry and exit positions and orientations are fixed for all
    nucleosomes.

    The nucleosome is defined
    """

    def __init__(
        self, id_: int, r: np.ndarray, *, t3: np.ndarray, t2: np.ndarray,
        bead_length: float, omega_enter: float, omega_exit: float,
        bp_wrap: int, phi: float, rad: Optional[float] = 5,
        states: Optional[np.ndarray] = None,
        binder_names: Optional[Sequence[str]] = None
    ):
        """Initialize detailed nucleosome object.

        Parameters
        ----------
        see documentation for `Nucleosome` class

        omega_enter : float
            Angle of entry of DNA into nucleosome in radians
        omega_exit : float
            Angle of exit of DNA from nucleosome in radians
        bp_wrap : int
            Number of base pairs wrapped around nucleosome
        phi : float
            Tilt angle of DNA entering and exiting nucleosome in radians
        """
        super(DetailedNucleosome, self).__init__(
            id_, r, t3=t3, t2=t2, bead_length=bead_length, rad=rad,
            states=states, binder_names=binder_names
        )
        self.omega_enter = omega_enter
        self.omega_exit = omega_exit
        self.bp_wrap = bp_wrap
        self.phi = phi
        assert phi < np.pi / 2, "Phi must be less than pi/2"

        # Characterize DNA wrapping around nucleosome
        self.length_wrap = bp_wrap * LENGTH_BP
        self.circ = 2 * np.pi * self.rad
        self.n_wrap = self.length_wrap / self.circ
        self.frac_loop, self.full_loop = np.modf(self.n_wrap)
        self.theta_wrap = self.frac_loop * 2 * np.pi

        # Compute configuration in a local coordinate system
        self.get_entry_exit_positions()
        self.get_entry_exit_orientations()

    def get_entry_exit_positions(self):
        """Get entry and exit positions of DNA relative to nucleosome center.

        Notes
        -----
        The entry and exit positions are defined on a local coordinate system
        such that the circular face of the nucleosome is on the x-y plane. The
        t3 orientation is aligned with the +y direction and the t2 orientation
        is aligned with the +z direction. Therefore, the t1 orientation is
        implicitly aligned with the -x direction to maintain a right-handed
        coordinate system. DNA enters on the positive x-axis such that the
        angle of entry is aligned with the t3 orientation.
        """
        self.r_enter = np.array([self.rad, 0, -self.n_wrap * RISE_PER_LAP / 2])
        self.exit_rot_mat = np.array([
            [np.cos(self.theta_wrap), -np.sin(self.theta_wrap), 0],
            [np.sin(self.theta_wrap), np.cos(self.theta_wrap), 0],
            [0, 0, 1]
        ])
        self.r_exit = np.dot(self.exit_rot_mat, self.r_enter)
        self.r_exit = self.r_exit + np.array(
            [0, 0, self.n_wrap * RISE_PER_LAP / 2]
        )
        assert np.isclose(np.linalg.norm(self.r_exit), self.rad), \
            "Exit position is not on the nucleosome surface."

    def get_entry_exit_orientations(self):
        """Get the vector defining the orientation of DNA entering and exiting.

        Notes
        -----
        The entry and exit orientations are defined on a local coordinate
        system such that the circular face of the nucleosome is on the x-y
        plane. The t3 orientation is aligned with the +y direction and the t2
        orientation is aligned with the +z direction. Therefore, the t1
        orientation is implicitly aligned with the -x direction to maintain a
        right-handed coordinate system. DNA enters on the positive x-axis such
        that the entry vector is parallel to the t3 orientation when omega_entry
        is zero. The exit vector is dictated by the amount of wrapping around
        the nucleosome and omega_exit. In addition to the entry and exit angles,
        the tilt angle of the DNA entering and exiting the nucleosome is
        specified by phi and affects the z-component of the entry and exit
        vectors.

        As the DNA wraps around the nucleosome, the natural twist of DNA
        affects its orientation. The t2-orientation of the DNA strand rotates
        around the t3-vector based on the natural twist of DNA.

        TODO: Check implementation of entry and exit perp vectors.
        """
        # Generate the entry vector (t3-orientation at entry)
        self.entry_vec = np.array([
            -np.sin(self.omega_enter), np.cos(self.omega_exit), 0.
        ]) + np.array([0., 0., np.sin(self.phi)])
        self.entry_vec = self.entry_vec / np.linalg.norm(self.entry_vec)

        # Generate the exit vector (t3-orientation at exit)
        self.exit_vec = np.array([
            np.sin(self.omega_exit), np.cos(self.omega_exit), 0.
        ]) + np.array([0., 0., np.sin(self.phi)])
        self.exit_vec = self.exit_vec / np.linalg.norm(self.exit_vec)
        self.exit_vec = np.dot(self.exit_rot_mat, self.exit_vec)

        # Generate the perp vectors (t2-orientations at entry and exit)
        self.entry_perp_vec = np.cross(self.entry_vec, np.array([1, 0, 0]))
        self.entry_perp_vec /= np.linalg.norm(self.entry_perp_vec)
        self.exit_perp_vec = np.cross(self.exit_vec, np.array([1, 0, 0]))
        self.exit_perp_vec /= np.linalg.norm(self.exit_perp_vec)

        # The t2-orientation of the exiting DNA strand must take into account
        # the natural twist of DNA. We rotate the t2-orientation of the exiting
        # DNA strand about the t3-vector by the amount of natural twist.
        # The rotated perpendicular exit vector is computed by Rodrigues'
        # rotation formula.
        angle_twist = self.bp_wrap * NATURAL_TWIST_NUCLEOSOME
        angle_twist = angle_twist - 2 * np.pi * (angle_twist // (2 * np.pi))
        self.exit_perp_vec = (
            self.exit_perp_vec * np.cos(angle_twist) +
            np.cross(self.exit_vec, self.exit_perp_vec) * np.sin(angle_twist) +
            self.exit_vec * np.dot(self.exit_vec, self.exit_perp_vec) *
            (1 - np.cos(angle_twist))
        )

    def align_with_global_frame(self):
        """Align nucleosome with global frame.

        Notes
        -----
        The nucleosome is aligned to a local frame such that the circular face
        of the nucleosome is on the x-y plane. The t3 orientation is aligned
        with the +y direction and the t2 orientation is aligned with the +z
        direction. Therefore, the t1 orientation is implicitly aligned with the
        -x direction to maintain a right-handed coordinate system. This method
        rotates the nucleosome to so that the t3 axis of the nucleosome is
        properly aligned with the t3 orientation of the associated polymer
        segment in the global reference frame.

        Returns
        -------
        r_enter_global : np.ndarray (3,) of float
            Entry position of DNA defined in global reference frame
        r_exit_global : np.ndarray (3,) of float
            Exit position of DNA defined in global reference frame
        entry_vec_global : np.ndarray (3,) of float
            Entry vector of DNA defined in global reference frame
        exit_vec_global : np.ndarray (3,) of float
            Exit vector of DNA defined in global reference frame
        entry_perp_vec_global = np.ndarray (3,) of float
            Perpendicular vector to entry vector of DNA defined in global
            reference frame
        exit_perp_vec_global = np.ndarray (3,) of float
            Perpendicular vector to exit vector of DNA defined in global
            reference frame
        """
        R_global_to_local = la.get_rotation_matrix(self.t3, self.t2)
        R_local_to_global = np.linalg.inv(R_global_to_local)
        r_enter_global = np.dot(R_local_to_global, self.r_enter) + self.r
        r_exit_global = np.dot(R_local_to_global, self.r_exit) + self.r
        entry_vec_global = np.dot(R_local_to_global, self.entry_vec)
        exit_vec_global = np.dot(R_local_to_global, self.exit_vec)
        entry_perp_vec_global = np.dot(R_local_to_global, self.entry_perp_vec)
        exit_perp_vec_global = np.dot(R_local_to_global, self.exit_perp_vec)
        return r_enter_global, r_exit_global, entry_vec_global, \
            exit_vec_global, entry_perp_vec_global, exit_perp_vec_global

    def update_configuration(self, r, t3, t2):
        """Update the position and orientations of the nucleosome.

        Parameters
        ----------
        r : np.ndarray (3,) of float
            Position of nucleosome in global reference frame
        t3, t2 : np.ndarray (3,) of float
            Orthogonal unit vectors defining the orientation of the nucleosome
            in the global reference frame.

        Returns
        -------
        r_enter_global : np.ndarray (3,) of float
            Entry position of DNA defined in global reference frame
        r_exit_global : np.ndarray (3,) of float
            Exit position of DNA defined in global reference frame
        entry_vec_global : np.ndarray (3,) of float
            Entry vector of DNA defined in global reference frame
        exit_vec_global : np.ndarray (3,) of float
            Exit vector of DNA defined in global reference frame
        """
        self.r = r
        self.t3 = t3
        self.t2 = t2
        vals = self.align_with_global_frame()
        r_entry_global = vals[0]
        r_exit_global = vals[1]
        entry_vec_global = vals[2]
        exit_vec_global = vals[3]
        entry_perp_vec_global = vals[4]
        exit_perp_vec_global = vals[5]
        return r_entry_global, r_exit_global, entry_vec_global, \
            exit_vec_global, entry_perp_vec_global, exit_perp_vec_global
