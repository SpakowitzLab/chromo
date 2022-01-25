"""Modifications to the polymer's chemical state that affect the energy.

Notes
-----
Each simulation type will typically require a particular type of Mark, for
which there will be a class defined here. In addition, for common, physically
derived marks, such as known epigenetic modifications to DNA or
well-characterized modifications to other real polymers, the best-known
parameters for that mark should be documented here, as instances of the
appropriate `.Mark` subclass.
"""
import inspect
import sys

import pandas as pd
import numpy as np


cdef class Mark:
    """Class representation of an arbitrary mark binding to a polymer.

    Notes
    -----
    For this code to work, all marks need a string name.

    Attributes
    ----------
    name : str
        Name of the chemical mark
    sites_per_bead : int
        Number of sites to which the chemical mark can bind on each monomeric
        unit of the polymer
    """

    def __init__(self, name: str, sites_per_bead: int) -> None:
        """Initialize the mark object.

        Parameters
        ----------
        name : str
            Name of the mark binding the polymer
        sites_per_bead : int
            Quantity of the mark that can be bound to a single polymer bead
        """
        self.name = name
        self.sites_per_bead = sites_per_bead
        self.binding_seq = np.zeros((sites_per_bead,), dtype=int)


cdef class Epigenmark(Mark):
    """Information about the chemical properties of an epigenetic mark.
    """

    def __init__(
        self,
        name: str,
        sites_per_bead: int,
        bind_energy_mod: float,
        bind_energy_no_mod: float,
        interaction_energy: float,
        chemical_potential: float,
        interaction_radius: float
    ) -> None:
        """Initialize the epigenetic mark object with physical properties.

        Parameters
        ----------
        name : str
            Name of the epigenetic mark binding the polymer
        sites_per_bead : int
            Quantity of the epigenetic mark that can bind a single polymer bead
        bind_energy_mod : float
            Binding energy of the epigenetic mark to a bead with the associated
            chemical modification
        bind_energy_no_mod : float
            Binding energy of the epigenetic mark to a bead without the
            associated chemical modification
        interaction_energy : float
            Interaction energy contributed by each pair of the epigenetic mark
            within a specified interaction volume of one-another -- e.g., can
            represent oligomerization between epigenetic marks
        chemical_potential : float
            Chemical potential of the unbound epigenetic mark
        interaction_radius : float
            Cutoff distance between pairs of marks for which the interaction
            energy will be acquired
        """
        super().__init__(name, sites_per_bead)
        self.bind_energy_mod = bind_energy_mod
        self.bind_energy_no_mod = bind_energy_no_mod
        self.interaction_energy = interaction_energy
        self.chemical_potential = chemical_potential
        self.interaction_radius = interaction_radius
        self.interaction_volume = (4.0/3.0) * np.pi * interaction_radius ** 3
        self.field_energy_prefactor = 0.0
        self.interaction_energy_intranucleosome = 0.0
    
    def dict(self):
        """Represent a class instance as a dictionary (exclude derived values).

        Returns
        -------
        dict
            Dictionary with key attributes representing the marks
        """
        return {
            "name": self.name,
            "sites_per_bead": self.sites_per_bead,
            "bind_energy_mod": self.bind_energy_mod,
            "bind_energy_no_mod": self.bind_energy_no_mod,
            "interaction_energy": self.interaction_energy,
            "chemical_potential": self.chemical_potential,
            "interaction_radius": self.interaction_radius,
            "interaction_volume": self.interaction_volume,
            "field_energy_prefactor": self.field_energy_prefactor,
            "interaction_energy_intranucleosome":\
                self.interaction_energy_intranucleosome
        }


null_mark = Epigenmark(
    'null_mark', sites_per_bead=0, bind_energy_mod=0, bind_energy_no_mod=0,
    interaction_energy=0, chemical_potential=0, interaction_radius=0
)
"""Placeholder mark.

Notes
-----
Right now, the simulator requires at least one mark to be defined; this serves
as a placeholder.
"""


hp1 = Epigenmark(
    'HP1', sites_per_bead=2, bind_energy_mod=-0.01, bind_energy_no_mod=1.52,
    interaction_energy=-4, chemical_potential=-1, interaction_radius=3
)
"""Heterochromatin Protein 1, binds H3K9me marks.

Notes
-----
For now, we just use filler values for the energies. In the future, this
documentation string should go through the process of explaining exactly how we
arrive at the values that we actually use.
The filler values for interaction energy and chemical potential are both 1.
"""

prc1 = Epigenmark(
    'PRC1', sites_per_bead=2, bind_energy_mod=-0.01, bind_energy_no_mod=1.52,
    interaction_energy=-4, chemical_potential=-1, interaction_radius=3
)
"""Protein Regulator of Cytokinesis 1, binds H3K27me marks.

Notes
-----
For now, we just use filler values for the energies. In the future, this
documentation string should go through the process of explaining exactly how we
arrive at the values that we actually use.
The filler values for interaction energy and chemical potential are both 1.
"""


def get_by_name(name):
    """Look up saved mark by name.

    Parameters
    ----------
    name : str
        Name of the pre-defined mark to retrieve

    Returns
    -------
    Mark
        Object representing the mark that was queried.
    """
    all_marks = [
        obj for name_, obj in inspect.getmembers(sys.modules[__name__])
        if isinstance(obj, Mark)
    ]
    matching_marks = [mark for mark in all_marks if mark.name == name]
    if not matching_marks:
        raise ValueError(f"No marks found in {__name__} with name: {name}")
    if len(matching_marks) > 1:
        raise ValueError(f"More than one mark has the name requested: {name}")
    return matching_marks[0]


def make_mark_collection(marks):
    """Construct summary DataFrame from sequence of marks.

    Parameters
    ----------
    marks : Mark or str or Sequence[Mark] or Sequence[str]
        The marks to be summarized by the DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns are the properties of each Mark.
    """
    df = pd.DataFrame(
        columns=[
            'name', 'sites_per_bead', 'bind_energy',
            'interaction_energy', 'chemical_potential'
        ]
    )
    if marks is None:
        return None
    input_type = type(marks)
    if input_type is str or issubclass(input_type, Mark):
        marks = [marks]  # allow the "one mark" case
    for mark in marks:
        if type(mark) is str:
            mark = get_by_name(mark)
        df = df.append(mark.dict(), ignore_index=True)
    return df
