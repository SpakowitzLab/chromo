"""
Modifications to the polymer's chemical state that affect the energy.

Each simulation type will typically require a particular type of Mark, for
which there will be a class defined here. In addition, for common, physically
derived marks, such as known epigenetic modifications to DNA or
well-characterized modifications to other real polymers, the best-known
parameters for that mark should be documented here, as instances of the
appropriate `.Mark` subclass.
"""

from dataclasses import dataclass
import inspect
import sys

@dataclass
class Mark:
    """
    In order for our code to work, all marks need a string name.
    """
    name: str

@dataclass
class Epigenmark(Mark):
    """Information about the chemical properties of an epigenetic mark."""
    bind_energy: float
    interaction_energy: float
    chemical_potential: float

hp1 = Epigenmark('HP1', bind_energy=1, interaction_energy=1, chemical_potential=1)
"""
Heterochromatin Protein 1, binds H3K9me marks.

For now, we just use filler values for the energies. In the future, this
documentation string should go through the process of explaining exactly how we
arrive at the values that we actually use.
"""

def get_by_name(name):
    all_marks = [obj for name, obj in inspect.getmembers(sys.modules[__name__])
                 if isinstance(obj, Mark)]
    matching_marks = [mark for mark in all_marks if mark.name == name]
    if not matching_marks:
        raise ValueError(f"No marks found in {__name__} with name: {name}")
    if len(matching_marks) > 1:
        raise ValueError(f"More than one mark has the name requested: {name}")
    return matching_marks[0]
