from io import StringIO

import numpy as np
import pandas as pd
import pytest

from chromo.polymers import Chromatin
from chromo.fields import UniformDensityField
import chromo.binders


def test_uniform_density_field_roundtrip():
    binders = [chromo.binders.get_by_name('HP1')]
    binders = chromo.binders.make_binder_collection(binders)

    p = Chromatin.straight_line_in_x(
        'Chr-1', 10, 1, states=np.zeros((10, 1), dtype=int),
        binder_names=np.array(['HP1']),
        chemical_mods=np.zeros((10, 1), dtype=int),
        chemical_mod_names=np.array(['H3K9me3'])
    )

    udf = UniformDensityField([p], binders, 10, 20, 30, 40, 50, 60)
    udf_round_trip = UniformDensityField.from_file(
        StringIO(udf.to_file(None)), [p], binders
    )

    assert udf == udf_round_trip

    # ensure the code errors if the wrong number of polymers or binders is passed
    with pytest.raises(ValueError):
        UniformDensityField.from_file(
            StringIO(udf.to_file(None)), [], binders
        )
    with pytest.raises(ValueError):
        UniformDensityField.from_file(
            StringIO(udf.to_file(None)), [p, p], binders
        )
    with pytest.raises(ValueError):
        UniformDensityField.from_file(
            StringIO(udf.to_file(None)), [p], pd.concat([binders, binders])
        )

    # should also error if the poly/binder has the wrong name
    with pytest.raises(ValueError):
        p2 = Chromatin.straight_line_in_x(
            'Chr-1', 10, 1, states=np.zeros((10,)),
            binder_names=np.array(['HP1'])
        )
        p2.name = "different"
        UniformDensityField.from_file(
            StringIO(udf.to_file(None)), [p2], binders
        )
    with pytest.raises(ValueError):
        binders2 = chromo.binders.make_binder_collection(binders)
        binders2.loc[0, 'name'] = 'different'
        UniformDensityField.from_file(
            StringIO(udf.to_file(None)), [p], binders2
        )
