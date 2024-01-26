from io import StringIO

from pandas.testing import assert_frame_equal
import numpy as np

from chromo.polymers import Chromatin


def test_polymer_saving():
    p = Chromatin.straight_line_in_x(
        'Chr-1', np.array([1.] * 9), states=np.zeros((10, 1), dtype=int),
        binder_names=np.array(['HP1']),
        chemical_mods=np.zeros((10, 1), dtype=int),
        chemical_mod_names=np.array(['H3K9me3'])
    )
    df = p.from_file(StringIO(p.to_file(None)), 'Chr-1').to_dataframe()
    df_pre_round_trip = p.to_dataframe()
    assert_frame_equal(df, df_pre_round_trip)
