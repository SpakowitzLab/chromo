from pathlib import Path
import numpy as np


_util_folder = Path(__file__).parent.absolute()
dss_params = np.loadtxt(_util_folder / Path("dssWLCparams"))
