#!/bin/bash
# Run simulations varying the chemical potential.
# Second batch of chemical potentials

python plot_cmap_avg_batch_BPS.py 3 4 17 19 25 &
python plot_cmap_avg_batch_BPS.py 2 8 20 21 23 &
python plot_cmap_avg_batch_BPS.py 5 9 12 13 24 &
python plot_cmap_avg_batch_BPS.py 10 11 14 15 22 &
python plot_cmap_avg_batch_BPS.py 6 7 16 18 26 &
