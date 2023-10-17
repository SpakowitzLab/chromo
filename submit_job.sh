#!/bin/bash
#SBATCH --job-name=chromo
#SBATCH --time=47:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
source ~/.bashrc
conda activate chromo
cd /scratch/users/ahirsch1/chromo_scratch/chromo/chromo
python run_simulation.py 100 0.34 0.34 50 100 1000 40000
echo 'Job complete'
# Number of beads
# First bead spacing value
# Second bead spacing value
# lp
# lt
# Number of snapshots
# MC steps per snapshot
