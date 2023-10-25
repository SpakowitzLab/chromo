#!/bin/bash
#SBATCH --job-name=chromo
#SBATCH --time=47:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
source ~/.bashrc
conda activate chromo
cd /scratch/users/ahirsch1/chromo_scratch/chromo/chromo
num_beads = 100
bead_spacing_1 = 0.34
bead_spacing_2 = 0.34
lp = 50 
lt = 100
num_snapshots = 1000
mc_per_snap = 40000
python run_simulation.py $num_beads $bead_spacing_1 $bead_spacing_2 $lp $lt $num_snapshots $mc_per_snap
echo 'Number of beads'
echo $num_beads
echo 'First bead spacing'
echo $bead_spacing_1
echo 'Second bead spacing'
echo $bead_spacing_2
echo 'lp'
echo $lp
echo 'lt'
echo $lt
echo 'Number of snapshots'
echo $num_snapshots
echo 'MC steps per snapshot'
echo $mc_per_snap
echo 'Job complete'
# Number of beads
# First bead spacing value
# Second bead spacing value
# lp
# lt
# Number of snapshots
# MC steps per snapshot
