import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt

sherlock = False

if sherlock:
    file_path = "/scratch/users/ahirsch1/chromo_scratch/output/"
    simulation_number = int(sys.argv[1])
    snap_range1 = int(sys.argv[2])
    snap_range1 = int(sys.argv[3])
else:
    file_path = "/Users/angelikahirsch/Documents/chromo/output/"
    simulation_number = 218
    snap_range1 = 177
    snap_range2 = 179

euclidean_distances = []
for snap in range(snap_range1, snap_range2):
    print(snap)
    snap_path = f"{file_path}sim_{simulation_number}/poly_1-{snap}.csv"
    df = pd.read_csv(snap_path)
    coordinates = df[["('r', 'x')", "('r', 'y')", "('r', 'z')"]].values
    num_beads = len(df)

    for i in range(0, num_beads - 2):
        difference = coordinates[i] - coordinates[i + 2]
        distance = np.linalg.norm(difference)
        euclidean_distances.append(distance/0.34)


np.savetxt(f"{file_path}sim_{simulation_number}/{simulation_number}1,3bead_distances.txt", euclidean_distances)

plt.hist(euclidean_distances, bins=20, color='blue', alpha=0.7)
plt.xlabel('Distance in bp')
plt.ylabel('Frequency')
plt.title('Histogram of Distances between Beads Two Spaces Apart')
plt.grid(True)
plt.savefig(f"{file_path}sim_{simulation_number}/{simulation_number}1,3bead_distances.png")
#plt.show()

