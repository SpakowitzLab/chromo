import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def vector_magnitude(vector):
    return np.linalg.norm(vector)


def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude_product = vector_magnitude(vector1) * vector_magnitude(vector2)
    return dot_product / magnitude_product


def t1function(df, initial_comparison_location):
    end_point_1 = df["t1"][initial_comparison_location]
    start_point_r1 = np.array(df.iloc[initial_comparison_location, 1:4])

    correlations_t1 = []
    distance = []

    for i in range(len(df)):
        end_point_2 = df["t1"][i]
        start_point_r2 = np.array(df.iloc[i, 1:4])

        vector1 = end_point_1 - start_point_r1
        vector2 = end_point_2 - start_point_r2

        correlation = cosine_similarity(vector1, vector2)
        correlations_t1.append(correlation)

        distance.append(abs(initial_comparison_location - i))
    t1_array = [distance, correlations_t1]
    return np.array(t1_array)


def t3function(df, initial_comparison_location):
    end_point_1 = np.array(df.iloc[initial_comparison_location, 4:7])  # Extract columns 4, 5, and 6
    start_point_r1 = np.array(df.iloc[initial_comparison_location, 1:4])  # Extract columns 1, 2, and 3

    correlations_t3 = []
    distance = []

    for i in range(len(df)):
        end_point_2 = np.array(df.iloc[i, 4:7])  # Extract columns 4, 5, and 6 for the second end point
        start_point_r2 = np.array(df.iloc[i, 1:4])  # Extract columns 1, 2, and 3 for the second start point

        vector1 = end_point_1 - start_point_r1
        vector2 = end_point_2 - start_point_r2

        correlation = cosine_similarity(vector1, vector2)
        correlations_t3.append(correlation)

        distance.append(abs(initial_comparison_location - i))

    t3_array = [distance, correlations_t3]
    return np.array(t3_array)



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
    snap_range2 = 178

t1_final_list = []
t3_final_list = []
print(simulation_number)
for snap in range(snap_range1, snap_range2):
    print(snap)
    snap_path = f"{file_path}sim_{simulation_number}/poly_1-{snap}.csv"
    df = pd.read_csv(snap_path)

    t2coordinates = df[["('t2', 'x')", "('t2', 'y')", "('t2', 'z')"]].values
    t3coordinates = df[["('t3', 'x')", "('t3', 'y')", "('t3', 'z')"]].values

    t1coordinates = np.cross(t2coordinates, t3coordinates)
    df['t1'] = [t1coordinates[i] for i in range(len(t1coordinates))]

    t1_final_list.extend([t1function(df, initial_comparison) for initial_comparison in range(len(df))])
    t3_final_list.extend([t3function(df, initial_comparison) for initial_comparison in range(len(df))])

t1_final_array = np.array(t1_final_list)
distance_1 = t1_final_array[:, 0].flatten()
correlation_1 = t1_final_array[:, 1].flatten()

t3_final_array = np.array(t3_final_list)
distance_3 = t3_final_array[:, 0].flatten()
correlation_3 = t3_final_array[:, 1].flatten()

np.savetxt(f"{file_path}sim_{simulation_number}/{simulation_number}correlation1.txt", correlation_1)
np.savetxt(f"{file_path}sim_{simulation_number}/{simulation_number}correlation3.txt", correlation_3)
np.savetxt(f"{file_path}sim_{simulation_number}/{simulation_number}distance1.txt", distance_1)
np.savetxt(f"{file_path}sim_{simulation_number}/{simulation_number}distance3.txt", distance_3)

plt.plot(distance_1, correlation_1)
plt.xlabel('nm distance')
plt.ylabel('T1 Correlation')
plt.title('T1 Correlations vs Distances')
plt.grid(True)
plt.savefig(f"{file_path}sim_{simulation_number}/{simulation_number}_T1_tangent_correlations.png")
#plt.show()

plt.plot(distance_3, correlation_3)
plt.xlabel('nm distance')
plt.ylabel('T3 Correlation')
plt.title('T3 Correlations vs Distances')
plt.grid(True)
plt.savefig(f"{file_path}sim_{simulation_number}/{simulation_number}_T3_tangent_correlations.png")
#plt.show()
