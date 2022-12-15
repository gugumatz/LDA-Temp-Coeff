import sys
import math
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree

# Read peak list 1
peak_list_1 = pd.read_excel(sys.argv[1])
SSN1 = peak_list_1.iloc[:len(peak_list_1), 0]
PKL1 = peak_list_1.copy()   # Make copy to work with
PKL1 = PKL1.drop(['SSN'], axis=1)
header1 = PKL1.columns.tolist()

# Read peak list 2
peak_list_2 = pd.read_excel(sys.argv[2])
SSN2 = peak_list_2.iloc[:len(peak_list_2), 0]
PKL2 = peak_list_2.copy()   # Make copy to work with
PKL2 = PKL2.drop(['SSN'], axis=1)
header2 = PKL2.columns.tolist()

# Remove non-common Chemical Shifts from lists
header = list(set(header1).intersection(header2))
not_CH1 = [x for x in header1 if x not in header2]
not_CH2 = [x for x in header2 if x not in header1]
PKL1 = PKL1.drop(not_CH1, axis=1)
PKL2 = PKL2.drop(not_CH2, axis=1)
PKL1 = PKL1.reindex(sorted(PKL1.columns), axis=1)
PKL2 = PKL2.reindex(sorted(PKL2.columns), axis=1)

# Scale CH values to range 0-1
dfs = [PKL1, PKL2]
full_df = pd.concat(dfs)
min_vals = full_df.min()
diff_vals = full_df.max() - min_vals

for k in PKL1:
    PKL1[k] = (PKL1[k] - min_vals[k])/diff_vals[k]
data1 = PKL1.to_numpy()

for k in PKL2:
    PKL2[k] = (PKL2[k] - min_vals[k])/diff_vals[k]
data2 = PKL2.to_numpy()

# Build the trees
tree1 = KDTree(data1, leaf_size=40, metric='euclidean')
tree2 = KDTree(data2, leaf_size=40, metric='euclidean')

# Classify closest neighbour to peak list 1 from peak list 2
closest12 = []
for i, k in enumerate(data1):
    dist, ind = tree2.query(k.reshape(1, -1), k=2)
    if dist[0, 1] > 2*dist[0, 0] and dist[0, 0] < 0.01:
        closest12.append([i, ind[0, 0]])

# Classify closest neighbour to peak list 2 from peak list 1
closest21 = []
for i, k in enumerate(data2):
    dist, ind = tree1.query(k.reshape(1, -1), k=2)
    if dist[0, 1] > 2*dist[0, 0] and dist[0, 0] < 0.01:
        closest21.append(ind[0, 0])
closest12 = pd.DataFrame(closest12, columns=['SSN peak list 1', 'SSN peak list 2'])

# Find coincidences of closest neighbours
list1 = np.zeros(len(closest12), dtype=bool)
for i, k in enumerate(closest12.iloc[:, 0]):
    if k in closest21:
        list1[i] = not list1[i]
pairs = closest12.iloc[list1, :]

# Save to files
pairs.to_excel('Pairs.xlsx', index=False)

closest_list1 = peak_list_1.iloc[pairs['SSN peak list 1'], :]
closest_list1.to_excel('closest_list1.xlsx', index=False)
remaining1 = peak_list_1.drop(pairs['SSN peak list 1'])
remaining1.to_excel('remaining_list1.xlsx', index=False)

closest_list2 = peak_list_2.iloc[pairs['SSN peak list 2'], :]
closest_list2.to_excel('closest_list2.xlsx', index=False)
remaining2 = peak_list_2.drop(pairs['SSN peak list 2'])
remaining2.to_excel('remaining_list2.xlsx', index=False)


'''
list2 = np.ones(len(data2))
aux2 = []
for k, i in enumerate(aux23):
    if i[0] not in aux12:
        aux23.pop(k)
        list2[i[0]] = 0

df1 = pd.DataFrame(aux23)
df1.to_excel('Paired.xlsx')

# Save to files
closest_neighbour1 = data1[list1 == 1, :]
out = pd.DataFrame(closest_neighbour1)
for k in out:
    out[k] = (out[k]*diff_vals[k])+min_vals[k]
out.to_excel('closest1.xlsx')
remaining1 = data1[list1 == 0, :]
out = pd.DataFrame(remaining1)
for k in out:
    out[k] = (out[k]*diff_vals[k])+min_vals[k]
out.to_excel('remaining1.xlsx')

closest_neighbour2 = data2[list2 == 1, :]

out = pd.DataFrame(closest_neighbour2)
for k in out:
    out[k] = (out[k]*diff_vals[k])+min_vals[k]
out.to_excel('closest2.xlsx')
remaining2 = data2[list2 == 0, :]
out = pd.DataFrame(remaining2)
for k in out:
    out[k] = (out[k]*diff_vals[k])+min_vals[k]
out.to_excel('remaining2.xlsx')


distances = []
for i, k in enumerate(closest_neighbour2):
    dist, ind = tree1.query(k.reshape(1, -1), k=2)
    if dist[0, 1] > 2*dist[0, 0] and dist[0, 0] < 0.02:
        distances.append(dist[0, 0])
dist_matrix = np.array(distances)
dist_mean = np.mean(dist_matrix)

# Classify unrelated residues
unrelated = np.zeros(len(data2))
for i, k in enumerate(data2):
    dist, ind = tree1.query(k.reshape(1, -1), k=1)
    if dist[0, 0] > 0.05:
        unrelated[i] = 1

lose_residues = data2[unrelated == 1, :]
out = pd.DataFrame(lose_residues)
for k in out:
    out[k] = (out[k]*diff_vals[k])+min_vals[k]
out.to_excel('lose.xlsx')
'''
