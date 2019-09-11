import numpy as np
from eolearn.core import EOTask, FeatureType
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
from functools import reduce

'''
org_shape = (3, 3)
mask = np.array([[1, 0, 1],
                 [1, 1, 1],
                 [1, 1, 1]])

org_shape = (2, 2)
mask = np.array([[1, 0],
                 [1, 1]])

# graph_args = {'n_x': org_shape[0], 'n_y': org_shape[1]}
temp_mask = grid_to_graph(3, 3, mask=mask, return_as=np.ndarray)
temp = grid_to_graph(3, 3, return_as=np.ndarray)

print(temp)
print(temp_mask)
a = np.where(np.ravel(mask) == 0)
print(a)


def apply_mask(acc, i):
    acc[i, :] = 0
    acc[:, i] = 0
    return acc
temp = reduce(apply_mask, a, temp)

for x in a:
    temp[x, :] = 0
    temp[:, x] = 0
'''

data_long = [0, 1, 2, 3, 4, 5, 6, 7, 8]
data_long = data_long[::-1]
print(data_long)
mask_long = [1, 0, 0, 1, 0, 1, 1, 0, 1]

locations = [i for i, elem in enumerate(mask_long) if elem == 0]
print(locations)
tracker = range(len(data_long))
data_cleared = np.delete(data_long, locations)
former = np.delete(tracker, locations)
new_data = [None] * len(data_long)
for f, val in zip(former, data_cleared):
    new_data[f] = val

# new_data = np.insert(data_cleared, locations, np.zeros(5))

print(new_data)
