# https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
# Metrika SAM (Spectral angle mapper) boljši za EO kot euclidean - mitigate effects of variable illumination Yan, Roy 2015 dimensionality reduction

import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import numpy as np
from eolearn.core import EOTask, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
import os
from Display_clusters import display


# spectral angle mapper
def sam(a, b):  # just cosine similarity???
    prod = np.dot(a, b)
    return prod / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))


class Clustering(EOTask):

    def __init__(self, data_name, output_name, distance_threshold=None, n_clusters=None, metric=sam, affinity="cosine",
                 linkage="single"):
        self.distance_threshold = distance_threshold
        self.metric = metric
        self.affinity = affinity
        self.linkage = linkage
        self.data_name = data_name
        self.output_name = output_name
        self.n_clusters = n_clusters
        self.compute_full_tree = 'auto'
        if distance_threshold is not None:
            self.compute_full_tree = True

    def execute(self, eopatch):
        data = eopatch.data_timeless[self.data_name]

        org_shape = data.shape
        model = AgglomerativeClustering(distance_threshold=self.distance_threshold, affinity=self.affinity,
                                        linkage=self.linkage,
                                        connectivity=grid_to_graph(org_shape[0], org_shape[1]),
                                        n_clusters=self.n_clusters,
                                        compute_full_tree=self.compute_full_tree)

        data_long = np.reshape(data, (-1, org_shape[-1]))

        # Optimization, left for later
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.inner.html#numpy.inner
        # https://docs.scipy.org/doc/numpy/reference/arrays.classes.html#arrays-classes
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.outer.html#numpy.ufunc.outer
        # https://docs.scipy.org/doc/numpy/reference/ufuncs.html

        # compute distances
        # length = len(data_long)
        # print(length)
        '''
        distances = np.zeros((length, length))

        for x in range(length):
            for y in range(length):
                distances[x][y] = self.metric(data_long[x], data_long[y])
        '''
        # perform clustering

        model.fit(data_long)
        # labels = agglomerate.fit_predict(data_long)

        packed_labels = np.reshape(model.labels_, org_shape[:-1])

        eopatch.add_feature(FeatureType.DATA_TIMELESS, self.output_name, packed_labels[..., np.newaxis])

        return eopatch


#clustering = Clustering("FILIP_FEATURES", "CLUSTERS", n_clusters=30000) #0.000000015
clustering = Clustering("FILIP_FEATURES", "CLUSTERS", distance_threshold=5*10**-9)

patch_location = './eopatch/'
load = LoadFromDisk(patch_location)

save_path_location = './eopatch/'
if not os.path.isdir(save_path_location):
    os.makedirs(save_path_location)

save = SaveToDisk(save_path_location, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

extra_param = {
    load: {'eopatch_folder': 'patch4'},
    save: {'eopatch_folder': 'patch5'}
}
'''
workflow = LinearWorkflow(
    load,
    AddFeatures(),
    save)
'''
workflow = LinearWorkflow(
    load,
    clustering,
    save
)

workflow.execute(extra_param)
display()
