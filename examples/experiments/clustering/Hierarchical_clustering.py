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
import time


# spectral angle mapper
def sam(a, b):  # just cosine similarity???
    prod = np.dot(a, b)
    return prod / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))


class Clustering(EOTask):

    def __init__(self, data_name, output_name, distance_threshold=None, n_clusters=None, metric=sam, affinity="cosine",
                 linkage="single", remove_small=0):
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

        self.remove_small = remove_small

    def execute(self, eopatch):
        data = eopatch.data_timeless[self.data_name]

        org_shape = data.shape
        data_long = np.reshape(data, (-1, org_shape[-1]))
        st_time = time.time()
        model = AgglomerativeClustering(distance_threshold=self.distance_threshold, affinity=self.affinity,
                                        linkage=self.linkage,
                                        connectivity=grid_to_graph(org_shape[0], org_shape[1]),
                                        n_clusters=self.n_clusters,
                                        compute_full_tree=self.compute_full_tree)

        # perform clustering
        model.fit(data_long)

        print(self.data_name + ': ', time.time() - st_time)

        # non_leaves = model.children_[model.n_leaves:]
        # print(non_leaves)
        # labels = agglomerate.fit_predict(data_long)
        labels = np.zeros(model.n_clusters_)
        trimmed_labels = model.labels_

        for i in model.labels_:
            labels[i] += 1

        for i, no_lab in enumerate(labels):
            if no_lab < self.remove_small:
                trimmed_labels[trimmed_labels == i] = 0

        packed_labels = np.reshape(trimmed_labels, org_shape[:-1])


        eopatch.add_feature(FeatureType.DATA_TIMELESS, self.output_name, packed_labels[..., np.newaxis])

        return eopatch


# clustering = Clustering("FILIP_FEATURES", "CLUSTERS", n_clusters=30000) #0.000000015
# clustering = Clustering("FILIP_FEATURES", "CLUSTERS", distance_threshold=5*10**-9)
# clustering = Clustering("FILIP_ALL", "CLUSTERS1", distance_threshold=5*10**-20)

clust_no = 30000
# d = 3*10**-8
d = None
clustering = Clustering("FILIP_FEATURES", "CLUSTERS", remove_small=10, n_clusters=None, linkage='average',
                        distance_threshold=5 * 10 ** -8)
# clustering1 = Clustering("FILIP_ALL", "CLUSTERS1", n_clusters=clust_no, linkage='average', distance_threshold=d)

patch_location = './eopatch/'
load = LoadFromDisk(patch_location)

save_path_location = './eopatch/'
if not os.path.isdir(save_path_location):
    os.makedirs(save_path_location)

save = SaveToDisk(save_path_location, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

extra_param = {
    load: {'eopatch_folder': 'patch08_LULC'},
    save: {'eopatch_folder': 'patch09_clusters'}
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
    # clustering1,
    save
)

workflow.execute(extra_param)
display()
