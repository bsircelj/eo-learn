"""
Module for computing clusters in EOPatch
"""

import numpy as np
from eolearn.core import EOTask, FeatureType
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
from functools import reduce


class Clustering(EOTask):
    """
    Tasks computes agglomerative clustering using selected features.

    The algorithm produces a timeless data feature where each cell has a number which corresponds to specific group.
    :param features: A collection of features used for clustering
    :type features: dict(FeatureType: set(str))
    :param new_feature: Name and type of feature that is the result of clustering
    :type new_feature: (FeatureType, str)
    :param distance_threshold: The linkage distance threshold above which, clusters will not be merged. If non None,
        n_clusters must be None nd compute_full_tree must be True
    :type distance_threshold: float or None
    :param n_clusters: The number of clusters found by the algorithm. If distance_threshold=None, it will be equal to
        the given n_clusters.
    :type n_clusters: int or None
    :param affinity: Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”.
    :type affinity: string
    :param linkage: Which linkage criterion to use. The linkage criterion determines which distance to use between sets
        of observation. The algorithm will merge the pairs of cluster that minimize this criterion.
        - ward minimizes the variance of the clusters being merged.
        - average uses the average of the distances of each observation of the two sets.
        - complete or maximum linkage uses the maximum distances between all observations of the two sets.
        - single uses the minimum of the distances between all observations of the two sets.
    :type linkage: {“ward”, “complete”, “average”, “single”}
    :param remove_small: If greater than 0, removes all clusters that have less points as "remove_small"
    :type remove_small: int
    :param connectivity: Connectivity matrix. Defines for each sample the neighboring samples following a given
        structure of the data. This can be a connectivity matrix itself or a callable that transforms the data into a
        connectivity matrix, such as derived from kneighbors_graph. If set to None it uses the graph that has adjacent
        pixels connected.
    :type connectivity: array-like, callable or None
    :param mask: An optional mask of the area to be excluded from clustering
    """

    def __init__(self, features, new_feature, distance_threshold=None, n_clusters=None, affinity="cosine",
                 linkage="single", remove_small=0, connectivity=None, mask=None):
        self.features = features
        self.distance_threshold = distance_threshold
        self.affinity = affinity
        self.linkage = linkage
        self.data_name = features,
        self.new_feature = new_feature
        self.n_clusters = n_clusters
        self.compute_full_tree = 'auto'
        if distance_threshold is not None:
            self.compute_full_tree = True

        self.remove_small = remove_small
        self.connectivity = connectivity
        self.mask = mask

    @staticmethod
    def construct_data(eopatch, features):
        """
        :param eopatch: EOPatch where with all features
        :param features: Features
        :return: vector constructed from the features listed
        """

        feature = EOTask._parse_features(features)

        return reduce(
            lambda acc, v: np.concatenate((acc, eopatch.data_timeless[v[1]]), axis=2) if acc is not None else
            eopatch.data_timeless[v[1]], feature, None)

    def execute(self, eopatch):
        """
        :param eopatch: Input EOPatch.
        :type eopatch: EOPatch
        :return: Transformed EOpatch
        :rtype: EOPatch
        """
        data = self.construct_data(eopatch, self.features)

        org_shape = data.shape
        data_long = np.reshape(data, (-1, org_shape[-1]))

        # If connectivity is not set, it uses graph with pixel-to-pixel connections
        if not self.connectivity:
            graph_args = {'n_x': org_shape[0], 'n_y': org_shape[1]}
            if self.mask:
                graph_args['mask'] = self.mask
            self.connectivity = grid_to_graph(**graph_args)

        model = AgglomerativeClustering(distance_threshold=self.distance_threshold, affinity=self.affinity,
                                        linkage=self.linkage,
                                        connectivity=self.connectivity,
                                        n_clusters=self.n_clusters,
                                        compute_full_tree=self.compute_full_tree)

        model.fit(data_long)
        labels = np.zeros(model.n_clusters_)
        trimmed_labels = model.labels_
        
        if self.remove_small != 0:
            def count_labels(acc, i):
                acc[i] += 1
                return acc

            labels = reduce(count_labels, model.labels_, labels)

            def trim(acc, v):
                (i, no_lab) = v
                if no_lab < self.remove_small:
                    acc[acc == i] = 0
                return acc

            reduce(trim, enumerate(labels), trimmed_labels)

        packed_labels = np.reshape(trimmed_labels, org_shape[:-1])

        eopatch.add_feature(FeatureType.DATA_TIMELESS, self.new_feature, packed_labels[..., np.newaxis])

        return eopatch
