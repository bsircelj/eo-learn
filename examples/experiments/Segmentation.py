import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from eolearn.core import EOTask, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk
# sys.stderr = open(os.devnull, "w")

import cv2
from scipy.ndimage.measurements import label

import os

os.getcwd()
os.chdir("/home/beno/Documents/IJS/Perceptive-Sentinel/eo-learn/examples/experiments")
os.getcwd()


class Segmentation(EOTask):

    def __init__(self,
                 edge_features,
                 structuring_element,
                 excluded_features,
                 dilation_mask,
                 erosion_mask,
                 output_feature):

        self.edge_features = edge_features
        self.structuring_element = structuring_element
        self.excluded_features = excluded_features
        self.dilation_mask = dilation_mask
        self.erosion_mask = erosion_mask
        self.output_feature = output_feature

    def debug(self, object):
        print(type(object), object)

    def extract_edges(self, eopatch, feature_type, feature_name, feature_weight, low_threshold, high_threshold, blur):

        image = eopatch[feature_type][feature_name]
        t, w, h, _ = image.shape
        all_edges = np.zeros((t, w, h))
        for time in range(t):
            image_one = image[time]
            edge = self.one_edge(image_one, low_threshold, high_threshold, blur)
            all_edges[time] = edge
        eopatch.add_feature(FeatureType.MASK, feature_name + '_EDGE', all_edges[..., np.newaxis])
        return all_edges

    def one_edge(self, image, low_threshold, high_threshold, blur):
        ##########QUICK NORMALIZATION -  SHOULD BE LATER IMPROVED / MOVED SOMEWHERE ELSE
        f_min = np.min(image)
        f_max = np.max(image)
        image = (image - f_min) / f_max * 255
        image = image.squeeze()
        kernel_size, sigma = blur
        smoothed_image = cv2.GaussianBlur(image, kernel_size, sigma)
        edges = cv2.Canny(smoothed_image.astype(np.uint8), low_threshold, high_threshold)
        return edges

    def filter_unwanted_areas(self, eopatch, feature, threshold):
        # Returns mask of areas that should be excluded (low NDVI etc...)
        bands = eopatch[feature[0]][feature[1]]
        t, w, h, _ = bands.shape
        mask = np.zeros((t, w, h))

        for time in range(t):
            fet = eopatch[feature[0]][feature[1]][time].squeeze()
            mask_cur = fet <= threshold
            mask_cur = cv2.dilate((mask_cur * 255).astype(np.uint8), self.dilation_mask * 255)
            mask_cur = cv2.erode((mask_cur * 255).astype(np.uint8), self.erosion_mask * 255)
            mask[time] = mask_cur

        mask = mask > 0
        mask.shape
        plt.figure()
        plt.imshow(mask[10].squeeze(), cmap="gray")
        plt.show()
        eopatch.add_feature(FeatureType.MASK, 'LOW_' + feature[1], mask[..., np.newaxis])
        return mask

    def connected_components(self, image):

        # input_image = eopatch[self.feature_mask[0]][self.feature_mask[1]].squeeze()
        img = image.astype(np.uint8)
        # connected = cv2.connectedComponentsWithStats(img)
        # plt.figure(2,figsize=(50, 50 * aspect_ratio))
        # plt.imshow(color_patches(connected[1]))
        # eopatch.add_feature()
        connected, no_feat = label(img, self.structuring_element)
        print(no_feat)

        plt.figure("Whole Image")
        time = 8
        plt.imshow(image[time])

        plt.figure("segments")
        print(image[time][1:20, 1:20])
        plt.imshow(connected[time][1:20, 1:20], cmap="gray")

        plt.figure("Components")
        print(connected[time][1:20, 1:20])
        plt.imshow(connected[time][1:20, 1:20], cmap="nipy_spectral")
        
        plt.show()

        return connected

    def execute(self, eopatch):
        eopatch.add_feature(FeatureType.DATA, 'B0', eopatch.data['BANDS'][..., [0]])
        eopatch.add_feature(FeatureType.DATA, 'B1', eopatch.data['BANDS'][..., [1]])
        eopatch.add_feature(FeatureType.DATA, 'B2', eopatch.data['BANDS'][..., [2]])
        bands = eopatch.data['BANDS']
        t, w, h, _ = bands.shape

        # final_mask = np.zeros((w,h))
        edge_vector = np.zeros((t, w, h))
        # plt.figure("Edges")
        for arg in self.edge_features:
            one_edge = self.extract_edges(eopatch, arg['FeatureType'], arg['FeatureName'], arg['YearlyThreshold'],
                                          arg['CannyThresholds'][0], arg['CannyThresholds'][1], arg['BlurArguments'])
            # plt.imshow(one_edge[10])
            # plt.show()
            edge_vector = edge_vector + one_edge
        edge_vector = edge_vector > 0
        # print(eopatch)
        # print(edge_vector[..., np.newaxis].shape)
        eopatch.add_feature(FeatureType.MASK, 'SUM_EDGES', edge_vector[..., np.newaxis])

        for unwanted, threshold in self.excluded_features:
            mask = self.filter_unwanted_areas(eopatch, unwanted, threshold)
            edge_vector = np.logical_and(edge_vector, mask)

        components = self.connected_components(1 - edge_vector)

        eopatch.add_feature(self.output_feature[0], self.output_feature[1], components[..., np.newaxis])
        # print(eopatch)
        return eopatch


segmentation = Segmentation(
    edge_features=[
        {"FeatureType": FeatureType.DATA,
         "FeatureName": 'B0',
         "CannyThresholds": (20, 50),
         "YearlyThreshold": 0.4,
         "BlurArguments": ((5, 5), 2)
         },

        {"FeatureType": FeatureType.DATA,
         "FeatureName": 'B1',
         "CannyThresholds": (40, 60),
         "YearlyThreshold": 0.4,
         "BlurArguments": ((5, 5), 2)
         },

        {"FeatureType": FeatureType.DATA,
         "FeatureName": 'B2',
         "CannyThresholds": (30, 70),
         "YearlyThreshold": 0.4,
         "BlurArguments": ((5, 5), 2)
         },

        {"FeatureType": FeatureType.DATA,
         "FeatureName": 'NDVI',
         "CannyThresholds": (60, 120),
         "YearlyThreshold": 0.2,
         "BlurArguments": ((5, 5), 2)
         }
    ],
    structuring_element=[[[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]
                          ],
                         [[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]
                          ],
                         [[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]]
                         ],
    excluded_features=[((FeatureType.DATA, 'NDVI'), 0.3)],
    dilation_mask=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    erosion_mask=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    output_feature=(FeatureType.DATA, 'SEGMENTS'))

patch_location = './eopatch/'
load = LoadFromDisk(patch_location)

save_path_location = './eopatch/'
if not os.path.isdir(save_path_location):
    os.makedirs(save_path_location)

save = SaveToDisk(save_path_location, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

extra_param = {
    load: {'eopatch_folder': 'patch'},
    save: {'eopatch_folder': 'patch2'}
}

workflow = LinearWorkflow(
    load,
    segmentation,
    save
)

workflow.execute(extra_param)
