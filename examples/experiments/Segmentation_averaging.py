from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
from eolearn.core import EOTask, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk
from eolearn.ml_tools.utilities import rolling_window

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
                 output_feature,
                 window_sizes):

        self.edge_features = edge_features
        self.structuring_element = structuring_element
        self.excluded_features = excluded_features
        self.dilation_mask = dilation_mask
        self.erosion_mask = erosion_mask
        self.output_feature = output_feature
        self.window_sizes = window_sizes

    def debug(self, object):
        print(type(object), object)

    def extract_edges(self, eopatch, feature_type, feature_name, low_threshold, high_threshold, blur):

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
        #mask.shape
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

        return connected

    # computes logical AND between all adjacent cells with set offset along temporal dimension
    def join_temporal_neighbours(self, array, offset):
        t, w, h = array.shape
        new_array = np.zeros((t, w, h))
        for i in range(t):
            low = i - offset if i - offset > 0 else 0
            high = i + offset + 1 if i + offset + 1 <= t else t
            for wi in range(w):  # Width index
                for hi in range(h):  # Height index
                    # print(low, high, low:high , wi, hi, array.shape)
                    # print(array[low:high])
                    # print("___________")
                    # print((array[..., wi, hi])[low:high])
                    new_array[i][wi][hi] = reduce(lambda x, y: np.logical_and(x, y), ((array[..., wi, hi])[low:high]))

    def join_temporal_neighbours2(self, array, offset):
        t, w, h = array.shape
        new_array = np.zeros((t, w, h))
        for i in range(t):
            low = i - offset if i - offset > 0 else 0
            high = i + offset + 1 if i + offset + 1 <= t else t
            # print(low, high, low:high , wi, hi, array.shape)
            # print(array[low:high])
            # print("___________")
            # print((array[..., wi, hi])[low:high])
            new_array[i] = reduce(lambda x, y: np.logical_or(x, y), array[low:high])

        return new_array

    def execute(self, eopatch):

        bands = eopatch.data['BANDS']
        t, w, h, _ = bands.shape

        edge_vector = np.zeros((t, w, h))
        averaged_edges = np.zeros((t, w, h, len(self.window_sizes)))

        for i in range(len(self.edge_features)):
            arg = self.edge_features[i]
            one_edge = self.extract_edges(eopatch, arg['FeatureType'], arg['FeatureName'],
                                          arg['CannyThresholds'][0], arg['CannyThresholds'][1], arg['BlurArguments'])
            edge_vector = edge_vector + one_edge
        edge_vector = edge_vector > 0
        eopatch.add_feature(FeatureType.MASK, 'SUM_EDGES', edge_vector[..., np.newaxis])

        for w_size in range(len(self.window_sizes)):
            averaged_edges[..., w_size] = self.join_temporal_neighbours2(edge_vector, self.window_sizes[w_size])
            eopatch.add_feature(FeatureType.MASK, 'SUM' + str(w_size) + '_EDGES',
                                averaged_edges[..., w_size, np.newaxis])

            for unwanted, threshold in self.excluded_features:
                mask = self.filter_unwanted_areas(eopatch, unwanted, threshold)
                averaged_edges[..., w_size] = np.logical_or(averaged_edges[..., w_size], mask)

            # con = self.find_contours(edge_vector)

            averaged_edges[..., w_size] = 1 - averaged_edges[..., w_size]
            #print(averaged_edges[..., w_size].shape)
            eopatch.add_feature(FeatureType.MASK, 'UNLABELED' + str(w_size) + '_SEGMENTS',
                                averaged_edges[..., w_size, np.newaxis])
            components = self.connected_components(averaged_edges[..., w_size])
            eopatch.add_feature(self.output_feature[0], self.output_feature[1] + ' ' + str(w_size),
                                components[..., np.newaxis])

        return eopatch


b_low = 10
b_high = 40
st1 = [[[0, 0, 0],
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
       ]

connectivity_1 = \
    [[[0, 0, 0],
      [0, 1, 0],
      [0, 0, 0]
      ],
     [[0, 1, 0],
      [1, 1, 1],
      [0, 1, 0]
      ],
     [[0, 0, 0],
      [0, 1, 0],
      [0, 0, 0]]
     ]

no_temporal = \
    [[[0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]
      ],
     [[0, 1, 0],
      [1, 1, 1],
      [0, 1, 0]
      ],
     [[0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]]
     ]

segmentation = Segmentation(
    edge_features=[
        {"FeatureType": FeatureType.DATA,
         "FeatureName": 'EVI',
         "CannyThresholds": (40, 80),
         "BlurArguments": ((3, 3), 1)
         },

        {"FeatureType": FeatureType.DATA,
         "FeatureName": 'ARVI',
         "CannyThresholds": (40, 80),
         "BlurArguments": ((3, 3), 1)
         },
        {"FeatureType": FeatureType.DATA,
         "FeatureName": 'NDVI',
         "CannyThresholds": (40, 80),
         "BlurArguments": ((3, 3), 1)
         },
        {"FeatureType": FeatureType.DATA,
         "FeatureName": 'GRAY',
         "CannyThresholds": (5, 40),
         "BlurArguments": ((3, 3), 2)
         }
    ],
    structuring_element=connectivity_1,
    excluded_features=[((FeatureType.DATA, 'NDVI'), 0.3)],
    dilation_mask=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    erosion_mask=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    output_feature=(FeatureType.DATA, 'SEGMENTS'),
    # window_sizes=[0,1,2,3,23])
    window_sizes=[0, 1, 2])


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


class Preprocess(EOTask):

    def execute(self, eopatch):
        img = np.clip(eopatch.data['BANDS'][..., [2, 1, 0]] * 3.5, 0, 1)
        t, w, h, _ = img.shape
        gray_img = np.zeros((t, w, h))
        print(img[0].shape)
        for time in range(t):
            img0 = np.clip(eopatch[FeatureType.DATA]['BANDS'][time][..., [2, 1, 0]] * 3.5, 0, 1)
            img = rgb2gray(img0)
            gray_img[time] = (img * 255).astype(np.uint8)

        eopatch.add_feature(FeatureType.DATA, 'GRAY', gray_img[..., np.newaxis])
        print(eopatch)
        return eopatch


patch_location = './eopatch/'
load = LoadFromDisk(patch_location)

save_path_location = './eopatch/'
if not os.path.isdir(save_path_location):
    os.makedirs(save_path_location)

save = SaveToDisk(save_path_location, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

extra_param = {
    load: {'eopatch_folder': 'patch1'},
    save: {'eopatch_folder': 'patch2'}
}

workflow = LinearWorkflow(
    load,
    segmentation,
    # Preprocess(),
    save
)

workflow.execute(extra_param)
