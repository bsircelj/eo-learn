import numpy as np
import matplotlib.pyplot as plt
from eolearn.core import EOTask, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk

import cv2
from scipy.ndimage.measurements import label
import os

from examples.experiments.Yearly_visualization import display

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
                 adjust_function,
                 # adjust_feature,
                 adjust_threshold,
                 yearly_low_threshold):

        self.edge_features = edge_features
        self.structuring_element = structuring_element
        self.excluded_features = excluded_features
        self.dilation_mask = dilation_mask
        self.erosion_mask = erosion_mask
        self.output_feature = output_feature
        self.adjust_function = adjust_function
        # self.adjust_feature = adjust_feature
        self.adjust_threshold = adjust_threshold
        self.yearly_low_threshold = yearly_low_threshold

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
        return edges > 0

    def filter_unwanted_areas(self, eopatch, feature, threshold):
        # Returns mask of areas that should be excluded (low NDVI etc...)
        bands = eopatch[feature[0]][feature[1]]
        t, w, h, _ = bands.shape
        mask = np.zeros((w, h))

        for time in range(t):
            fet = eopatch[feature[0]][feature[1]][time].squeeze()
            mask_cur = fet <= threshold
            mask_cur = cv2.dilate((mask_cur * 255).astype(np.uint8), self.dilation_mask * 255)
            mask_cur = cv2.erode((mask_cur * 255).astype(np.uint8), self.erosion_mask * 255)
            mask = mask + mask_cur

        # print(mask[0:20, 0:20]/t)
        mask = (mask / t) > self.yearly_low_threshold
        # mask.shape
        eopatch.add_feature(FeatureType.MASK_TIMELESS, 'LOW_' + feature[1], mask[..., np.newaxis])
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

    def normalize_feature(self, feature):
        f_min = np.min(feature)
        f_max = np.max(feature)
        return (feature - f_min) / f_max

    def execute(self, eopatch):
        # eopatch.add_feature(FeatureType.DATA, 'WEIGHT',

        bands = eopatch.data['BANDS']
        t, w, h, _ = bands.shape

        no_feat = len(self.edge_features)
        edge_vector = np.zeros((no_feat, t, w, h))
        # sum_vector = np.zeros((no_feat,t, w, h))
        for i in range(no_feat):
            arg = self.edge_features[i]
            one_edge = self.extract_edges(eopatch, arg['FeatureType'], arg['FeatureName'],
                                          arg['CannyThresholds'][0], arg['CannyThresholds'][1], arg['BlurArguments'])
            # print(eopatch[arg['FeatureType']][arg['FeatureName']].squeeze()[0].shape)
            v1 = eopatch[arg['FeatureType']][arg['FeatureName']].squeeze()
            v1 = self.normalize_feature(v1)
            # print(v1.shape)
            # print(v1[5][100:120][100:120])
            v1 = [self.adjust_function(x) for x in v1]
            # print(v1[5][100:120][100:120])
            # weight = [self.adjust_function(x) for x in eopatch[arg['FeatureType']][arg['FeatureName']].squeeze()]
            edge_vector[i] = one_edge * v1

        # edge_vector = edge_vector
        edge_vector1 = np.sum(edge_vector, (0, 1))
        edge_vector1 = edge_vector1 / (t * len(self.edge_features))
        edge_vector = edge_vector1 > self.adjust_threshold
        eopatch.add_feature(FeatureType.MASK_TIMELESS, 'WEIGHTED_EDGES', edge_vector[..., np.newaxis])

        for unwanted, threshold in self.excluded_features:
            mask = self.filter_unwanted_areas(eopatch, unwanted, threshold)

            edge_vector = np.logical_or(edge_vector, mask)

        # con = self.find_contours(edge_vector)

        edge_vector = 1 - edge_vector
        eopatch.add_feature(FeatureType.MASK_TIMELESS, 'WEIGHTED_SEGMENTS', edge_vector[..., np.newaxis])
        components = self.connected_components(edge_vector)

        eopatch.add_feature(self.output_feature[0], self.output_feature[1], components[..., np.newaxis])
        print(eopatch)
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

st2 = [[[0, 0, 0],
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
structuring_2d = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]
                  ]

segmentation = Segmentation(
    edge_features=[
        {"FeatureType": FeatureType.DATA,
         "FeatureName": 'EVI',
         "CannyThresholds": (40, 80),
         "BlurArguments": ((5, 5), 2)
         },

        {"FeatureType": FeatureType.DATA,
         "FeatureName": 'ARVI',
         "CannyThresholds": (40, 80),
         "BlurArguments": ((5, 5), 2)
         },
        {"FeatureType": FeatureType.DATA,
         "FeatureName": 'NDVI',
         "CannyThresholds": (40, 100),
         "BlurArguments": ((5, 5), 2)
         },
        {"FeatureType": FeatureType.DATA,
         "FeatureName": 'GRAY',
         "CannyThresholds": (5, 40),
         "BlurArguments": ((3, 3), 2)
         }
    ],
    structuring_element=structuring_2d,
    excluded_features=[((FeatureType.DATA, 'NDVI'), 0.3)],
    dilation_mask=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    erosion_mask=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    output_feature=(FeatureType.DATA_TIMELESS, 'SEGMENTS'),
    adjust_function=lambda x: cv2.GaussianBlur(x, (9, 9), 5),
    adjust_threshold=0.05,
    yearly_low_threshold=0.8)


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

display()
