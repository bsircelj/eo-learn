import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from eolearn.core import EOTask, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk
# sys.stderr = open(os.devnull, "w")

import cv2
from eolearn.ml_tools.utilities import rolling_window
from scipy.ndimage.measurements import label

import os

os.getcwd()
os.chdir("/home/beno/Documents/IJS/Perceptive-Sentinel/eo-learn/examples/experiments")
os.getcwd()


def normalize_feature(feature):  # Assumes similar max and min throughout different features
    f_min = np.min(feature)
    f_max = np.max(feature)
    if f_max != 0:
        return (feature - f_min) / (f_max - f_min)


def temporal_derivative(data, window_size=(3,)):
    padded_slope = np.zeros(data.shape)
    window = rolling_window(data, window_size, axes=0)

    slope = window[..., -1] - window[..., 0]  # TODO Missing division with time
    padded_slope[1:-1] = slope  # Padding with zeroes at the beginning and end

    return normalize_feature(padded_slope)


class AddFeatures(EOTask):

    def __init__(self, c1=6, c2=7.5, L=1):
        self.c1 = c1
        self.c2 = c2
        self.L = L

    def execute(self, eopatch):
        eopatch.add_feature(FeatureType.DATA, 'B0', eopatch.data['BANDS'][..., [0]])
        eopatch.add_feature(FeatureType.DATA, 'B1', eopatch.data['BANDS'][..., [1]])
        eopatch.add_feature(FeatureType.DATA, 'B2', eopatch.data['BANDS'][..., [2]])

        nir = eopatch.data['BANDS'][..., [3]]
        eopatch.add_feature(FeatureType.DATA, 'NIR', nir)
        blue = eopatch.data['BANDS'][..., [0]]
        # green = eopatch.data['BANDS'][..., [1]]
        red = eopatch.data['BANDS'][..., [2]]

        arvi = np.clip((nir - (2 * red) + blue) / (nir + (2 * red) + blue), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'ARVI', arvi)
        arvi_slope = temporal_derivative(arvi.squeeze())
        #print(arvi_slope.shape)
        eopatch.add_feature(FeatureType.DATA, 'ARVI_SLOPE', arvi_slope[..., np.newaxis])

        evi = np.clip(2.5 * ((nir - red) / (nir + (self.c1 * red) - (self.c2 * blue) + self.L)), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'EVI', evi)
        evi_slope = temporal_derivative(evi.squeeze())
        eopatch.add_feature(FeatureType.DATA, 'EVI_SLOPE', evi_slope[..., np.newaxis])
        #plt.imshow(evi_slope[10])
        #plt.show()

        band_a = eopatch.data['BANDS'][..., 3]
        band_b = eopatch.data['BANDS'][..., 2]
        ndvi = np.clip((band_a - band_b) / (band_a + band_b), -1, 1)
        #print(ndvi.shape)
        '''
        padded_slope = np.zeros(ndvi.shape)
        eopatch.add_feature(FeatureType.DATA, 'NDVI', ndvi[..., np.newaxis])
        window = rolling_window(ndvi, (3,), axes=0)
        slope = window[..., 2] - window[..., 0]
        padded_slope[1:-1] = slope
        padded_slope = normalize_feature(padded_slope)
        '''
        ndvi_slope = temporal_derivative(ndvi)
        eopatch.add_feature(FeatureType.DATA, 'NDVI_SLOPE', ndvi_slope[..., np.newaxis])  # ASSUMES EVENLY SPACED

        print(eopatch)

        return eopatch


patch_location = './eopatch/'
load = LoadFromDisk(patch_location)

save_path_location = './eopatch/'
if not os.path.isdir(save_path_location):
    os.makedirs(save_path_location)

save = SaveToDisk(save_path_location, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

extra_param = {
    load: {'eopatch_folder': 'patch'},
    save: {'eopatch_folder': 'patch1'}
}

workflow = LinearWorkflow(
    load,
    AddFeatures(),
    save
)

workflow.execute(extra_param)
