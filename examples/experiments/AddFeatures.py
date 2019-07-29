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
        # nir = nir.squeeze()

        arvi = np.clip((nir - (2 * red) + blue) / (nir + (2 * red) + blue), -10, 10)

        eopatch.add_feature(FeatureType.DATA, 'ARVI', arvi)

        evi = np.clip(2.5 * ((nir - red) / (nir + (self.c1 * red) - (self.c2 * blue) + self.L)), -10, 10)

        eopatch.add_feature(FeatureType.DATA, 'EVI', evi)
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
