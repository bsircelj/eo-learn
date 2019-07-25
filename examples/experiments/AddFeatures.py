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

    def execute(self, eopatch):


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
