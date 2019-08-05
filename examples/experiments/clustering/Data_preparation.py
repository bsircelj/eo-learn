import os
import sys
import numpy as np
# import matplotlib.pyplot as plt
from eolearn.core import EOTask, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk
# sys.stderr = open(os.devnull, "w")
# from eolearn.features import AddStreamTemporalFeaturesTask
# import cv2
# from eolearn.ml_tools.utilities import rolling_window
# from scipy.ndimage.measurements import label
from temporal_features_copy import AddStreamTemporalFeaturesTask


class printPatch(EOTask):
    def __init__(self, message="\npatch:"):
        self.message = message

    def execute(self, eopatch):
        print(self.message)
        print(eopatch)
        return eopatch


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
        red = eopatch.data['BANDS'][..., [2]]

        arvi = np.clip((nir - (2 * red) + blue) / (nir + (2 * red) + blue), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'ARVI', arvi)

        evi = np.clip(2.5 * ((nir - red) / (nir + (self.c1 * red) - (self.c2 * blue) + self.L)), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'EVI', evi)

        band_a = eopatch.data['BANDS'][..., 3]
        band_b = eopatch.data['BANDS'][..., 2]
        ndvi = np.clip((band_a - band_b) / (band_a + band_b), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'NDVI', ndvi[..., np.newaxis])

        band_a = eopatch.data['BANDS'][..., 1]
        band_b = eopatch.data['BANDS'][..., 3]
        ndvi = np.clip((band_a - band_b) / (band_a + band_b), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'NDWI', ndvi[..., np.newaxis])

        sipi = np.clip((nir - blue) / (nir - red + 0.000000001), 0, 2)  # TODO nekako boljše to rešit division by 0
        eopatch.add_feature(FeatureType.DATA, 'SIPI', sipi)

        Lvar = 0.5
        savi = np.clip(((nir - red) / (nir + red + Lvar)) * (1 + Lvar), -1, 1)
        eopatch.add_feature(FeatureType.DATA, 'SAVI', savi)

        print(eopatch)

        return eopatch


'''
3 ['NDVI_sd_val' 'NDVI_max_mean_surf' 'EVI_mean_val']
4 ['NDVI_sd_val' 'EVI_min_val' 'ARVI_max_mean_len' 'SIPI_mean_val']
5 ['NDVI_min_val' 'NDVI_sd_val' 'NDWI_max_mean_len' 'ARVI_max_mean_len' 'SAVI_min_val']
 
 '''


class ConstructVector(EOTask):
    def __init__(self, name, *args):
        self.name = name
        self.values = args

    def execute(self, eopatch): # TODO, probat dodat se lokacije not
        # h, w, _ = eopatch.data['BANDS']
        vector = None
        for v in self.values:
            print(v)
            vector = np.concatenate((vector, eopatch.data_timeless[v]), axis=2) if vector is not None else \
                eopatch.data_timeless[v]
        eopatch.add_feature(FeatureType.DATA_TIMELESS, self.name, vector)
        print(vector.shape)
        return eopatch


addStreamNDVI = AddStreamTemporalFeaturesTask(data_feature='NDVI')
addStreamSAVI = AddStreamTemporalFeaturesTask(data_feature='SAVI')
addStreamEVI = AddStreamTemporalFeaturesTask(data_feature='EVI')
addStreamARVI = AddStreamTemporalFeaturesTask(data_feature='ARVI')
addStreamSIPI = AddStreamTemporalFeaturesTask(data_feature='SIPI')
addStreamNDWI = AddStreamTemporalFeaturesTask(data_feature='NDWI')

create4 = ConstructVector('FILIP_FEATURES', 'NDVI_sd_val', 'EVI_min_val', 'ARVI_max_mean_len', 'SIPI_mean_val')

'''
3 ['NDVI_sd_val' 'NDVI_max_mean_surf' 'EVI_mean_val']
4 ['NDVI_sd_val' 'EVI_min_val' 'ARVI_max_mean_len' 'SIPI_mean_val']
5 ['NDVI_min_val' 'NDVI_sd_val' 'NDWI_max_mean_len' 'ARVI_max_mean_len' 'SAVI_min_val']

 '''

patch_location = './eopatch/'
load = LoadFromDisk(patch_location)

save_path_location = './eopatch/'
if not os.path.isdir(save_path_location):
    os.makedirs(save_path_location)

save = SaveToDisk(save_path_location, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

extra_param = {
    load: {'eopatch_folder': 'patch3'},
    save: {'eopatch_folder': 'patch4'}
}
'''
workflow = LinearWorkflow(
    load,
    AddFeatures(),
    save)
'''
workflow = LinearWorkflow(
    load,
    create4,
    save
)
'''
workflow1 = LinearWorkflow(
    load,
    # AddFeatures(),
    addStreamNDVI,
    printPatch("NDVI"),

    addStreamSAVI,
    printPatch("SAVI"),

    addStreamEVI,
    printPatch("EVI"),

    addStreamARVI,
    printPatch("ARVI"),

    addStreamSIPI,
    printPatch("SIPI"),

    addStreamNDWI,
    printPatch("NDWI"),
    save
)
'''
workflow.execute(extra_param)
