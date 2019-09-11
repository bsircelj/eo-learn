import enum
import os
import sys
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib as mpl
from eolearn.core import EOTask, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk
# sys.stderr = open(os.devnull, "w")
# from eolearn.features import AddStreamTemporalFeaturesTask
# import cv2
# from eolearn.ml_tools.utilities import rolling_window
# from scipy.ndimage.measurements import label
from temporal_features_copy import AddStreamTemporalFeaturesTask
import geopandas as gpd

from geometry.eolearn.geometry import VectorToRaster


class allValid(EOTask):

    def __init__(self, mask_name):
        self.mask_name = mask_name

    def execute(self, eopatch):
        #print(eopatch)
        t, w, h, _ = eopatch.data['BANDS'].shape
        eopatch.add_feature(FeatureType.MASK, self.mask_name, np.ones((t, w, h, 1)))
        return eopatch


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


class LULC(enum.Enum):
    NO_DATA = (0, 'No Data', 'white')
    CULTIVATED_LAND = (1, 'Cultivated Land', 'xkcd:lime')
    FOREST = (2, 'Forest', 'xkcd:darkgreen')
    GRASSLAND = (3, 'Grassland', 'orange')
    SHRUBLAND = (4, 'Shrubland', 'xkcd:tan')
    WATER = (5, 'Water', 'xkcd:azure')
    WETLAND = (6, 'Wetlands', 'xkcd:lightblue')
    TUNDRA = (7, 'Tundra', 'xkcd:lavender')
    ARTIFICIAL_SURFACE = (8, 'Artificial Surface', 'crimson')
    BARELAND = (9, 'Bareland', 'xkcd:beige')
    SNOW_AND_ICE = (10, 'Snow and Ice', 'black')

    def __init__(self, val1, val2, val3):
        self.id = val1
        self.class_name = val2
        self.color = val3


class ConstructVector(EOTask):
    def __init__(self, name, *args):
        self.name = name
        self.values = args

    def execute(self, eopatch):
        # h, w, _ = eopatch.data['BANDS']
        vector = None
        for v in self.values:
            print(v)
            vector = np.concatenate((vector, eopatch.data_timeless[v]), axis=2) if vector is not None else \
                eopatch.data_timeless[v]
        eopatch.add_feature(FeatureType.DATA_TIMELESS, self.name, vector)
        print(vector.shape)
        return eopatch


class JoinTemporalFeatures(EOTask):
    def __init__(self, *args):
        self.features = args
        # self.description = ['max_val','min_val','mean_val','sd_val','diff_max','diff_min',

    def execute(self, eopatch):
        data_sum = None
        for f in self.features:
            names, data = f.get_data(eopatch)
            data_sum = data if data_sum is None else np.append(data_sum, data, axis=2)
        eopatch.add_feature(FeatureType.DATA_TIMELESS, 'FILIP_ALL', data_sum)
        print(eopatch)
        return eopatch


lulc_cmap = mpl.colors.ListedColormap([entry.color for entry in LULC])
lulc_norm = mpl.colors.BoundaryNorm(np.arange(-0.5, 11, 1), lulc_cmap.N)

land_cover_path = 'land_cover_subset_small.shp'

land_cover = gpd.read_file(land_cover_path)

land_cover_val = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
land_cover_array = []
for val in land_cover_val:
    temp = land_cover[land_cover.lulcid == val]
    temp.reset_index(drop=True, inplace=True)
    land_cover_array.append(temp)
    del temp

rshape = (FeatureType.MASK, 'IS_VALID')

land_cover_task_array = []
for el, val in zip(land_cover_array, land_cover_val):
    land_cover_task_array.append(VectorToRaster(
        feature=(FeatureType.MASK_TIMELESS, 'LULC'),
        vector_data=el,
        raster_value=val,
        raster_shape=rshape,
        raster_dtype=np.uint8))

addStreamNDVI = AddStreamTemporalFeaturesTask(data_feature='NDVI')
addStreamSAVI = AddStreamTemporalFeaturesTask(data_feature='SAVI')
addStreamEVI = AddStreamTemporalFeaturesTask(data_feature='EVI')
addStreamARVI = AddStreamTemporalFeaturesTask(data_feature='ARVI')
addStreamSIPI = AddStreamTemporalFeaturesTask(data_feature='SIPI')
addStreamNDWI = AddStreamTemporalFeaturesTask(data_feature='NDWI')

create4 = ConstructVector('FILIP_FEATURES', 'NDVI_sd_val', 'EVI_min_val', 'ARVI_max_mean_len', 'SIPI_mean_val')

create6 = ConstructVector('FILIP_FEATURES', 'NDVI_sd_val', 'EVI_min_val', 'ARVI_max_mean_len', 'SIPI_mean_val',
                          'NDVI_max_mean_surf', 'SAVI_min_val')

create_all = JoinTemporalFeatures(addStreamNDVI, addStreamARVI, addStreamEVI, addStreamNDWI, addStreamSAVI,
                                  addStreamSIPI)
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
    load: {'eopatch_folder': 'patch2'},
    save: {'eopatch_folder': 'patch11_edges'}
}
'''
workflow = LinearWorkflow(
    load,
    AddFeatures(),
    save)
'''
workflow = LinearWorkflow(
    load,

    create6,
    create_all,
    allValid('IS_VALID'),
    *land_cover_task_array,
    printPatch(),
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
