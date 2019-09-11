import enum
import random
from eolearn.core import EOPatch, FeatureType
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from examples.experiments.clustering.Clustering_evaluation import evaluate_clusters


def color_patch(image, colors=None):
    # Just for visualization of segments
    w, h = image.shape
    # print(image.shape)
    if colors is None:
        labels = np.max(image)
        colors = np.array([[0, 0, 0]])
        for _ in range(labels):
            n_col = np.array([[random.randint(15, 255), random.randint(15, 255), random.randint(15, 255)]])
            colors = np.concatenate((colors, n_col), axis=0)

    new_image = np.zeros((w, h, 3))
    for x in range(w):
        for y in range(h):
            a = image[x][y]
            c = colors[a]
            new_image[x][y] = c
            # new_image[x][y] = colors[image[x][y]]

    return new_image / 255


def filter_unwanted_areas(eopatch, feature, threshold):
    # Returns mask of areas that should be excluded (low NDVI etc...)
    bands = eopatch[feature[0]][feature[1]]
    t, w, h, _ = bands.shape
    mask = np.zeros((t, w, h))
    mask_sum = np.zeros((w, h))
    for time in range(t):
        fet = eopatch[feature[0]][feature[1]][time].squeeze()
        mask_cur = fet <= threshold
        mask_cur = cv2.dilate((mask_cur * 255).astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) * 255)
        mask_cur = cv2.erode((mask_cur * 255).astype(np.uint8),
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) * 255)
        mask[time] = mask_cur
        mask_sum = mask_sum + mask_cur

    # mask = mask > 0
    mask_sum = mask_sum / t > 0.9
    # mask.shape
    # eopatch.add_feature(FeatureType.MASK, 'LOW_' + feature[1], mask[..., np.newaxis])
    return mask_sum


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


def display(title='Clustering'):
    eopatch = EOPatch.load('./eopatch/TestResults', lazy_loading=True)
    # print(eopatch)
    # plt.suptitle(title)
    plt.subplot(1, 2, 1)
    data = eopatch.data_timeless['clusters_small']
    # data = [-1 if x is None else x for x in data]
    # data = data + 1
    seg = data.squeeze()
    plt.imshow(color_patch(seg))

    plt.subplot(1, 2, 2)
    data = eopatch.data_timeless['clusters_mask']
    # data = [-1 if x is None else x for x in data]
    # data = data + 1
    seg = data.squeeze()
    plt.imshow(color_patch(seg))
    # print(seg)
    # filtered = filter_unwanted_areas(eopatch, (FeatureType.DATA, 'NDVI'), 0.3)

    # seg[filtered == 1] = 0

    # ax.imshow(color_patch(seg))
    # ax.set_title("4 features")
    '''
    ax2 = plt.subplot(1, 2, 2)
    seg1 = eopatch.data_timeless['CLUSTERS1'].squeeze()
    plt.imshow(color_patch(seg1))
    ax2.set_title("108 features")
    '''
    '''
    ax2 = plt.subplot(1, 3, 3)

    lulc_cmap = mpl.colors.ListedColormap([entry.color for entry in LULC])
    lulc_norm = mpl.colors.BoundaryNorm(np.arange(-0.5, 11, 1), lulc_cmap.N)
    lulc = eopatch.mask_timeless['LULC'].squeeze()

    ax2.imshow(lulc, cmap=lulc_cmap, norm=lulc_norm)

    ax.set_title("{:.3f}%".format(evaluate_clusters(seg, lulc) * 100))

    ax3 = plt.subplot(1, 3, 2)
    seg_edge = eopatch.data_timeless['SEGMENTS'].squeeze()
    ax3.imshow(color_patch(seg_edge))
    ax3.set_title("{:.2f}%".format(evaluate_clusters(seg_edge, lulc) * 100))
    '''
    plt.show()


if __name__ == '__main__':
    display()
