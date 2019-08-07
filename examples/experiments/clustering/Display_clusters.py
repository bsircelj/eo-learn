import random
from eolearn.core import EOPatch, FeatureType

import numpy as np
import matplotlib.pyplot as plt


def color_patch(image, colors=None):
    # Just for visualization of segments
    w, h = image.shape
    print(image.shape)
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


def display():
    eopatch = EOPatch.load('./eopatch/patch5', lazy_loading=True)
    print(eopatch)

    #plt.subplot(1, 2, 1)
    seg = eopatch.data_timeless['CLUSTERS'].squeeze()
    print(seg.shape)
    plt.imshow(color_patch(seg))


    plt.show()

display()