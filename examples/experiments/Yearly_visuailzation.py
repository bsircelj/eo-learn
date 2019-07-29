
import random
from eolearn.core import EOPatch, FeatureType

import numpy as np
import matplotlib.pyplot as plt

def color_patch(image, colors=None):
    # Just for visualization of segments
    w, h = image.shape

    if colors is None:
        labels = np.max(image)
        colors = np.array([[0, 0, 0]])
        for _ in range(1, labels):
            n_col = np.array([[random.randint(15, 255), random.randint(15, 255), random.randint(15, 255)]])
            colors = np.concatenate((colors, n_col), axis=0)

    new_image = np.zeros((w, h, 3))
    for x in range(w - 1):
        for y in range(h - 1):
            new_image[x][y] = colors[image[x][y]]

    return new_image / 255


eopatch = EOPatch.load('./eopatch/patch2', lazy_loading=True)

bands = eopatch.data_timeless['SEGMENTS']
plt.imshow(color_patch(bands))

plt.show()
