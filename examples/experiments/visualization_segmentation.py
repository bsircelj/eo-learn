from __future__ import print_function
import random
from eolearn.core import EOPatch, FeatureType

import numpy as np
import matplotlib.pyplot as plt

#https://matplotlib.org/2.1.2/gallery/animation/image_slices_viewer.html
class IndexTracker(object):
    def __init__(self, ax, X, title, timestamps=None, cmap='gray', fig=None):
        self.timestamps = timestamps
        self.fig = fig
        self.ax = ax
        ax.set_title(title)
        # ax.set_title('use scroll wheel to navigate images')

        self.X = X
        sh = X.shape
        self.slices = sh[0]
        # self.ind = self.slices//2
        self.ind = 0

        self.im = ax.imshow(self.X[self.ind, ...], cmap=cmap)
        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, :, :])
        # self.ax.set_title('%s' % self.ind)
        # self.ax.set_title(timestamps[self.ind])
        if self.fig != None:
            self.fig.suptitle(timestamps[self.ind])
        self.im.axes.figure.canvas.draw()


def color_patches_temporal(temporal_image):
    t, w, h = temporal_image.shape
    new_colored = np.zeros((t, w, h, 3))

    labels = np.max(temporal_image)+1
    colors = np.array([[0, 0, 0]])
    for _ in range(1, labels):
        n_col = np.array([[random.randint(30, 255), random.randint(30, 255), random.randint(30, 255)]])
        colors = np.concatenate((colors, n_col), axis=0)

    for time in range(t):
        new_colored[time] = color_patch(temporal_image[time], colors)

    return new_colored


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


if __name__ == "__main__":
    eopatch = EOPatch.load('./eopatch/patch2', lazy_loading=True)
    print(eopatch)
    argument_data = [
        (FeatureType.DATA, 'B0'),
        (FeatureType.DATA, 'B1'),
        (FeatureType.DATA, 'B2'),
        (FeatureType.DATA, 'NDVI')]


    bands = np.clip(eopatch.data['BANDS'][..., [2, 1, 0]] * 3.5, 0, 1)
    timestamps = eopatch.timestamp

    # print(bands.shape)
    width, height = 4, 2
    tracker = []
    fig, ax = plt.subplots(height, width)
    fig.set_size_inches(20, 20)
    i = 0

    data = [[FeatureType.DATA, 'SEGMENTS', None],
            [FeatureType.MASK, 'UNLABELED_SEGMENTS', 'gray'],
            #[FeatureType.MASK, 'LOW_NDVI', 'gray'],
            #[FeatureType.MASK, 'SUM_EDGES', 'gray'],
            [FeatureType.DATA, 'NDVI', 'YlGn'],
            [FeatureType.MASK, 'NDVI_EDGE', 'gray'],
            [FeatureType.DATA, 'EVI', 'RdPu'],
            [FeatureType.MASK, 'EVI_EDGE', 'gray'],
            [FeatureType.DATA, 'ARVI', 'Blues'],
            [FeatureType.MASK, 'ARVI_EDGE', 'gray'],
            #[FeatureType.DATA, 'GRAY', 'gray'],
            #[FeatureType.MASK, 'GRAY_EDGE', 'gray']
            ]

    i = 0
    for w in range(width):
        for h in range(height):
            x = eopatch[data[i][0]][data[i][1]].squeeze()
            if data[i][1] == 'SEGMENTS':
                x = color_patches_temporal(x)
            t = IndexTracker(ax[h][w], x, data[i][1], eopatch.timestamp, data[i][2], fig)
            tracker.append(t)
            fig.canvas.mpl_connect('scroll_event', t.onscroll)
            i = i + 1


    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.15, hspace=0.13)
    plt.show()
