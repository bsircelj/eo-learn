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

    '''
    i = 1
    for feature in argument_data:
        plt.figure(feature[1], figsize=(17, 10))
        # plt.title()
        i = i + 1
        img_base = eopatch[feature[0]][feature[1]][10].squeeze()
        img_edge = eopatch[FeatureType.MASK][feature[1] + '_EDGE'][10].squeeze()
        plt.subplot(1, 2, 1)
        plt.imshow(img_base)
        plt.subplot(1, 2, 2)
        plt.imshow(img_edge)
    
    time = 4
    plt.figure('Colors', figsize=(17, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(np.clip(eopatch.data['BANDS'][time][..., [2, 1, 0]] * 3.5, 0, 1))
    plt.subplot(1, 2, 2)
    #plt.imshow(color_patches(eopatch.data['SEGMENTS'][time].squeeze()))
    plt.imshow(eopatch.data['SEGMENTS'][time].squeeze(),cmap="nipy_spectral")

    plt.figure('Mask', figsize=(17, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(eopatch.mask['SUM_EDGES'][time].squeeze(), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(eopatch.mask['LOW_NDVI'][time].squeeze(), cmap="gray")

    plt.show()
    '''

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
            [FeatureType.DATA, 'NDVI', 'YlGn'],
            [FeatureType.MASK, 'NDVI_EDGE', 'gray'],
            [FeatureType.MASK, 'LOW_NDVI', 'gray'],
            [FeatureType.MASK, 'SUM_EDGES', 'gray'],
            [FeatureType.DATA, 'GRAY', 'gray'],
            [FeatureType.MASK, 'GRAY_EDGE', 'gray']
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

    '''
    # plt.subplot(width, height, i)
    X = eopatch.data['SEGMENTS'].squeeze()
    tracker.append(IndexTracker(ax[i], X, "Segments", cmap="nipy_spectral", timestamps=timestamps, fig=fig))

    # plt.subplot(width, height, i)
    i = i + 1
    X = eopatch.data['NDVI'].squeeze()
    tracker.append(IndexTracker(ax[i], X, 'NDVI', cmap="YlGn"))

    # plt.subplot(width, height, i)
    i = i + 1
    name = "SUM_EDGES"
    X = eopatch.data[name].squeeze()
    tracker.append(IndexTracker(ax[i], X, name))

    # plt.subplot(width, height, i)
    i = i + 1
    name = "LOW_NDVI"
    X = eopatch.data[name].squeeze()
    tracker.append(IndexTracker(ax[i], X, name))
    '''

    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.15, hspace=0.1)
    plt.show()
