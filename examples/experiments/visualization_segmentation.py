from __future__ import print_function
import random
from eolearn.core import EOPatch, FeatureType

import numpy as np
import matplotlib.pyplot as plt


# https://matplotlib.org/2.1.2/gallery/animation/image_slices_viewer.html
class IndexTracker(object):
    # def __init__(self, ax, X, title, timestamps=None, cmap='gray', fig=None):
    def __init__(self, t):

        # images = [None] * len(t)
        self.t = t
        self.timestamps = t[0]['timestamp']
        sh = self.t[0]['data'].shape
        self.slices = sh[0]
        # self.ind = self.slices//2
        self.ind = 0

        for i in range(len(t)):
            self.fig = fig
            # self.ax = ax
            self.t[i]['ax'].set_title(self.t[i]['title'])
            # ax.set_title('use scroll wheel to navigate images')

            # self.X = X

            self.t[i]['im'] = t[i]['ax'].imshow(self.t[i]['data'][self.ind, ...], cmap=self.t[i]['cmap'])

            #plt.figure('init')
            #plt.imshow(self.t[i]['data'][self.ind, ...])

        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        for a in t:
            a['im'].set_data(a['data'][self.ind, :, :])

            #plt.figure('update')
            #plt.imshow(a['data'][self.ind, :, :])
            # self.ax.set_title('%s' % self.ind)
            # self.ax.set_title(timestamps[self.ind])
            if self.fig != None:
                self.fig.suptitle(timestamps[self.ind])
            # a['im'].axes.figure.canvas.draw()
        self.fig.canvas.draw()


def color_patches_temporal(temporal_image):
    t, w, h = temporal_image.shape
    new_colored = np.zeros((t, w, h, 3))

    labels = np.max(temporal_image) + 1
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
    #print(eopatch)
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

    data = [  # [FeatureType.DATA, 'SEGMENTS 0', None],
        # [FeatureType.MASK, 'UNLABELED0_SEGMENTS', 'gray'],
        [FeatureType.DATA, 'SEGMENTS 1', None],
        [FeatureType.MASK, 'UNLABELED1_SEGMENTS', 'gray'],
        # [FeatureType.DATA, 'SEGMENTS 2', None],
        # [FeatureType.MASK, 'UNLABELED2_SEGMENTS', 'gray'],
        [FeatureType.DATA, 'EVI', 'RdPu'],
        [FeatureType.DATA, 'EVI_SLOPE', 'RdPu'],
        [FeatureType.DATA, 'ARVI', 'Blues'],
        [FeatureType.DATA, 'ARVI_SLOPE', 'Blues'],
        [FeatureType.DATA, 'NDVI', 'YlGn'],
        [FeatureType.DATA, 'NDVI_SLOPE', 'YlGn'],
        # [FeatureType.DATA, 'EVI', 'RdPu'],
        # [FeatureType.MASK, 'EVI_EDGE', 'gray'],
        # [FeatureType.DATA, 'ARVI', 'Blues'],
        # [FeatureType.MASK, 'ARVI_EDGE', 'gray'],
        # [FeatureType.DATA, 'GRAY', 'gray'],
        # [FeatureType.MASK, 'GRAY_EDGE', 'gray']
    ]

    # aaa = eopatch[FeatureType.DATA, 'NDVI_SLOPE'][10].squeeze()
    #aaa = eopatch[data[3][0]][data[3][1]].squeeze()

    # aaa = aaa[10]
    #print(aaa[100:120, 100:120])
    #plt.figure(2)
    #plt.imshow(aaa[10], cmap='gray')
    # plt.show()

    i = 0
    t = []
    for w in range(width):
        for h in range(height):

            #w = 0
            #h = 1
            x = eopatch[data[i][0]][data[i][1]].squeeze()
            print(str(data[i][0]) + ' ' + data[i][1])
            print(x.shape)

            print(data[i][1])
            print(type(x))
            print(x[10, 100:110, 100:110])
            if data[i][1] == 'SEGMENTS 0' or data[i][1] == 'SEGMENTS 1' or data[i][1] == 'SEGMENTS 2':
                x = color_patches_temporal(x)
            t.append(
                {'ax': ax[h][w], 'data': x, 'title': data[i][1], 'timestamp': eopatch.timestamp, 'cmap': data[i][2],
                 'fig': fig})

            i = i + 1

    tracker = IndexTracker(t)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.15, hspace=0.2)
    plt.show()
