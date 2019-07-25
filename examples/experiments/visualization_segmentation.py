import numpy as np
import random
import matplotlib.pyplot as plt

from eolearn.core import EOPatch, FeatureType


def color_patches(image):
    # Just for visualization of segments
    w, h = image.shape
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
    '''

    plt.figure('Colors', figsize=(17, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(np.clip(eopatch.data['BANDS'][0][..., [2, 1, 0]] * 3.5, 0, 1))
    plt.subplot(1, 2, 2)
    #plt.imshow(color_patches(eopatch.data['SEGMENTS'][10].squeeze()))
    plt.imshow(eopatch.data['SEGMENTS'][10].squeeze(),cmap="nipy_spectral")

    plt.figure('Mask', figsize=(17, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(eopatch.mask['SUM_EDGES'][10].squeeze())
    plt.subplot(1, 2, 2)
    plt.imshow(eopatch.mask['LOW_NDVI'][10].squeeze())

    plt.show()
