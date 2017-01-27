import numpy as np
from ImageUtils import convert_cspace, hog_feature_size, hog_features, bin_spatial, color_hist
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class ColorSpaceConverter(BaseEstimator, TransformerMixin):
    def __init__(self, cspace='RGB', single_channel=None):
        self.single_channel = single_channel
        self.cspace = cspace

    def fit(self, data, y=None):
        return self

    def transform(self, images):
        if len(images[0]) == 0:
            return images

        if self.single_channel is not None or self.cspace == 'GRAY':
            result = np.zeros((*images.shape[:-1], 1))
        else:
            result = np.zeros(images.shape)

        for i, img in enumerate(images):
            result[i] = convert_cspace(img, self.cspace)

        return result


class SpatialBining(BaseEstimator, TransformerMixin):
    def __init__(self, bins=32):
        if type(bins) in (tuple, list):
            self.bins = bins
        else:
            self.bins = (bins, bins)

    def fit(self, data, y=None):
        return self

    def transform(self, images):
        if len(images[0]) == 0:
            return images

        spatial = np.zeros((len(images), self.bins * self.bins * images.shape[-1]))

        for i, img in enumerate(images):
            spatial[i] = bin_spatial(img, size=(self.bins, self.bins))

        return spatial


class ColorHistogram(BaseEstimator, TransformerMixin):
    def __init__(self, bins=32, bins_range=(0, 256)):
        self.bins_range = bins_range
        self.bins = bins

    def fit(self, data, y=None):
        return self

    def transform(self, images):
        if len(images[0]) == 0:
            return images

        hists = np.zeros((len(images), self.bins * images.shape[-1]))

        for i, img in enumerate(images):
            hists[i] = color_hist(img, bins=self.bins, bins_range=self.bins_range)

        return hists


class HogExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, orient=9, pix_per_cell=8, cells_per_block=2):
        self.pix_per_cell = pix_per_cell
        self.cells_per_block = cells_per_block
        self.orient = orient

    def fit(self, data, y=None):
        return self

    def transform(self, images):
        if len(images[0]) == 0:
            return images

        hogs = np.zeros(
            (len(images), 3 * hog_feature_size(images[0], self.pix_per_cell, self.cells_per_block, self.orient)))

        for i, img in enumerate(images):
            hogs[i] = hog_features(img, self.orient, self.pix_per_cell, self.cells_per_block, vis=False)

        return hogs


class OptionalBranch(BaseEstimator, TransformerMixin):
    def __init__(self, use=True):
        self.use = use

    def fit(self, data, y=None):
        return self

    def transform(self, images):
        if self.use:
            return images

        return np.ndarray((images.shape[0], 0))


class OptionalPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        if n_components is None:
            self.pca = None
        else:
            self.pca = PCA(n_components=n_components)

    def fit(self, images, y):
        if len(images[0]) == 0:
            return self

        if self.pca is not None:
            self.pca.fit(images, y)
        return self

    def transform(self, images):
        if len(images[0]) == 0:
            return images

        if self.pca is not None:
            return self.pca.transform(images)
        else:
            return images


class AcceptEmptyMinMaxScaler(MinMaxScaler):
    def fit(self, images, y=None):
        if len(images[0]) == 0:
            return self

        return super(AcceptEmptyMinMaxScaler, self).fit(images, y=None)

    def transform(self, images):
        if len(images[0]) == 0:
            return images

        return super(AcceptEmptyMinMaxScaler, self).transform(images)
