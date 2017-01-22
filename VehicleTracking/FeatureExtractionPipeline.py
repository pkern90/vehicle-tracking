import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from VehicleTracking.ImageUtils import convert_cspace, hog_feature_size, hog_features, bin_spatial, color_hist


class ColorSpaceConverter(BaseEstimator, TransformerMixin):
    def __init__(self, cspace, single_channel=None):
        self.single_channel = single_channel
        self.cspace = cspace

    def fit(self, data, y=None):
        return self

    def transform(self, images):
        if self.single_channel is None or self.cspace == 'GRAY':
            result = np.zeros(images.shape)
        else:
            result = np.zeros((*images.shape[:-1], 1))

        for i, img in enumerate(images):
            result[i] = convert_cspace(img, self.cspace)

        return result


class SpatialBining(BaseEstimator, TransformerMixin):
    def __init__(self, bins):
        if type(bins) in (tuple, list):
            self.bins = bins
        else:
            self.bins = (bins, bins)

    def fit(self, data, y=None):
        return self

    def transform(self, images):
        spatial = np.zeros((len(images), self.bins[0] * self.bins[0] * images.shape[-1]))

        for i, img in enumerate(images):
            spatial[i] = bin_spatial(img, size=self.bins)

        return spatial


class ColorHistogram(BaseEstimator, TransformerMixin):
    def __init__(self, bins=32, bins_range=(0, 256)):
        self.bins_range = bins_range
        self.bins = bins

    def fit(self, data, y=None):
        return self

    def transform(self, images):
        hists = np.zeros((len(images), self.bins * images.shape[-1]))

        for i, img in enumerate(images):
            hists[i] = color_hist(img, bins=self.bins, bins_range=self.bins_range)

        return hists


class HogExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, orient=9, pix_per_cell=8,
                 cells_per_block=2):
        self.cells_per_block = cells_per_block
        self.pix_per_cell = pix_per_cell
        self.orient = orient

    def fit(self, data, y=None):
        return self

    def transform(self, images):
        hogs = np.zeros(
            (len(images), 3 * hog_feature_size(images[0], self.pix_per_cell, self.cells_per_block, self.orient)))

        for i, img in enumerate(images):
            hogs[i] = hog_features(img, self.orient, self.pix_per_cell, self.cells_per_block, vis=False)

        return hogs
