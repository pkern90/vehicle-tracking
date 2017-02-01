"""
Contains transformer classes, supposed to use in a scikit-learn pipeline.
"""

import numpy as np
from ImageUtils import convert_cspace, hog_feature_size, hog_features, bin_spatial, color_hist, hog_features_opencv
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class ColorSpaceConverter(BaseEstimator, TransformerMixin):
    def __init__(self, cspace='RGB', single_channel=None):
        """
        Transforms given images into another color space

        :param cspace: one of RGB, HSV, LUV, HLS, YUV, LAB or GRAY
        :param single_channel:
        """
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
        """
        Transforms given images into feature vectors through spatial binning.

        :param bins:
        """
        if type(bins) in (tuple, list):
            self.bins = bins
        else:
            self.bins = (bins, bins)

    def fit(self, data, y=None):
        return self

    def transform(self, images):
        if len(images[0]) == 0:
            return images

        spatial = np.zeros((len(images), self.bins[0] * self.bins[1] * images.shape[-1]))

        for i, img in enumerate(images):
            spatial[i] = bin_spatial(img, size=(self.bins, self.bins))

        return spatial


class ColorHistogram(BaseEstimator, TransformerMixin):
    def __init__(self, bins=32, bins_range=(0, 256)):
        """
        Transforms given images into feature vectors by creating color histograms.

        :param bins:
        :param bins_range:
        """

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
        """
        Transforms given images into feature vectors by extracting hog features.

        :param orient: number of orientation bins
        :param pix_per_cell: Cell size in pixel. Takes a scalar value which is used as width and height
        :param cells_per_block: Block size in cells. Takes a scalar value which is used as width and height
        """
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


class HogExtractorOpenCV(BaseEstimator, TransformerMixin):
    def __init__(self, layout=None, orient=9, win_sigma=4.,
                 l2_hys_threshold=0.2, gamma_correction=1):
        """
        Transforms given images into feature vectors by extracting hog features using OpenCV.

        :param layout: (block_size_px, block_stride, pix_per_cell)
        :param orient: number of orientation bins
        :param win_sigma: Gaussian blur
        :param l2_hys_threshold:
        :param gamma_correction: Applies gamma correction of true.
        """

        self.gamma_correction = gamma_correction
        self.l2_hys_threshold = l2_hys_threshold
        self.win_sigma = win_sigma
        self.orient = orient

        if layout is None:
            layout = (16, 8, 8)

        self.pix_per_cell = layout[2]
        self.block_stride = layout[1]
        self.block_size_px = layout[0]

    def fit(self, data, y=None):
        return self

    def transform(self, images):
        if len(images[0]) == 0:
            return images

        cells_per_block = self.block_size_px // self.pix_per_cell
        hogs = np.zeros(
            (len(images), 3 * hog_feature_size(images[0], self.pix_per_cell, cells_per_block, self.orient)))

        for i, img in enumerate(images):
            hogs[i] = hog_features_opencv(img, self.block_size_px, self.block_stride, self.pix_per_cell, self.orient,
                                          self.win_sigma, self.l2_hys_threshold, self.gamma_correction)

        return hogs


class OptionalBranch(BaseEstimator, TransformerMixin):
    def __init__(self, use=True):
        """
        Transformer which allows to turn of a complete branch by replacing the arrays of images
        with an empty array.
        :param use: If set to false, branch will be deactivated
        """
        self.use = use

    def fit(self, data, y=None):
        return self

    def transform(self, images):
        if self.use:
            return images

        return np.ndarray((images.shape[0], 0))


class OptionalPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        """
        Wrapper for sklearns PCA which allows to turn it off completely (use all features).

        :param n_components: When set to None no PCA will be applied.
        """

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
    """
    Wrapper for sklearns MinMaxScaler which can handle empty input arrays.
    Needed when combined with OptionalBranch.
    """
    def fit(self, images, y=None):
        if len(images[0]) == 0:
            return self

        return super(AcceptEmptyMinMaxScaler, self).fit(images, y=None)

    def transform(self, images):
        if len(images[0]) == 0:
            return images

        return super(AcceptEmptyMinMaxScaler, self).transform(images)
