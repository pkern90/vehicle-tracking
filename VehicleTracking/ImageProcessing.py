import pickle

import matplotlib.pyplot as plt
from scipy.misc import imread
from itertools import compress

from CameraCalibration import get_camera_calibration
from ImageUtils import slide_window, draw_boxes, cut_out_windows, extract_features

orient = 9
pix_per_cell = 8
cells_per_block = 2
spatial_size = (32, 32)
hist_bins = 32
hist_range = (0, 256)
cspace = 'HSV'
hog_channel = 0

if __name__ == '__main__':
    cam_calibration = get_camera_calibration()

    img = imread('../test_images/test7.jpg')
    with open('../models/svm.p', 'rb') as f:
        clf = pickle.load(f)

    windows = slide_window(img, y_start_stop=[img.shape[0] - 512, None], xy_window=(256, 256), stride=(32, 32))
    samples = cut_out_windows(img, windows, resize=(64, 64))

    X = extract_features(samples, cspace=cspace, spatial_size=spatial_size, hist_bins=hist_bins,
                         hist_range=hist_range, orient=orient,
                         pix_per_cell=pix_per_cell, cells_per_block=cells_per_block, hog_channel=hog_channel,
                         normalize=True, pca=None)

    pred = clf.predict(X)
    windows = list(compress(windows, pred))

    img = draw_boxes(img, windows)
    plt.imshow(img)
    plt.show()

    # # Plot the examples
    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(img, cmap='gray')
    # plt.title('Example Car Image')
    # plt.subplot(122)
    # plt.imshow(hog_imaimg.shape[0]ge, cmap='gray')
    # plt.title('HOG Visualization')
    #
    # plt.show()
