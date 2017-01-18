import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
from skimage.transform import pyramid_gaussian

from CameraCalibration import get_camera_calibration
from ImageUtils import slide_window, cut_out_windows, extract_features, draw_boxes

orient = 9
pix_per_cell = 8
cells_per_block = 2
spatial_size = (32, 32)
hist_bins = 32
hist_range = (0, 256)
cspace = 'LAB'
hog_channel = 0

xy_window = (64, 64)
max_pyramid_layer = 2

if __name__ == '__main__':
    cam_calibration = get_camera_calibration()

    img = imread('../test_images/test1.jpg')
    # img = imread('../data/vehicles/GTI_Right/image0001.png')
    with open('../models/svm_hog.p', 'rb') as f:
        clf = pickle.load(f)

    detections = np.empty((0, 4), dtype=np.uint32)
    pyramid = pyramid_gaussian(img, downscale=2, max_layer=max_pyramid_layer)
    for scale, img_scaled in enumerate(pyramid):
        img_scaled = (img_scaled * 255).astype(np.uint8)

        # check if search area is smaller then window.
        y_start_stop = [img_scaled.shape[0] * 0.55, img.shape[0] * 0.93]
        search_area_height = y_start_stop[1] - y_start_stop[0]
        if search_area_height < xy_window[1] or img_scaled.shape[1] < xy_window[0]:
            break

        windows = slide_window(img_scaled, y_start_stop=y_start_stop, xy_window=xy_window,
                               stride=(16, 16))

        samples = cut_out_windows(img, windows)
        X = extract_features(samples, cspace=cspace, orient=orient, pix_per_cell=pix_per_cell,
                             cells_per_block=cells_per_block, normalize=True)

        pred = clf.predict(X)
        des_func = clf.decision_function(X)
        windows = windows[(pred == 1) & (des_func > 2)]

        img_scaled = draw_boxes(img_scaled, windows)
        scale_factor = 2 * scale if scale > 0 else 1
        windows *= scale_factor
        detections = np.append(detections, windows, axis=0)

    img = draw_boxes(img, detections)
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
