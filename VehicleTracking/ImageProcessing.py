import pickle

import matplotlib.pyplot as plt
import numpy as np
from ImageUtils import gaussian_blur, bb_by_contours, draw_boxes
from scipy.misc import imread

from VehicleTracking.CarDetector import detect_cars_multi_area

# Defines different search windows.
# They are only limited by the y coordinate
# The coordinates are based on the original window
# size and will be adjusted when resizing
Y_START_STOPS = np.array([
    [400, 400 + 96],
    [400, 400 + 128 + 16],
    [528 - 32, 528 + 128],
    [400, 400 + 256],
])

# Stride to use for each search area
STRIDE = np.array([
    [24, 24],
    [16, 16],
    [16, 16],
    [16, 16],
])

# Resize factor for each search area.
IMAGE_SIZE_FACTORS = [
    1.5,
    1,
    1,
    1 / 1.5,
]

X_PADDING = [
    24,
    32,
    32,
    32,
]

# The window size has to be the same for all
# search areas since the classifier expects
# a fixed feature size.
XY_WINDOW = (64, 64)

# The algorithm tries to run each search area on
# a separate cpu core. Therefore jobs > number of search areas
# wont't yield any implements
N_JOBS = 1

if __name__ == '__main__':
    with open('../models/svm_final.p', 'rb') as f:
        clf = pickle.load(f)

        # fig, axis = plt.subplots(2, 3)
        # for i in tqdm(range(0, 1)):
        #     img = imread('../test_images/test%s.jpg' % (i + 1))
        #     heat = detect_cars_multi_scale(img, clf, XY_WINDOW, STRIDE, Y_START_STOPS, IMAGE_SIZE_FACTORS, heatmap=True,
        #                                    n_jobs=N_JOBS)
        #     t1=time.time()
        #     heat_thresh = np.zeros(heat.shape, dtype=np.uint8)
        #     heat[heat == 0] = heat.min()
        #     heat = normalize(heat, new_max=255, new_min=0, dtype=np.uint8)
        #     # heat_blur = gaussian_blur(heat, 21)
        #     heat_thresh[heat > 30] = 255
        #
        #     boxes_contours = bb_by_contours(heat_thresh)
        #     img_contours = draw_boxes(img, boxes_contours, thick=3, color=(255, 0, 0))
        #
        #     axis[i // 3, i % 3].imshow(img_contours)
        #     axis[i // 3, i % 3].axis('off')
        #
        # plt.tight_layout()
        # plt.show()

img = imread('../test_images/test5.jpg')
img_blur = gaussian_blur(np.copy(img), 3)
heat = detect_cars_multi_area(img_blur,
                              clf,
                              XY_WINDOW,
                              STRIDE,
                              Y_START_STOPS,
                              IMAGE_SIZE_FACTORS,
                              X_PADDING,
                              heatmap=True,
                              n_jobs=N_JOBS)

# heat = normalize(heat, new_max=1., new_min=0, dtype=np.float64)
heat_thresh = np.zeros(heat.shape, dtype=np.uint8)
heat_blur = gaussian_blur(heat, 21)

heat_thresh[heat > 3.5] = 255

boxes_contours = bb_by_contours(heat_thresh)
img_contours = draw_boxes(img, boxes_contours, thick=3, color=(255, 0, 0))
plt.imshow(heat, cmap='gray')
plt.show()
