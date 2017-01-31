import pickle

import matplotlib.pyplot as plt
import numpy as np
from ImageUtils import detect_cars_multi_scale, gaussian_blur, normalize, bb_by_contours, draw_boxes
from scipy.misc import imread
from tqdm import tqdm

Y_START_STOPS = np.array([
    [400, 496],
    [400, 496],
    [400, 592],
    [366, 690]
])
STRIDE = np.array([
    [24, 24],
    [16, 16],
    [16, 16],
    [16, 16]
])
IMAGE_SIZE_FACTORS = [
    1.5,
    1,
    1 / 1.5,
    1 / 3
]
XY_WINDOW = (64, 64)
N_JOBS = 4

if __name__ == '__main__':
    with open('../models/svm_adj.p', 'rb') as f:
        clf = pickle.load(f)

    fig, axis = plt.subplots(2, 3)
    for i in tqdm(range(0, 6)):
        img = imread('../test_images/test%s.jpg' % (i + 1))
        heat = detect_cars_multi_scale(img, clf, XY_WINDOW, STRIDE, Y_START_STOPS, IMAGE_SIZE_FACTORS, heatmap=True,
                                       n_jobs=N_JOBS)

        heat_thresh = np.zeros(heat.shape, dtype=np.uint8)
        heat_blur = gaussian_blur(heat, 21)
        heat_thresh[heat_blur > 2] = 255

        boxes_contours = bb_by_contours(heat_thresh)
        img_contours = draw_boxes(img, boxes_contours, thick=3, color=(255, 0, 0))

        axis[i // 3, i % 3].imshow(img_contours)
        axis[i // 3, i % 3].axis('off')

    plt.tight_layout()
    plt.show()
