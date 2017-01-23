import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hcluster
from scipy.misc import imread
from skimage.feature import blob_doh
from tqdm import tqdm
import timeit

from ImageUtils import center_points, surrounding_box, detect_cars_multi_scale
from ImageUtils import draw_boxes

Y_START_STOPS = np.array([[400, 464],
                          [400, 592],
                          [400, 688],
                          [400, 688]])
IMAGE_SIZE_FACTORS = [1.5, 1, 1 / 1.5, 1 / 3]
XY_WINDOW = (64, 64)
STRIDE = (16, 16)
N_JOBS = 4

if __name__ == '__main__':
    with open('../models/svm_best_all_train.p', 'rb') as f:
        clf = pickle.load(f)

    fig, axis = plt.subplots(2, 3)
    for i in tqdm(range(0, 6)):
        img = imread('../test_images/test%s.jpg' % (i + 1))
        cars = detect_cars_multi_scale(img, clf, XY_WINDOW, STRIDE, Y_START_STOPS, IMAGE_SIZE_FACTORS, heatmap=True,
                                       n_jobs=N_JOBS)

        blobs = []
        max_val = cars.max()
        if max_val > 0:
            cars = ((cars.astype(np.float32) / cars.max()) * 255).astype(np.uint8)

            blobs = blob_doh(cars, num_sigma=5, min_sigma=1, max_sigma=255, threshold=.005)
        cars = np.stack([cars, np.zeros_like(cars), np.zeros_like(cars)], axis=2)

        cars = cv2.resize(cars, (426, 240))
        img[:240, 1280-426:1280] = cars

        axis[i // 3, i % 3].imshow(img)
        axis[i // 3, i % 3].axis('off')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='g', linewidth=2, fill=False)
            axis[i // 3, i % 3].add_patch(c)

    plt.tight_layout()
    plt.show()
