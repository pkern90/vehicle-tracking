import pickle

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
from scipy.misc import imread
from tqdm import tqdm
import numpy as np
from ImageUtils import center_points, surrounding_box, detect_cars_multi_scale
from ImageUtils import draw_boxes

Y_START_STOPS = np.array([[400, 496],
                          [400, 496],
                          [400, 592],
                          [368, 688]])
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
        cars = detect_cars_multi_scale(img, clf, XY_WINDOW, STRIDE, Y_START_STOPS, IMAGE_SIZE_FACTORS, n_jobs=N_JOBS)
        centers = None
        if len(cars) > 0:
            img = draw_boxes(img, cars, color=(0, 0, 255))

            centers = center_points(cars)
            thresh = 128
            if len(centers) > 1:
                clusters = hcluster.fclusterdata(centers, thresh, criterion="distance")
                avg_boxes, groups = surrounding_box(cars, clusters)
                img = draw_boxes(img, avg_boxes, color=(0, 255, 0))

        axis[i // 3, i % 3].imshow(img)
        if centers is not None:
            axis[i // 3, i % 3].plot(centers[:, 0], centers[:, 1], 'x', color='r')
        axis[i // 3, i % 3].axis('off')

    plt.tight_layout()
    plt.show()
