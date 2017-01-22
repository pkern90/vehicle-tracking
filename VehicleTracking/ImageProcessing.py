import pickle

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
from scipy.misc import imread
from tqdm import tqdm

from VehicleTracking.ImageUtils import center_points, surrounding_box, detect_cars
from VehicleTracking.ImageUtils import draw_boxes

if __name__ == '__main__':
    # img = imread('../data/vehicles/GTI_Right/image0001.png')
    with open('../models/svm_hnm.p', 'rb') as f:
        clf = pickle.load(f)

    print(clf.get_params())
    fig, axis = plt.subplots(2, 3)
    for i in tqdm(range(0, 6)):
        img = imread('../test_images/test%s.jpg' % (i + 1))
        cars, detection_confidence = detect_cars(img, clf)
        img = draw_boxes(img, cars, color=(0, 0, 255))

        centers = center_points(cars)
        thresh = 128
        clusters = hcluster.fclusterdata(centers, thresh, criterion="distance")
        avg_boxes = surrounding_box(cars, clusters)
        img = draw_boxes(img, avg_boxes, color=(0, 255, 0))

        axis[i // 3, i % 3].imshow(img)
        axis[i // 3, i % 3].plot(centers[:, 0], centers[:, 1], 'x', color='r')
        axis[i // 3, i % 3].axis('off')

    # detections = non_max_suppression_fast(detections, detection_confidence, 0.0)
    # img = draw_boxes(img, cars)
    # plt.imshow(img)
    plt.show()
