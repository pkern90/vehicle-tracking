import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ImageUtils import detect_cars_multi_scale, gaussian_blur, normalize, bb_by_contours, bb_by_blob_doh, \
    bb_by_blob_doh_watershed, bb_by_watershed
from ImageUtils import draw_boxes
from scipy.misc import imread
from tqdm import tqdm

Y_START_STOPS = np.array([[400, 464],
                          [400, 592],
                          [400, 688],
                          [400, 688]])
IMAGE_SIZE_FACTORS = [1.5, 1, 1 / 1.5, 1 / 3]
XY_WINDOW = (64, 64)
STRIDE = (12, 12)
N_JOBS = 4

if __name__ == '__main__':
    with open('../models/svm_best_all_train.p', 'rb') as f:
        clf = pickle.load(f)

    fig, axis = plt.subplots(2, 3)
    for i in tqdm(range(0, 1)):
        img = imread('../test_images/test%s.jpg' % (i + 1))
        heat = detect_cars_multi_scale(img, clf, XY_WINDOW, STRIDE, Y_START_STOPS, IMAGE_SIZE_FACTORS, heatmap=True,
                                       n_jobs=N_JOBS)

        heat = normalize(heat, new_max=255, new_min=0, old_max=heat.max(), old_min=0)
        heat_thresh = np.zeros(heat.shape, dtype=np.uint8)
        heat_blur = gaussian_blur(heat, 21)
        heat_thresh[heat_blur > 50] = 255

        boxes_contours = bb_by_contours(heat_thresh)
        boxes_blob = bb_by_blob_doh(heat_thresh)
        boxes_blob_watershed = bb_by_blob_doh_watershed(heat_thresh)
        boxes_watershed = bb_by_watershed(heat_thresh)

        img_contours = draw_boxes(img, boxes_contours, thick=3, color=(255, 0, 0))
        img_blob = draw_boxes(img, boxes_blob, thick=3, color=(255, 0, 0))
        img_blob_watershed = draw_boxes(img, boxes_blob_watershed, thick=3, color=(255, 0, 0))
        img_watershed = draw_boxes(img, boxes_watershed, thick=3, color=(255, 0, 0))

        cv2.putText(img_contours, "Find Contours", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=5)
        cv2.putText(img_blob, "Blob DoH", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=5)
        cv2.putText(img_blob_watershed, "Blob DoH + Watershed", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
                    thickness=5)
        cv2.putText(img_watershed, "Watershed", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=5)

        img[:360, :640] = cv2.resize(img_contours, (640, 360))
        img[360:, :640] = cv2.resize(img_blob, (640, 360))
        img[360:, 640:] = cv2.resize(img_blob_watershed, (640, 360))
        img[:360, 640:] = cv2.resize(img_watershed, (640, 360))

        axis[i // 3, i % 3].imshow(img)
        axis[i // 3, i % 3].axis('off')

    plt.tight_layout()
    plt.show()
