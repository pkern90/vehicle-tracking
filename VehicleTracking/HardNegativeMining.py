import os
import pickle
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ImageUtils import multi_bb_intersection_over_box
from ImageUtils import slide_window, cut_out_windows
from scipy.misc import imread
from skimage.transform import pyramid_gaussian
from tqdm import tqdm

# specifies the threshold for overlap. Defines how much percent of the
# detected boxs area can be inside a car box without counting as car.
iob_thresh = 0.1
des_func_thresh = 1
xy_window = (64, 64)
stride = (32, 32)
max_pyramid_layer = 2
downscale = 1.5
# Number of frames to skip to prevent similar detections.
skip_frames = 10

image_tar_size = (720*2, 1280*2)
image_org_size = (1200, 1920)
scale_factor = np.array(image_tar_size) / np.array(image_org_size)

out_cnt = 0
output_dir = '../data/neg_mining/'
data_dir = '../data/udacity/object-detection-crowdai/'
pattern = "*.jpg"

if __name__ == '__main__':
    """
    This script uses a trained car classifier and applies it on a
    labeled data set (from Udacity) through a sliding window. False positive
    with a high confidence will be saved.
    """

    paths = []
    for dir, _, _ in os.walk(data_dir):
        paths.extend(glob(os.path.join(dir, pattern)))
    print('Images found: ', len(paths))

    labels = pd.read_csv('%slabels.csv' % data_dir)

    with open('../models/svm_best_all_train.p', 'rb') as f:
        clf = pickle.load(f)

    detections = np.empty((0, 4), dtype=np.uint32)
    paths = paths[::skip_frames]
    for path in tqdm(paths):
        # First get annotated bounding boxes from the data set for the current image
        file_name = path.split('/')[-1]
        img_labels = labels[labels['Frame'] == file_name]
        ann_boxes = img_labels[['xmin', 'xmax', 'ymin', 'ymax']].values

        # rescale boxes since org image size was full hd (1200,1920)
        ann_boxes[:, ::2] = (ann_boxes[:, ::2] * scale_factor[1]).astype(np.uint32)
        ann_boxes[:, 1::2] = (ann_boxes[:, 1::2] * scale_factor[0]).astype(np.uint32)

        # Use image pyramid to get detection with different sizes
        img = imread(path)
        img = cv2.resize(img, image_tar_size[::-1])
        pyramid = pyramid_gaussian(img, downscale=downscale, max_layer=max_pyramid_layer)
        for scale, img_scaled in enumerate(pyramid):
            img_scaled = (img_scaled * 255).astype(np.uint8)

            # check if search area is smaller then window.
            y_start_stop = [img_scaled.shape[0] * 0.30, img.shape[0] * 0.93]
            search_area_height = y_start_stop[1] - y_start_stop[0]
            if search_area_height < xy_window[1] or img_scaled.shape[1] < xy_window[0]:
                break

            windows = slide_window(img_scaled, y_start_stop=y_start_stop, xy_window=xy_window,
                                   stride=stride)

            samples = cut_out_windows(img_scaled, windows)
            des_funct = clf.decision_function(samples)

            windows = windows[(des_funct > des_func_thresh)]
            samples = samples[(des_funct > des_func_thresh)]

            # rescale annotated bb to fit the current image size
            if scale > 0:
                ann_boxes = (ann_boxes / downscale).astype(np.uint32)

            # select false positive
            mask = np.zeros(windows.shape[0])
            for i, win in enumerate(windows):
                iob = multi_bb_intersection_over_box(win, ann_boxes)
                mask[i] = (iob > iob_thresh).any()

            false_pos = windows[mask == 0]

            # save false positive as file
            if false_pos.shape[0] > 0:
                neg_samples = samples[mask == 0]
                for sample in neg_samples:
                    plt.imsave('%s%06d.png' % (output_dir, out_cnt), sample)
                    out_cnt += 1

                # Uncomment to display boxes
                # img = draw_boxes(img_scaled, ann_boxes, color=(0, 0, 255), thick=2)
                # img = draw_boxes(img, windows, color=(0, 255, 0), thick=2)
                # img = draw_boxes(img, false_pos, color=(255, 0, 0), thick=2)
                #
                # plt.imshow(img)
                # plt.show()
