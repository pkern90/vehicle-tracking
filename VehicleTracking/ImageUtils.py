import cv2
import numpy as np
from scipy.misc import imread
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thick)
    # Return the image copy with boxes drawn
    return draw_img


def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel()

    return features


def cut_out_window(img, window):
    return img[window[1]:window[3], window[0]:window[2], :]


def cut_out_windows(img, windows):
    height = windows[0][3] - windows[0][1]
    width = windows[0][2] - windows[0][0]

    cut_outs = np.zeros((len(windows), height, width, img.shape[-1]), dtype=np.float32)
    for i in range(len(windows)):
        cut_outs[i] = cut_out_window(img, windows[i])

    return cut_outs.astype(np.uint8)


# Define a function to return HOG features and visualization
def hog_features(img, orient, pix_per_cell, cells_per_block, vis=False):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)

    features = np.zeros((img.shape[2], hog_feature_size(img, pix_per_cell, cells_per_block, orient)))

    if vis:
        hog_image = np.zeros(img.shape, dtype=np.float32)

    for ch in range(img.shape[2]):
        hog_result = hog(img[:, :, ch], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                         cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=True,
                         visualise=vis, feature_vector=True)

        if vis:
            features[ch] = hog_result[0]
            hog_image[:, :, ch] = hog_result[1]
        else:
            features[ch] = hog_result

    features = features.ravel()

    if vis:
        return features, hog_image
    else:
        return features


def color_hist(img, bins=32, bins_range=(0, 256)):
    hists = np.zeros((bins * img.shape[-1]))
    for ch in range(img.shape[-1]):
        hists[ch] = np.histogram(img[:, :, ch], bins=bins, range=bins_range)

    return hists.ravel()


def convert_cspace(img, cspace='RGB'):
    if cspace == 'RGB':
        img = np.copy(img)
    elif cspace == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif cspace == 'LUV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif cspace == 'HLS':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif cspace == 'YUV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif cspace == 'LAB':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    elif cspace == 'GRAY':
        img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), axis=-1)
    else:
        raise ValueError('Unknown color space. Please choose one of RGB, HSV, LUV, HLS, YUV, LAB or GRAY')

    return img


def hog_feature_size(image, pixels_per_cell, cells_per_block, orient):
    s = image.shape[0]
    n_cells = int(np.floor(s // pixels_per_cell))
    n_blocks = (n_cells - cells_per_block) + 1

    return n_blocks ** 2 * cells_per_block ** 2 * orient


# def extract_features(images, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256), orient=9,
#                      pix_per_cell=8, cells_per_block=2, hog_channel=0, normalize=True, pca=None):
#     spatials = np.zeros((len(images), spatial_size[0] * spatial_size[1] * 3))
#     hists = np.zeros((len(images), hist_bins * 3))
#     hogs = np.zeros((len(images), hog_feature_size(images[0], pix_per_cell, cells_per_block, orient)))
#
#     for i, img in enumerate(images):
#         feature_image = convert_cspace(img, cspace)
#
#         spatials[i] = bin_spatial(feature_image, size=spatial_size)
#         hists[i] = color_hist(feature_image, bins=hist_bins, bins_range=hist_range)
#         hogs[i] = hog_features(feature_image[:, :, hog_channel], orient, pix_per_cell, cells_per_block, vis=False)
#
#     if normalize:
#         spatials = StandardScaler().fit_transform(spatials)
#         hists = StandardScaler().fit_transform(hists)
#         hogs = StandardScaler().fit_transform(hogs)
#
#     if pca and type(pca) is int:
#         pca_spatials = PCA(n_components=pca)
#         spatials = pca_spatials.fit_transform(spatials)
#         pca_hists = PCA(n_components=pca)
#         hists = pca_hists.fit_transform(hists)
#         pca_hogs = PCA(n_components=pca)
#         hogs = pca_hogs.fit_transform(hogs)
#
#         features = np.concatenate((spatials, hists, hogs), axis=1)
#         return features, (pca_spatials, pca_hists, pca_hogs)
#
#     elif pca and (type(pca) is list or type(pca) is tuple):
#         spatials = pca[0].transform(spatials)
#         hists = pca[1].transform(hists)
#         hogs = pca[2].transform(hogs)
#
#     features = np.concatenate((spatials, hists, hogs), axis=1)
#     return features
#

def load_and_resize_images(paths, resize):
    images = np.zeros((len(paths), *resize, 3), dtype=np.uint8)
    for i, path in enumerate(paths):
        img = imread(path)
        img = cv2.resize(img, resize[::-1])
        images[i] = img

    return images





def slide_window(img, x_start_stop=None, y_start_stop=None,
                 xy_window=(64, 64), stride=(32, 32)):
    # If x and/or y start/stop positions not defined, set to image size
    if y_start_stop is None:
        y_start_stop = [None, None]
    if x_start_stop is None:
        x_start_stop = [None, None]
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = stride[0]
    ny_pix_per_step = stride[0]
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    windows = np.zeros((nx_windows * ny_windows, 4), dtype=np.uint32)
    invalid_win = []

    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            if endx >= img.shape[1]:
                endx = img.shape[1] - 1
                startx = endx - xy_window[0]

            if endy >= img.shape[0]:
                endy = img.shape[0] - 1
                starty = endy - xy_window[1]

            windows[ys * nx_windows + xs] = [startx, starty, endx, endy]

    return np.delete(windows, invalid_win, 0)


# http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# Malisiewicz et al.
def non_max_suppression_fast(boxes, confidence, overlap_thresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype(np.float32)

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(confidence)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def are_overlapping(box, other_boxes):
    box = box.astype(np.int32)
    other_boxes = other_boxes.astype(np.int32)
    si = np.maximum(0, np.minimum(box[2], other_boxes[:, 2]) - np.maximum(box[0], other_boxes[:, 0])) * \
         np.maximum(0, np.minimum(box[3], other_boxes[:, 3]) - np.maximum(box[1], other_boxes[:, 1]))

    return si > 0


def center_points(boxes):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    width = x2-x1
    height = y2-y1
    x = x1 + width//2
    y = y1 + height//2

    return np.stack((x, y)).T


def average_clusters(boxes, clusters):
    unique_groups = np.unique(clusters)
    avg_boxes = np.zeros((len(unique_groups), 4))
    for i, cluster in enumerate(unique_groups):
        avg_boxes[i] = np.mean(boxes[clusters == cluster], axis=0, keepdims=False)

    return np.rint(avg_boxes).astype(np.uint32), unique_groups


def surrounding_box(boxes, clusters):
    unique_groups = np.unique(clusters)
    sur_boxes = np.zeros((len(unique_groups), 4))
    for i, cluster in enumerate(unique_groups):
        sur_boxes[i, :2] = np.min(boxes[:, :2][clusters == cluster], axis=0, keepdims=False)
        sur_boxes[i, 2:] = np.max(boxes[:, 2:][clusters == cluster], axis=0, keepdims=False)

    return np.rint(sur_boxes).astype(np.uint32), unique_groups


def detect_cars(img, clf):
    y_start_stop = np.array([[400, 496],
                             [400, 592],
                             [368, 688]])

    xy_window = (64, 64)
    stride = (32, 32)
    max_pyramid_layer = 2

    detections = np.empty((0, 4), dtype=np.uint32)
    detection_confidence = None

    pyramid = pyramid_gaussian(img, downscale=2, max_layer=max_pyramid_layer)
    for scale, img_scaled in enumerate(pyramid):
        img_scaled = (img_scaled * 255).astype(np.uint8)
        #img_scaled = convert_cspace(img_scaled, cspace)

        # check if search area is smaller then window.
        scale_factor = 2 * scale if scale > 0 else 1
        cur_y_start_stop = y_start_stop[scale] / scale_factor

        search_area_height = cur_y_start_stop[1] - cur_y_start_stop[0]
        if search_area_height < xy_window[1] or img_scaled.shape[1] < xy_window[0]:
            break

        windows = slide_window(img_scaled, y_start_stop=cur_y_start_stop, xy_window=xy_window,
                               stride=stride)

        samples = cut_out_windows(img_scaled, windows)

        features = clf.named_steps['feature_extractor'].transform(samples)
        prediction = clf.named_steps['clf'].decision_function(features)
        most_likely = prediction > -.2
        windows = windows[most_likely]

        windows *= scale_factor
        detections = np.append(detections, windows, axis=0)
        if detection_confidence is None:
            detection_confidence = prediction[most_likely]
        else:
            detection_confidence = np.append(detection_confidence, prediction[most_likely], axis=0)

    return detections, detection_confidence