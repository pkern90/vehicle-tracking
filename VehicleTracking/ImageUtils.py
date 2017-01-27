import multiprocessing

import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from scipy.misc import imread
from skimage.feature import blob_doh
from skimage.feature import hog
from skimage.feature import peak_local_max
from skimage.morphology import watershed


def create_bb_comparision(img, heat):
    boxes_contours = bb_by_contours(heat)
    boxes_blob = bb_by_blob_doh(heat)
    boxes_blob_watershed = bb_by_blob_doh_watershed(heat)
    boxes_watershed = bb_by_watershed(heat)

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

    return img


def normalize(images, new_max=1., new_min=0., old_max=255, old_min=0, dtype=None):
    if dtype is None:
        dtype = images.dtype

    if images.dtype.kind == 'u':
        images = images.astype(np.float32)

    return ((images - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min).astype(dtype)


def gaussian_blur(img, kernel_size):
    """
    Applies a Gaussian Noise kernel
    :param img:
    :param kernel_size:
    :return:
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def bb_by_contours(heat):
    _, contours, _ = cv2.findContours(np.copy(heat), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = np.zeros((len(contours), 4), dtype=np.uint32)
    for j, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w < 32 or h < 32:
            continue

        boxes[j] = [x, y, x + w, y + h]

    return boxes


def bb_by_blob_doh(heat):
    try:
        blobs = blob_doh(heat, num_sigma=4, min_sigma=1, max_sigma=255, threshold=.01)
    except IndexError:
        return np.empty((0, 4), dtype=np.uint32)

    blobs = blobs[blobs[:, 2] > 16]
    boxes = np.zeros((len(blobs), 4), dtype=np.uint32)
    for i, blob in enumerate(blobs):
        blob = np.array(blob, dtype=np.int32)
        blob_area = np.array([max(blob[1] - blob[2], 0),
                              max(blob[0] - blob[2], 0),
                              min(blob[1] + blob[2], heat.shape[1] - 1),
                              min(blob[0] + blob[2], heat.shape[0] - 1)])
        sec = heat[blob_area[1]:blob_area[3], blob_area[0]:blob_area[2]]
        filled = sec.nonzero()

        box = np.array([[filled[1].min(), filled[0].min(), filled[1].max(), filled[0].max()]])

        box[0, :2] += np.expand_dims(blob_area, axis=0)[0, :2]
        box[0, 2:] += np.expand_dims(blob_area, axis=0)[0, :2]
        boxes[i] = box

    return boxes


def bb_by_blob_doh_watershed(heat):
    try:
        blobs = blob_doh(heat, num_sigma=4, min_sigma=1, max_sigma=255, threshold=.01)
    except IndexError:
        return np.empty((0, 4), dtype=np.uint32)

    blobs = blobs.astype(np.uint32)
    blobs = blobs[blobs[:, 2] > 16]
    centroids = np.zeros(heat.shape, dtype=np.bool)
    centroids[blobs[:, 0], blobs[:, 1]] = True

    markers = ndi.label(centroids)[0]
    labels = watershed(-heat, markers, mask=heat)

    boxes = np.zeros((len(np.unique(labels)) - 1, 4), dtype=np.uint32)
    for i, label in enumerate(np.unique(labels)):
        if label == 0:
            continue
        filled = np.where(labels == label)
        boxes[i - 1] = np.array([[filled[1].min(), filled[0].min(), filled[1].max(), filled[0].max()]])

    return boxes


def bb_by_watershed(heat):
    local_maxi = peak_local_max(heat,
                                indices=False,
                                footprint=np.ones((10, 10)),
                                labels=np.copy(heat))

    markers = ndi.label(local_maxi)[0]
    labels = watershed(-heat, markers, mask=heat)

    boxes = np.zeros((len(np.unique(labels)) - 1, 4), dtype=np.uint32)
    for i, label in enumerate(np.unique(labels)):
        if label == 0:
            continue
        filled = np.where(labels == label)
        boxes[i - 1] = np.array([[filled[1].min(), filled[0].min(), filled[1].max(), filled[0].max()]])

    return boxes


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

    cut_outs = np.zeros((len(windows), height, width, img.shape[-1]), dtype=np.uint8)
    for i in range(len(windows)):
        cut_outs[i] = cut_out_window(img, windows[i])

    return cut_outs


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


def hog_features_(img, orient, pix_per_cell, cells_per_block, vis=False):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)

    img = img.astype(np.uint8)

    features = np.zeros((img.shape[2], hog_feature_size(img, pix_per_cell, cells_per_block, orient)))

    win_size = (img.shape[1] // pix_per_cell * pix_per_cell, img.shape[0] // pix_per_cell * pix_per_cell)
    block_size_px = (cells_per_block * pix_per_cell, cells_per_block * pix_per_cell)

    block_stride = (8, 8)
    deriv_aperture = 1
    win_sigma = 1.
    histogram_norm_type = 0
    l2_hys_threshold = 0.2
    gamma_correction = True
    nlevels = 64

    hog = cv2.HOGDescriptor(_winSize=win_size,
                            _blockSize=block_size_px,
                            _blockStride=block_stride,
                            _cellSize=(pix_per_cell, pix_per_cell),
                            _nbins=orient,
                            # _derivAperture=derivAperture,
                            # _winSigma=winSigma,
                            #_histogramNormType=histogram_norm_type,
                            # _L2HysThreshold=L2HysThreshold,
                            _gammaCorrection=gamma_correction
                            )

    for ch in range(img.shape[2]):
        hog_result = hog.compute(img[:, :, ch])[0]

        features[ch] = hog_result

    features = features.ravel()
    return features


def color_hist(img, bins=32, bins_range=(0, 256)):
    hists = np.zeros((img.shape[-1], bins))
    for ch in range(img.shape[-1]):
        hists[ch] = np.histogram(img[:, :, ch], bins=bins, range=bins_range)[0]

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
    elif cspace == 'YCrCb':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
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


def center_points(boxes):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    width = x2 - x1
    height = y2 - y1
    x = x1 + width // 2
    y = y1 + height // 2

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


def detect_cars_multi_scale(img,
                            clf,
                            xy_window=(64, 64),
                            stride=None,
                            y_start_stops=None,
                            image_size_factors=[1],
                            heatmap=False,
                            n_jobs=1):
    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count()

    if stride is None:
        stride = np.repeat([[xy_window[0], xy_window[1]]], len(image_size_factors), axis=0)

    if y_start_stops is None:
        y_start_stops = np.repeat([[0, img.shape[0] - 1]], len(image_size_factors), axis=0)

    bounding_boxes = Parallel(n_jobs=n_jobs)(
        delayed(detect_cars)(
            img,
            clf,
            xy_window,
            cur_stride,
            cur_sizes_factors,
            cur_y_start_stop)
        for cur_stride, cur_sizes_factors, cur_y_start_stop in
        zip(stride, image_size_factors, y_start_stops))

    bounding_boxes = np.vstack(bounding_boxes)
    if heatmap:
        heat = np.zeros(img.shape[:2], dtype=np.uint8)
        for bb in bounding_boxes:
            heat[bb[1]:bb[3], bb[0]:bb[2]] += 1

        return heat

    return bounding_boxes


def detect_cars(img, clf, xy_window, stride, cur_sizes_factors, cur_y_start_stop):
    image_tar_size = (int(img.shape[0] * cur_sizes_factors),
                      int(img.shape[1] * cur_sizes_factors))

    # open cv needs the shape in reversed order (width, height)
    img_scaled = cv2.resize(img, image_tar_size[::-1])

    # check if search area is smaller then window.
    cur_y_start_stop = cur_y_start_stop * cur_sizes_factors

    search_area_height = cur_y_start_stop[1] - cur_y_start_stop[0]
    if search_area_height < xy_window[1] or img_scaled.shape[1] < xy_window[0]:
        return np.ndarray((0, 4))

    windows = slide_window(img_scaled, y_start_stop=cur_y_start_stop, xy_window=xy_window,
                           stride=stride)

    samples = cut_out_windows(img_scaled, windows)
    des_funct = clf.decision_function(samples)
    windows = windows[(des_funct > 0)]

    windows = (windows / cur_sizes_factors).astype(np.uint32)
    return windows


def are_overlapping(box, other_boxes):
    box = box.astype(np.int32)
    other_boxes = other_boxes.astype(np.int32)
    si = np.maximum(0, np.minimum(box[2], other_boxes[:, 2]) - np.maximum(box[0], other_boxes[:, 0])) * \
         np.maximum(0, np.minimum(box[3], other_boxes[:, 3]) - np.maximum(box[1], other_boxes[:, 1]))

    return si > 0


def multi_bb_intersection_over_box(box, other_boxes):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(box[0], other_boxes[:, 0])
    yA = np.maximum(box[1], other_boxes[:, 1])
    xB = np.minimum(box[2], other_boxes[:, 2])
    yB = np.minimum(box[3], other_boxes[:, 3])

    w = np.abs(xB - xA + 1)
    h = np.abs(yB - yA + 1)

    # compute the area of intersection rectangle
    interArea = (w * h).astype(np.float32)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / boxAArea.astype(np.float32)
    # return the intersection over union value
    iou[~are_overlapping(box, other_boxes)] = 0
    return iou
