import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.misc import imread
from skimage.feature import blob_doh
from skimage.feature import hog
from skimage.feature import peak_local_max
from skimage.morphology import watershed


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def create_bb_comparision(heat):
    """
    Creates a comparison image displaying results of four different segmentation algorithms for
    bounding boxes. The methods used include:

    Watershed
    Blob detection (Determinant of Hessian)
    Watershed + Blob doh
    OpenCV findContours

    :param heat: one channel heat map. 0 is treated as background.
    :return:
    """

    img = np.zeros(heat.shape)

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


def normalize(images, new_max=1., new_min=0., old_max=None, old_min=None, dtype=None):
    """
    Normalizes images to a specified range

    :param images:
    :param new_max:
    :param new_min:
    :param old_max:
    :param old_min:
    :param dtype: dtype for the output image. If not specified, the input type will be used.
    :return: normalized image
    """

    if old_max is None:
        old_max = images.max()

    if old_min is None:
        old_min = images.min()

    if dtype is None:
        dtype = images.dtype

    if images.dtype.kind == 'u':
        images = images.astype(np.float32)

    return ((images - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min).astype(dtype)


def gaussian_blur(img, kernel_size=1):
    """
    Applies a Gaussian Noise kernel
    :param img:
    :param kernel_size:
    :return:
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def bb_by_contours(heat):
    """
    Detects bounding boxes on a heatmap. Uses OpenCV findContours internally.

    :param heat: one channel heat map. 0 is treated as background.
    :return: numpy array of bounding boxes
    """

    _, contours, _ = cv2.findContours(np.copy(heat), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for j, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x + w, y + h])

    return np.array(boxes)


def bb_by_blob_doh(heat):
    """
    Detects bounding boxes on a heatmap.
    Uses Skimage blob detection (Determinant of Hessian) internally.

    :param heat: one channel heat map. 0 is treated as background.
    :return: numpy array of bounding boxes
    """

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
    """
    Detects bounding boxes on a heat map. Uses Skimage blob
    detection (Determinant of Hessian) to find centroids and watershed for labeling internally.

    :param heat: one channel heatmap. 0 is treated as background.
    :return: numpy array of bounding boxes
    """

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
    """
    Detects bounding boxes on a heatmap.
    Uses Skimage watershed internally.

    :param heat: one channel heatmap. 0 is treated as background.
    :return: numpy array of bounding boxes
    """

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
    """
    Draws multiple bounding boxes onto an image.

    :param img: image to draw on
    :param bboxes: numpy array of boxes with each row containing:
     [left_upper_x, left_upper_y, right_lower_x, right_lower_y]
    :param color: (R, G, B)
    :param thick: thickness in pixels
    :return: copy of the input image with boxes drawn onto.
    """

    draw_img = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(draw_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thick)

    return draw_img


def bin_spatial(img, size=(32, 32)):
    """
    Creates a feature vector of an image by spatially binning the pixels.

    :param img:
    :param size: tuple of (width, height)
    :return: array with n elements where n = size[0] * size[1]
    """
    return cv2.resize(img, size).ravel()


def cut_out_window(img, window):
    """
    Extracts a window of an image.

    :param img:
    :param window: [left_upper_x, left_upper_y, right_lower_x, right_lower_y]
    :return:
    """

    return img[window[1]:window[3], window[0]:window[2], :]


def cut_out_windows(img, windows):
    """
    Extracts multiple windows from one image.
    :param img:
    :param windows: nx4 numpy array with n = number of windows and each window like
        [left_upper_x, left_upper_y, right_lower_x, right_lower_y]
    :return:
    """

    height = windows[0][3] - windows[0][1]
    width = windows[0][2] - windows[0][0]

    cut_outs = np.zeros((len(windows), height, width, img.shape[-1]), dtype=np.uint8)
    for i in range(len(windows)):
        cut_outs[i] = cut_out_window(img, windows[i])

    return cut_outs


def hog_features(img, orient=9, pix_per_cell=8, cells_per_block=2, vis=False):
    """
    Creates a feature vector of an image through hog (histogram of oriented gradients).

    :param img:
    :param orient: number of orientation bins
    :param pix_per_cell: Cell size in pixel. Takes a scalar value which is used as width and height
    :param cells_per_block: Block size in cells. Takes a scalar value which is used as width and height
    :param vis: if true also returns a image visualizing the hog
    :return: Feature vector when viz is false. Else A tuple of (feature vector, visualization image)
    """

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


def hog_image(img, orient=9, pix_per_cell=8, cells_per_block=2):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)

    n_cells_x = int(np.floor(img.shape[1] // pix_per_cell))
    n_cells_y = int(np.floor(img.shape[0] // pix_per_cell))
    n_blocks_x = (n_cells_x - cells_per_block) + 1
    n_blocks_y = (n_cells_y - cells_per_block) + 1

    features = np.zeros((img.shape[2], n_blocks_y, n_blocks_x, cells_per_block, cells_per_block, orient))

    for ch in range(img.shape[2]):
        hog_result = hog(img[:, :, ch], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                         cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=True,
                         visualise=False, feature_vector=False)

        features[ch] = hog_result

    return features


def hog_features_opencv(img, block_size_px=16, block_stride=8, pix_per_cell=8, orient=9, win_sigma=1,
                        l2_hys_threshold=0.2, gamma_correction=True, window_size=64):
    """
    Creates a feature vector of an image through hog (histogram of oriented gradients). Uses OpenCV implementation.

    :param img:
    :param block_size_px: Block size in pixel. Takes a scalar value which is used as width and height
    :param block_stride: Stride of the blocks (overlap).
    Takes a scalar value which is used for horizontal and vertical stride
    :param pix_per_cell: Cell size in pixel. Takes a scalar value which is used as width and height
    :param orient: number of orientation bins
    :param win_sigma: Gaussian blur
    :param l2_hys_threshold:
    :param gamma_correction: Applies gamma correction of true.
    :param window_size: Window size in pixel. Takes a scalar value which is used as width and height
    :return: Feature vector
    """

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)

    img = img.astype(np.uint8)

    cells_per_block = block_size_px // pix_per_cell
    features = np.zeros((img.shape[2], hog_feature_size(img, pix_per_cell, cells_per_block, orient)))

    hog = cv2.HOGDescriptor(_winSize=(window_size, window_size),
                            _blockSize=(block_size_px, block_size_px),
                            _blockStride=(block_stride, block_stride),
                            _cellSize=(pix_per_cell, pix_per_cell),
                            _nbins=orient,
                            _winSigma=win_sigma,
                            _L2HysThreshold=l2_hys_threshold,
                            _gammaCorrection=gamma_correction)

    for ch in range(img.shape[2]):
        hog_result = hog.compute(img[:, :, ch])[0]

        features[ch] = hog_result

    features = features.ravel()
    return features


def color_hist(img, bins=32, bins_range=(0, 256)):
    """
    Creates a feature vector of an image based on a color histogramm
    :param img:
    :param bins:
    :param bins_range:
    :return:
    """

    hists = np.zeros((img.shape[-1], bins))
    for ch in range(img.shape[-1]):
        hists[ch] = np.histogram(img[:, :, ch], bins=bins, range=bins_range)[0]

    return hists.ravel()


def convert_cspace(img, cspace='RGB'):
    """
    Utility function for easier color conversion. Supports the following color spaces:

    RGB, HSV, LUV, HLS, YUV, LAB and GRAY

    :param img:
    :param cspace:
    :return:
    """

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


def hog_feature_size(img, pix_per_cell, cells_per_block, orient):
    """
    Calculates the size of a hog feature vector.

    :param img:
    :param orient: number of orientation bins
    :param pix_per_cell: Cell size in pixel. Takes a scalar value which is used as width and height
    :param cells_per_block: Block size in cells. Takes a scalar value which is used as width and height
    :return:
    """

    if type(img) is np.ndarray and 1 < len(img.shape) < 4:
        span_y, span_x = img.shape[0], img.shape[1]
    elif type(img) in (tuple, list) and len(img) == 2:
        span_y, span_x = img
    else:
        span_y, span_x = img, img

    n_cells_x = int(np.floor(span_x // pix_per_cell))
    n_cells_y = int(np.floor(span_y // pix_per_cell))
    n_blocks_x = (n_cells_x - cells_per_block) + 1
    n_blocks_y = (n_cells_y - cells_per_block) + 1

    return n_blocks_y * n_blocks_x * cells_per_block * cells_per_block * orient


def load_and_resize_images(paths, resize):
    """
    Loads multiple images from a given list of paths and directly resizes them to save memory.

    :param paths:
    :param resize: (height, width) of the target image size
    :return:
    """

    images = np.zeros((len(paths), *resize, 3), dtype=np.uint8)
    for i, path in enumerate(paths):
        img = imread(path)
        img = cv2.resize(img, resize[::-1])
        images[i] = img

    return images


def slide_window(img, x_start_stop=None, y_start_stop=None,
                 xy_window=(64, 64), stride=(32, 32)):
    """
    Creates bounding boxes for every position a sliding window will have in a specified area of a given image.

    :param img:
    :param x_start_stop: limit on the x axis
    :param y_start_stop: limits on the y axis
    :param xy_window: size of the window in pixel
    :param stride: stride of the window in pixel (x, y)
    :return:
    """

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
    ny_pix_per_step = stride[1]
    # Compute the number of windows in x/y
    nx_windows = np.int((xspan - xy_window[0]) / nx_pix_per_step) + 1
    ny_windows = np.int((yspan - xy_window[1]) / ny_pix_per_step) + 1
    # Initialize a list to append window positions to
    windows = np.zeros((nx_windows * ny_windows, 4), dtype=np.uint32)

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

    return windows


def center_points(boxes):
    """
    Calculates the center coordinates of multiple bounding boxes.

    :param boxes: [[left_upper_x, left_upper_y, right_lower_x, right_lower_y]]
    :return:
    """

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
    """
    Averages bounding boxes belonging to the same cluster

    :param boxes: boxes to average
    :param clusters: cluster labels for the bounding boxes
    :return: n bounding boxes where n = the number of unique clusters
    """

    unique_groups = np.unique(clusters)
    avg_boxes = np.zeros((len(unique_groups), 4))
    for i, cluster in enumerate(unique_groups):
        avg_boxes[i] = np.mean(boxes[clusters == cluster], axis=0, keepdims=False)

    return np.rint(avg_boxes).astype(np.uint32), unique_groups


def surrounding_box(boxes, clusters):
    """
    Calculates the surrounding bounding box of bounding boxes belonging to the same cluster

    :param boxes: boxes to surround
    :param clusters: cluster labels for the bounding boxes
    :return: n bounding boxes where n = the number of unique clusters
    """

    unique_groups = np.unique(clusters)
    sur_boxes = np.zeros((len(unique_groups), 4))
    for i, cluster in enumerate(unique_groups):
        sur_boxes[i, :2] = np.min(boxes[:, :2][clusters == cluster], axis=0, keepdims=False)
        sur_boxes[i, 2:] = np.max(boxes[:, 2:][clusters == cluster], axis=0, keepdims=False)

    return np.rint(sur_boxes).astype(np.uint32), unique_groups


def are_overlapping(box, other_boxes):
    """
    Detects if a given bounding box is overlapping with one or multiple other bounding boxes.
    :param box: [left_upper_x, left_upper_y, right_lower_x, right_lower_y]
    :param other_boxes: [[left_upper_x, left_upper_y, right_lower_x, right_lower_y]]
    :return: boolean vector indicating overlap
    """

    box = box.astype(np.int32)
    other_boxes = other_boxes.astype(np.int32)
    si = np.maximum(0, np.minimum(box[2], other_boxes[:, 2]) - np.maximum(box[0], other_boxes[:, 0])) * \
         np.maximum(0, np.minimum(box[3], other_boxes[:, 3]) - np.maximum(box[1], other_boxes[:, 1]))

    return si > 0


def multi_bb_intersection_over_box(box, other_boxes):
    """
    Calculates how much the given box is overlapping with each of the other given boxes.
    :param box: [left_upper_x, left_upper_y, right_lower_x, right_lower_y]
    :param other_boxes: [[left_upper_x, left_upper_y, right_lower_x, right_lower_y]]
    :return:
    """
    x_a = np.maximum(box[0], other_boxes[:, 0])
    y_a = np.maximum(box[1], other_boxes[:, 1])
    x_b = np.minimum(box[2], other_boxes[:, 2])
    y_b = np.minimum(box[3], other_boxes[:, 3])

    w = np.abs(x_b - x_a + 1)
    h = np.abs(y_b - y_a + 1)

    inter_area = (w * h).astype(np.float32)
    box_a_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

    iob = inter_area / box_a_area.astype(np.float32)

    iob[~are_overlapping(box, other_boxes)] = 0
    return iob


def multi_bb_intersection_over_union(box, other_boxes):
    """
    Calculates the intersect og union for a given box with each of the other given boxes.

    :param box: [left_upper_x, left_upper_y, right_lower_x, right_lower_y]
    :param other_boxes: [[left_upper_x, left_upper_y, right_lower_x, right_lower_y]]
    :return:
    """
    x_a = np.maximum(box[0], other_boxes[:, 0])
    y_a = np.maximum(box[1], other_boxes[:, 1])
    x_b = np.minimum(box[2], other_boxes[:, 2])
    y_b = np.minimum(box[3], other_boxes[:, 3])

    w = np.abs(x_b - x_a + 1)
    h = np.abs(y_b - y_a + 1)

    inter_area = (w * h).astype(np.float32)

    box_a_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    boxes_area = (other_boxes[:, 2] - other_boxes[:, 0] + 1) * (other_boxes[:, 3] - other_boxes[:, 1] + 1)

    iou = inter_area / (box_a_area + boxes_area - inter_area).astype(np.float32)

    iou[~are_overlapping(box, other_boxes)] = 0

    return iou


def relative_distance(box, o_box):
    """
    Calculates the average diagonal between the given box and each of the other boxes and
    puts it in relation to the distance of the center points.
    :param box: [left_upper_x, left_upper_y, right_lower_x, right_lower_y]
    :param o_box: [[left_upper_x, left_upper_y, right_lower_x, right_lower_y]]
    :return:
    """

    box = box.astype(np.float)
    o_box = o_box.astype(np.float)
    mean_diag = (np.sqrt(box[0] ** 2 + box[2] ** 2) + np.sqrt(o_box[:, 0] ** 2 + o_box[:, 2] ** 2)) / 2

    c_box = center_points(np.expand_dims(box, axis=0))[0]
    c_o_box = center_points(o_box)
    dist_centers = np.sqrt(np.power(c_box[0] - c_o_box[:, 0], 2) + np.power(c_box[1] - c_o_box[:, 1], 2))

    return dist_centers / mean_diag
