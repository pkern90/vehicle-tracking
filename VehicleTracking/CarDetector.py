import multiprocessing

import cv2
import numpy as np
from ImageUtils import normalize, bb_by_contours
from joblib import Parallel
from joblib import delayed
from sklearn.pipeline import Pipeline

from VehicleTracking.Detection import Detection
from VehicleTracking.ImageUtils import cut_out_windows, hog_image, hog_feature_size, gaussian_blur
from VehicleTracking.ImageUtils import slide_window, convert_cspace

import matplotlib.pyplot as plt

class CarDetector:
    def __init__(self,
                 clf,
                 delete_after=5,
                 dist_thresh=0.5,
                 xy_window=(64, 64),
                 stride=None,
                 y_start_stops=None,
                 x_padding=None,
                 image_size_factors=[1],
                 n_frames=1,
                 n_jobs=1):

        if stride is None:
            stride = [(32, 32)]

        self.dist_thresh = dist_thresh
        self.n_jobs = n_jobs
        self.image_size_factors = image_size_factors
        self.y_start_stops = y_start_stops
        self.stride = stride
        self.xy_window = xy_window
        self.clf = clf
        self.detections = []
        self.delete_after = delete_after
        self.frame_cnt = 0
        self.n_vehicles = 0
        self.n_frames = n_frames
        self.heatmap = None
        self.x_padding = x_padding

    def _remove_outliers(self, boxes):
        filtered_boxes = []
        for bc in boxes:
            w = bc[2] - bc[0]
            h = bc[3] - bc[1]
            if bc[1] < 450 and w > 32 and h > 32:
                filtered_boxes.append(bc)
            elif bc[1] > 450 and w > 64 and h > 64:
                filtered_boxes.append(bc)

        return np.array(filtered_boxes)

    def process_frame(self, img):
        out = np.zeros(img.shape, dtype=np.uint8)

        if self.heatmap is None:
            self.heatmap = np.zeros((img.shape[0], img.shape[1], self.n_frames), dtype=np.float32)

        img_blur = gaussian_blur(np.copy(img), 3)
        heat = detect_cars_multi_area(img_blur, self.clf, self.xy_window, self.stride, self.y_start_stops,
                                      self.image_size_factors, self.x_padding, heatmap=True, n_jobs=self.n_jobs)

        heat = gaussian_blur(heat, 21)
        self.heatmap[:, :, self.frame_cnt % self.n_frames] = heat

        if self.frame_cnt < self.n_frames:
            heat = np.mean(self.heatmap[:, :, :self.frame_cnt], axis=2)
        else:
            heat = np.mean(self.heatmap, axis=2)

        debug = normalize(heat, new_max=255, new_min=0, dtype=np.uint8)
        out[:360, :640] = cv2.resize(np.stack((debug, debug, debug), axis=2), (640, 360))

        heat_thresh = np.zeros(heat.shape, dtype=np.uint8)
        heat_thresh[heat > 2.5] = 255

        debug = np.copy(heat_thresh)
        out[360:, 640:] = cv2.resize(np.stack((debug, debug, debug), axis=2), (640, 360))

        boxes_contours = bb_by_contours(heat_thresh)
        boxes_contours = self._remove_outliers(boxes_contours)

        used_boxes = np.zeros(len(boxes_contours), np.bool)
        for detection in self.detections:
            if boxes_contours is not None and len(boxes_contours) > 0:
                rd = detection.relative_distance_with(boxes_contours)
                min_rd = rd.min()
                argmin_rd = rd.argmin()
                if min_rd < self.dist_thresh:
                    if used_boxes[argmin_rd]:
                        detection.is_hidden = True

                    detection.update(boxes_contours[argmin_rd])
                    used_boxes[argmin_rd] = True
                else:
                    detection.update(None)
            else:
                detection.update(None)

        unused_boxes = boxes_contours[used_boxes == False]
        if len(unused_boxes) > 0:
            hidden = [detection for detection in self.detections if detection.is_hidden]
            for detection in hidden:
                rd = detection.relative_distance_with(unused_boxes)
                min_rd = rd.min()
                argmin_rd = rd.argmin()
                ix = np.where(np.all(boxes_contours == unused_boxes[argmin_rd], axis=1))[0][0]
                if min_rd < 1.5 * self.dist_thresh:
                    detection.unhide(boxes_contours[ix])
                    used_boxes[ix] = True

        for bb in boxes_contours[used_boxes == False]:
            d = Detection(bb)
            self.detections.append(d)

        keep_detections = []
        img_contours = img
        self.n_vehicles = 0
        for detection in self.detections:
            if detection.frames_undetected < self.delete_after and \
                    not (detection.is_hidden and detection.age < 8):
                keep_detections.append(detection)

            if len(detection.last_boxes) > 8:
                self.n_vehicles += 1
                img_contours = detection.draw(img_contours, thick=3, color=(255, 0, 0))

        self.detections = keep_detections

        # img_contours = draw_boxes(img, boxes_contours, thick=3, color=(255, 0, 0))
        out[:360, 640:] = cv2.resize(img_contours, (640, 360))

        self.frame_cnt += 1

        cv2.putText(out, 'Vehicles in sight: %s' % self.n_vehicles, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)

        return out


def detect_cars_multi_area(img,
                           clf,
                           xy_window=(64, 64),
                           stride=None,
                           y_start_stops=None,
                           image_size_factors=[1],
                           x_padding=None,
                           heatmap=False,
                           n_jobs=1):
    """
    Detects cars on an image and returns the corresponding bounding boxes.

    :param img:
    :param clf: binary classifier. 1 = car and 0 = non car.
    :param xy_window: size of the window to use for sliding window.
    Has to be the same size then the training images of the classifier
    :param stride: stride of the sliding window. Can be different for every search area.
    :param y_start_stops: Limits of the search area on the y axis. Can be different for each search area.
    :param image_size_factors: Factor for scaling the image. Can be different for every search area.
    :param x_padding:
    :param heatmap: if set to true the function will return a heat map instead of the separate bounding boxes.
    :param n_jobs: Number of parallel jobs to run. n_jobs > number of search areas won't yield any performance
    improvements since only different search areas are processed in parallel.
    :return:
    """

    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count()

    if stride is None:
        stride = np.repeat([[xy_window[0], xy_window[1]]], len(image_size_factors), axis=0)

    if y_start_stops is None:
        y_start_stops = np.repeat([[0, img.shape[0] - 1]], len(image_size_factors), axis=0)

    if x_padding is None:
        x_padding = np.repeat([0], len(image_size_factors), axis=0)

    # use joblib to run processing in parallel
    result = Parallel(n_jobs=n_jobs)(
        delayed(detect_cars)(
            img,
            clf,
            xy_window,
            cur_stride,
            cur_sizes_factors,
            cur_y_start_stop,
            cur_x_padding)
        for cur_stride, cur_sizes_factors, cur_y_start_stop, cur_x_padding in
        zip(stride, image_size_factors, y_start_stops, x_padding))

    bounding_boxes, des_func = zip(*result)
    bounding_boxes = np.vstack(bounding_boxes)

    des_func = np.concatenate(des_func)

    if heatmap:
        heat = np.zeros(img.shape[:2], dtype=np.float32)
        for bb, df in zip(bounding_boxes, des_func):
            heat[bb[1]:bb[3], bb[0]:bb[2]] += df

        return heat

    return bounding_boxes


def detect_cars(img, clf, xy_window, stride, cur_sizes_factors, cur_y_start_stop, cur_x_padding):
    """
    Detects cars on an image.
    :param img:
    :param clf: binary classifier. 1 = car and 0 = non car.
    :param xy_window: size of the window to use for sliding window.
    Has to be the same size then the training images of the classifier
    :param stride: stride of the sliding window.
    :param cur_sizes_factors: Factor for scaling the image.
    :param cur_y_start_stop: Limits of the search area on the y axis.
    :return: Bounding boxes for all detected cars
    """

    image_tar_size = (int(img.shape[0] * cur_sizes_factors),
                      int(img.shape[1] * cur_sizes_factors))

    # open cv needs the shape in reversed order (width, height)
    img_scaled = cv2.resize(img, image_tar_size[::-1])

    # check if search area is smaller than window.
    cur_y_start_stop = (cur_y_start_stop * cur_sizes_factors).astype(np.uint32)

    search_area_height = cur_y_start_stop[1] - cur_y_start_stop[0]
    if search_area_height < xy_window[1] or img_scaled.shape[1] < xy_window[0]:
        return np.ndarray((0, 4))

    w = img_scaled.shape[1] + cur_x_padding * 2
    img_with_padding = np.zeros((img_scaled.shape[0], w, 3), dtype=img_scaled.dtype)
    img_with_padding[:, cur_x_padding:img_scaled.shape[1] + cur_x_padding] = img_scaled
    img_scaled = img_with_padding

    windows = slide_window(img_scaled, y_start_stop=cur_y_start_stop, xy_window=xy_window,
                           stride=stride)

    img_scaled_lab = convert_cspace(img_scaled, 'LAB')
    samples_lab = cut_out_windows(img_scaled_lab, windows)

    img_scaled_hls = convert_cspace(img_scaled, 'HLS')
    samples_hls = cut_out_windows(img_scaled_hls, windows)
    search_area_lab = img_scaled_lab[cur_y_start_stop[0]:cur_y_start_stop[1], :, :]

    transformers = {k: v for k, v in clf.named_steps['features'].transformer_list}

    chist_transformer = transformers['chist']
    chist_transformer = Pipeline(chist_transformer.steps[2:])

    sb_transformer = transformers['sb']
    sb_transformer = Pipeline(sb_transformer.steps[2:])

    hog_vectors = get_hog_vector(search_area_lab, transformers['hog'], xy_window[0], stride)
    sb_vectors = sb_transformer.transform(samples_lab)
    chist_vectors = chist_transformer.transform(samples_hls)

    features = np.concatenate((hog_vectors, chist_vectors, sb_vectors), axis=1)

    cls = clf.named_steps['clf']
    des_func = cls.decision_function(features)

    windows = (windows / cur_sizes_factors).astype(np.uint32)

    windows = windows[des_func > 0]
    des_func = des_func[des_func > 0]

    windows[windows[:, 0] < cur_x_padding] = cur_x_padding
    windows[:, 0] -= cur_x_padding
    windows[:, 2] -= cur_x_padding

    return windows, des_func


def get_hog_vector(search_area, hog_transformer, win_size, stride):
    """
    Applies hog transformation on the complete search area and extracts windows afterwards.
    The features for each window will be returned as a one dimensional vector stacked in a numpy array.
    :param search_area:
    :param hog_transformer:
    :param win_size:
    :param stride:
    :return:
    """

    hog_minmax = hog_transformer.named_steps['hog_minmax']
    hog_extractor = hog_transformer.named_steps['hog_extractor']

    orient = hog_extractor.orient
    pix_per_cell = hog_extractor.pix_per_cell
    cells_per_block = hog_extractor.cells_per_block

    hog = hog_image(search_area, orient, pix_per_cell, cells_per_block)

    span_y, span_x = search_area.shape[:2]
    n_blocks_per_window = int(np.floor(win_size // pix_per_cell)) - cells_per_block + 1
    n_cells_x = span_x // pix_per_cell
    n_cells_y = span_y // pix_per_cell
    n_blocks_x = n_cells_x - cells_per_block + 1 - n_blocks_per_window
    n_blocks_y = n_cells_y - cells_per_block + 1 - n_blocks_per_window
    n_steps_x = stride[0] // pix_per_cell
    n_steps_y = stride[1] // pix_per_cell
    n_windows = (n_blocks_x // n_steps_x + 1) * (n_blocks_y // n_steps_y + 1)

    hog_vectors = np.zeros((n_windows, search_area.shape[-1] * hog_feature_size(win_size,
                                                                                pix_per_cell,
                                                                                cells_per_block,
                                                                                orient)))

    for y, block_y in enumerate(range(n_blocks_y)[::n_steps_y]):
        for x, block_x in enumerate(range(n_blocks_x)[::n_steps_x]):
            ix = y * n_blocks_x // n_steps_x + x
            hog_vectors[ix] = hog[:, block_y:block_y + n_blocks_per_window,
                              block_x:block_x + n_blocks_per_window, :, :, :].ravel()

    return hog_minmax.transform(hog_vectors)
