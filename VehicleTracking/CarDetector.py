import multiprocessing

import cv2
import numpy as np
from ImageUtils import normalize, gaussian_blur, bb_by_contours
from joblib import Parallel
from joblib import delayed
from sklearn.pipeline import Pipeline

from VehicleTracking.Detection import Detection
from VehicleTracking.ImageUtils import cut_out_windows, hog_image, hog_feature_size
from VehicleTracking.ImageUtils import slide_window, convert_cspace

N_FRAMES = 7


class CarDetector:
    def __init__(self,
                 clf,
                 delete_after=5,
                 iou_thresh=0.5,
                 xy_window=(64, 64),
                 stride=None,
                 y_start_stops=None,
                 image_size_factors=[1],
                 n_jobs=1):

        if stride is None:
            stride = [(32, 32)]

        self.iou_thresh = iou_thresh
        self.n_jobs = n_jobs
        self.image_size_factors = image_size_factors
        self.y_start_stops = y_start_stops
        self.stride = stride
        self.xy_window = xy_window
        self.clf = clf
        self.detections = []
        self.delete_after = delete_after
        self.frame_cnt = 0
        self.heatmap = None

    def process_frame(self, img):
        out = np.zeros(img.shape, dtype=np.uint8)

        if self.heatmap is None:
            self.heatmap = np.zeros((img.shape[0], img.shape[1], N_FRAMES), dtype=np.float32)

        heat = detect_cars_multi_scale(img, self.clf, self.xy_window, self.stride, self.y_start_stops,
                                       self.image_size_factors, heatmap=True, n_jobs=self.n_jobs)

        debug = np.copy(heat)
        debug = normalize(debug, new_max=255, new_min=0, old_max=heat.max(), old_min=0, dtype=np.uint8)
        out[:360, :640] = cv2.resize(np.stack((debug, debug, debug), axis=2), (640, 360))

        self.heatmap[:, :, self.frame_cnt % N_FRAMES] = heat

        heat = np.mean(self.heatmap, axis=2)
        heat_thresh = np.zeros(heat.shape, dtype=np.uint8)
        heat_thresh[heat > 3] = 255

        debug = np.copy(heat_thresh)
        out[360:, 640:] = cv2.resize(np.stack((debug, debug, debug), axis=2), (640, 360))

        heat = normalize(heat, new_max=255, new_min=0, old_max=heat.max(), old_min=0, dtype=np.uint8)

        debug = np.copy(heat)
        out[360:, :640] = cv2.resize(np.stack((debug, debug, debug), axis=2), (640, 360))

        boxes_contours = bb_by_contours(heat_thresh)
        used_boxes = np.zeros(len(boxes_contours), np.uint8)
        for detection in self.detections:
            if boxes_contours is not None and len(boxes_contours) > 0:
                rd = detection.relative_distance_with(boxes_contours)
                min_rd = rd.min()
                argmin_rd = rd.argmin()
                if min_rd < self.iou_thresh:
                    if used_boxes[argmin_rd] == 1:
                        detection.is_hidden = True

                    detection.update(boxes_contours[argmin_rd])
                    used_boxes[argmin_rd] = 1
                else:
                    detection.update(None)
            else:
                detection.update(None)

        unused_boxes = boxes_contours[used_boxes == 0]
        if len(unused_boxes) > 0:
            hidden = [detection for detection in self.detections if detection.is_hidden]
            for detection in hidden:
                rd = detection.relative_distance_with(unused_boxes)
                min_rd = rd.min()
                argmin_rd = rd.argmin()
                ix = np.where(np.all(boxes_contours == unused_boxes[argmin_rd], axis=1))[0][0]
                if min_rd < 1.5 * self.iou_thresh:
                    detection.unhide(boxes_contours[ix])
                    used_boxes[ix] = 1

        for bb in boxes_contours[used_boxes == 0]:
            self.detections.append(Detection(bb))

        keep_detections = []
        img_contours = img
        nb_vehicles = 0
        for detection in self.detections:
            if detection.frames_undetected < self.delete_after:
                keep_detections.append(detection)
            if len(detection.last_boxes) > 5:
                nb_vehicles += 1
                img_contours = detection.draw(img_contours, thick=3, color=(255, 0, 0))

        self.detections = keep_detections

        # img_contours = draw_boxes(img, boxes_contours, thick=3, color=(255, 0, 0))
        out[:360, 640:] = cv2.resize(img_contours, (640, 360))

        self.frame_cnt += 1

        cv2.putText(out, 'Number of Vehicles %s' % nb_vehicles, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)

        return out


def detect_cars_multi_scale(img,
                            clf,
                            xy_window=(64, 64),
                            stride=None,
                            y_start_stops=None,
                            image_size_factors=[1],
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

    # use joblib to run processing in parallel
    result = Parallel(n_jobs=n_jobs)(
        delayed(detect_cars)(
            img,
            clf,
            xy_window,
            cur_stride,
            cur_sizes_factors,
            cur_y_start_stop)
        for cur_stride, cur_sizes_factors, cur_y_start_stop in
        zip(stride, image_size_factors, y_start_stops))

    bounding_boxes, des_func = zip(*result)
    bounding_boxes = np.vstack(bounding_boxes)
    des_func = np.concatenate(des_func)
    if heatmap:
        heat = np.zeros(img.shape[:2], dtype=np.float32)
        for bb, df in zip(bounding_boxes, des_func):
            heat[bb[1]:bb[3], bb[0]:bb[2]] += 1 #* df

        return heat

    return bounding_boxes


def detect_cars(img, clf, xy_window, stride, cur_sizes_factors, cur_y_start_stop):
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

    windows = slide_window(img_scaled, y_start_stop=cur_y_start_stop, xy_window=xy_window,
                           stride=stride)
    img_scaled_lab = convert_cspace(img_scaled, 'LAB')
    samples_lab = cut_out_windows(img_scaled_lab, windows)

    img_scaled_hls = convert_cspace(img_scaled, 'HLS')
    samples_hls = cut_out_windows(img_scaled_hls, windows)

    search_area_lab = img_scaled_lab[cur_y_start_stop[0]:cur_y_start_stop[1], :, :]

    transformers = {k: v for k, v in clf.named_steps['features'].transformer_list}
    hog_minmax = transformers['hog'].named_steps['hog_minmax']

    chist_transformer = transformers['chist']
    chist_transformer = Pipeline(chist_transformer.steps[2:])

    sb_transformer = transformers['sb']
    sb_transformer = Pipeline(sb_transformer.steps[2:])

    hog = hog_image(search_area_lab, 18, 8, 2)

    sy, sx = search_area_lab.shape[:2]
    cx, cy = 8, 8
    bx, by = 2, 2
    n_cellsx = int(np.floor(sx // cx))
    n_cellsy = int(np.floor(sy // cy))
    n_blocksx = (n_cellsx - bx) + 1 - 7
    n_blocksy = (n_cellsy - by) + 1 - 7
    n_stepsx = stride[0] // 8
    n_stepsy = stride[1] // 8
    hog_vectors = np.zeros((len(windows), samples_lab.shape[-1] * hog_feature_size(samples_lab[0], 8, 2, 18)))
    for i, ys in enumerate(range(n_blocksy)[::n_stepsy]):
        for j, xs in enumerate(range(n_blocksx)[::n_stepsx]):
            hog_vectors[i * n_blocksx // n_stepsx + j] = hog[:, ys:ys + 7, xs:xs + 7, :, :, :].ravel()

    hog_vectors = hog_minmax.transform(hog_vectors)
    sb_vectors = sb_transformer.transform(samples_lab)
    chist_vectors = chist_transformer.transform(samples_hls)

    features = np.concatenate((hog_vectors, chist_vectors, sb_vectors), axis=1)

    cls = clf.named_steps['clf']
    des_func = cls.decision_function(features)
    windows = windows[(des_func > 0)]

    windows = (windows / cur_sizes_factors).astype(np.uint32)
    return windows, des_func[des_func > 0]
