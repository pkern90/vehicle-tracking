import cv2
import numpy as np
from ImageUtils import normalize, gaussian_blur, bb_by_contours, detect_cars_multi_scale

from VehicleTracking.Detection import Detection

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
            self.heatmap = np.zeros((img.shape[0], img.shape[1], N_FRAMES), dtype=np.uint8)

        heat = detect_cars_multi_scale(img, self.clf, self.xy_window, self.stride, self.y_start_stops,
                                       self.image_size_factors, heatmap=True, n_jobs=self.n_jobs)

        debug = np.copy(heat)
        debug = normalize(debug, new_max=255, new_min=0, old_max=heat.max(), old_min=0, dtype=np.uint8)
        out[:360, :640] = cv2.resize(np.stack((debug, debug, debug), axis=2), (640, 360))

        self.heatmap[:, :, self.frame_cnt % N_FRAMES] = heat

        heat = np.mean(self.heatmap, axis=2)
        heat_thresh = np.zeros(heat.shape, dtype=np.uint8)
        heat_blur = gaussian_blur(heat, 21)
        heat_thresh[heat_blur > 3] = 255

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
