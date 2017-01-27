import cv2
import numpy as np

from ImageUtils import detect_cars_multi_scale
from ImageUtils import draw_boxes, normalize, gaussian_blur, bb_by_contours

N_FRAMES = 5


class CarDetector:
    def __init__(self,
                 clf,
                 delete_after=5,
                 cluster_thresh=128,
                 xy_window=(64, 64),
                 stride=(32, 32),
                 y_start_stops=None,
                 image_size_factors=[1],
                 n_jobs=1):
        self.n_jobs = n_jobs
        self.image_size_factors = image_size_factors
        self.y_start_stops = y_start_stops
        self.stride = stride
        self.xy_window = xy_window
        self.clf = clf
        self.detections = []
        self.cluster_thresh = cluster_thresh
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
        heat = normalize(heat, new_max=255, new_min=0, old_max=heat.max(), old_min=0, dtype=np.uint8)

        debug = np.copy(heat)
        out[360:, :640] = cv2.resize(np.stack((debug, debug, debug), axis=2), (640, 360))

        heat_thresh = np.zeros(heat.shape, dtype=np.uint8)
        heat_blur = gaussian_blur(heat, 21)
        heat_thresh[heat_blur > 50] = 255

        debug = np.copy(heat_thresh)
        out[360:, 640:] = cv2.resize(np.stack((debug, debug, debug), axis=2), (640, 360))

        boxes_contours = bb_by_contours(heat_thresh)
        img_contours = draw_boxes(img, boxes_contours, thick=3, color=(255, 0, 0))
        out[:360, 640:] = cv2.resize(img_contours, (640, 360))

        self.frame_cnt += 1
        return out
