import numpy as np
import scipy.cluster.hierarchy as hcluster
from Detection import Detection
from ImageUtils import center_points, surrounding_box

from ImageUtils import detect_cars_multi_scale

N_FRAMES = 3


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

    def process_frame(self, img):
        boxes = detect_cars_multi_scale(img, self.clf, self.xy_window, self.stride, self.y_start_stops,
                                        self.image_size_factors, self.n_jobs)

        if len(boxes) > 0:
            centers = center_points(boxes)
            centers_with_last_detections = np.zeros(((len(centers) + len(self.detections)), 2))
            centers_with_last_detections[:len(centers)] = centers
            for i, detection in enumerate(self.detections):
                centers_with_last_detections[len(centers) + i] = detection.get_center()

            if len(boxes) > 1:
                clusters = hcluster.fclusterdata(centers_with_last_detections,
                                                 self.cluster_thresh,
                                                 criterion="distance")
                surrounding_boxes, agg_clusters = surrounding_box(boxes, clusters[:len(centers)])
                unused_sur_boxes = np.ones((len(agg_clusters), 1), dtype=np.uint8)
                for i, detection in enumerate(self.detections):
                    detection_cluster = clusters[len(centers) + i]
                    new_detection_i = np.where(agg_clusters == detection_cluster)
                    new_detection = surrounding_boxes[new_detection_i]
                    if len(new_detection) == 1:
                        detection.update(new_detection[0])
                        unused_sur_boxes[new_detection_i] = 1
                    elif len(new_detection) == 0:
                        detection.update(None)

                for i in unused_sur_boxes.nonzero()[0]:
                    self.detections.append(Detection(surrounding_boxes[i], n_frames=N_FRAMES))

                new_detection = []
                for detection in self.detections:
                    if detection.frames_undetected < self.delete_after:
                        new_detection.append(detection)
                self.detections = new_detection

        else:
            for detection in self.detections:
                detection.update(None)

        for detection in self.detections:
            detection.draw(img)

        self.frame_cnt += 1

        return img
