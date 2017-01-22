import numpy as np
import scipy.cluster.hierarchy as hcluster

from VehicleTracking.Detection import Detection
from VehicleTracking.ImageUtils import draw_boxes, center_points, detect_cars, surrounding_box

N_FRAMES = 5


class CarDetector:
    def __init__(self, clf, delete_after=5, cluster_thresh=128):
        self.clf = clf
        self.detections = []
        self.cluster_thresh = cluster_thresh
        self.delete_after = delete_after
        self.frame_cnt = 0

    def process_frame(self, img):
        boxes, detection_confidence = detect_cars(img, self.clf)

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
