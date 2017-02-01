import cv2
import numpy as np
from ImageUtils import center_points

from ImageUtils import multi_bb_intersection_over_union

from VehicleTracking.ImageUtils import dist_metric

CNT = 0


class Detection:
    def __init__(self, box):
        global CNT
        self.is_hidden = False
        self.last_boxes = []
        self.best_box = None
        self.frames_undetected = 0
        self.age = 0
        self.n_frames = 7

        self.update(box)

    def get_center(self):
        return center_points(np.expand_dims(self.best_box, axis=0))

    def update(self, box):
        if box is not None:
            self.last_boxes.append(box)
            bound = min(len(self.last_boxes), self.n_frames)
            self.best_box = np.mean(self.last_boxes[-bound:], axis=0).astype(np.uint32)

            self.frames_undetected = 0
        else:
            self.frames_undetected += 1

        self.age += 1

    def unhide(self, box):
        self.last_boxes.extend([box] * self.n_frames)
        self.is_hidden = False

    def draw(self, img, color=(0, 0, 255), thick=6):
        if self.best_box is not None:
            cv2.rectangle(img, (self.best_box[0], self.best_box[1]), (self.best_box[2], self.best_box[3]), color, thick)
        return img

    def iou_with(self, boxes):
        return multi_bb_intersection_over_union(self.best_box, boxes)

    def dist_metric_with(self, boxes):
        return dist_metric(self.best_box, boxes)
