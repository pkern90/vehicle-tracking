import time

import cv2
import numpy as np
from ImageUtils import center_points, multi_bb_intersection_over_union

from VehicleTracking.ImageUtils import relative_distance


class Detection:
    def __init__(self, box):
        """
        Describes a detection on a image.
        :param box: initial bounding box
        """
        self.is_hidden = False
        self.last_boxes = []
        self.best_box = None
        self.frames_undetected = 0
        self.age = 0
        self.n_frames = 10
        self.detection_time = time.time()

        self.update(box)

    def update(self, box):
        """
        Updates the detection object with a new bounding box.
        :param box:
        :return:
        """
        if box is not None:
            self.last_boxes.append(box)
            bound = min(len(self.last_boxes), self.n_frames)
            self.best_box = np.mean(self.last_boxes[-bound:], axis=0).astype(np.uint32)

            self.frames_undetected = 0
        else:
            self.frames_undetected += 1

        self.age += 1

    def unhide(self, box):
        """
        Unhides the detection and sets the bounding box to the no value
        :param box:
        """
        if self.is_hidden:
            self.last_boxes.extend([box] * self.n_frames)
            self.is_hidden = False

    def draw(self, img, color=(0, 0, 255), thick=6):
        """
        Draws the bounding box of the detection on a given image.
        It also adds information on how long the detection has been active.
        :param img: image to draw on
        :param color:
        :param thick:
        :return:
        """
        if self.best_box is None:
            return img

        box_to_draw = np.zeros(4, dtype=np.uint32)
        if self.is_hidden:
            w = self.best_box[2] - self.best_box[0]
            h = self.best_box[3] - self.best_box[1]

            box_to_draw[:2] = self.best_box[:2] + min(25, w // 2)
            box_to_draw[2:] = self.best_box[2:] - min(25, h // 2)
        else:
            box_to_draw = self.best_box

        cv2.rectangle(img, (box_to_draw[0], box_to_draw[1]), (box_to_draw[2], box_to_draw[3]), color, thick)
        cv2.putText(img, '%.1fs' % (time.time() - self.detection_time),
                    (box_to_draw[0] + 10, box_to_draw[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 2)
        return img

    def iou_with(self, boxes):
        """
        Calculates the intersect over union with all the given bounding boxes.
        :param boxes:
        :return:
        """
        return multi_bb_intersection_over_union(self.best_box, boxes)

    def relative_distance_with(self, boxes):
        """
        Calculates the relative distance with all the given bounding boxes by
        calculating the mean diagonal and dividing it by the distance.
        :param boxes:
        :return:
        """
        return relative_distance(self.best_box, boxes)

    def get_center(self):
        """
        Returns the center coordinates of the detection.
        :return:
        """
        return center_points(np.expand_dims(self.best_box, axis=0))[0]