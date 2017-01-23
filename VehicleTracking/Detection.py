import cv2
import numpy as np

from ImageUtils import center_points


class Detection:
    def __init__(self, box, n_frames=1):
        self.last_n_boxes = np.zeros((n_frames, 4))
        self.last_n_boxes[:] = box
        self.current_row = 0
        self.previous_row = 0
        self.best_box = box
        self.n_frames = n_frames
        self.frames_undetected = 0
        self.age = 0

        self.update(box)

    def update_current_row(self):
        self.previous_row = self.current_row
        self.current_row += 1

        if self.current_row >= self.n_frames:
            self.current_row = 0

    def get_center(self):
        return center_points(np.expand_dims(self.best_box, axis=0))

    def update(self, box):
        if box is not None:
            self.last_n_boxes[self.current_row] = box
            self.update_current_row()
        else:
            self.last_n_boxes[self.current_row] = self.last_n_boxes[self.previous_row]
            self.update_current_row()
            self.frames_undetected += 1

        if self.best_box is None:
            self.best_box = box
        else:
            self.best_box = np.mean(self.last_n_boxes, axis=0, keepdims=False).astype(np.uint32)

        self.age += 1

    def draw(self, img, color=(0, 0, 255), thick=6):
        if self.best_box is not None:
            cv2.rectangle(img, (self.best_box[0], self.best_box[1]), (self.best_box[2], self.best_box[3]), color, thick)
        return img
