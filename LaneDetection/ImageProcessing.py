import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread

from LaneDetection.CameraCalibration import get_camera_calibration, CameraCalibrator
from LaneDetection.LaneDetector import LaneDetector

HIST_STEPS = 10
OFFSET = 250
FRAME_MEMORY = 5
SRC = np.float32([
    (132, 703),
    (540, 466),
    (740, 466),
    (1147, 703)])

DST = np.float32([
    (SRC[0][0] + OFFSET, 720),
    (SRC[0][0] + OFFSET, 0),
    (SRC[-1][0] - OFFSET, 0),
    (SRC[-1][0] - OFFSET, 720)])

if __name__ == '__main__':
    cam_calibration = get_camera_calibration()
    img = imread('../test_images/test1.jpg')
    cam_calibrator = CameraCalibrator(img[:, :, 0].shape[::-1], cam_calibration)
    ld = LaneDetector(SRC, DST, n_frames=FRAME_MEMORY, cam_calibration=cam_calibrator, transform_offset=OFFSET)

    images = []
    for i in range(1, 12):
        images.append(imread('../test_images/test%s.jpg' % i))

    rows = len(images)
    cols = 2
    fig, axis = plt.subplots(rows, cols)
    for row in range(rows):
        img = images[row]

        axis[row, 0].imshow(img, cmap='gray')
        axis[row, 0].axis('off')
        ld = LaneDetector(SRC, DST, n_frames=FRAME_MEMORY, cam_calibration=cam_calibrator, transform_offset=OFFSET)
        img = ld.process_frame(img)

        axis[row, 1].imshow(img, cmap='gray')
        axis[row, 1].axis('off')

    plt.show()
