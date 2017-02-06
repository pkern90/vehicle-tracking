import pickle
import sys
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

from LaneDetection.CameraCalibration import get_camera_calibration, CameraCalibrator
from LaneDetection.LaneDetector import LaneDetector
from VehicleTracking.CarDetector import CarDetector

# ******* Vehicle Tracking Const ***********

# Numbers of frames to wait before deleting a
# detection object without new detections on
# the heatmap
DELETE_AFTER = 24

# Number of frame for averaging the detections
N_FRAMES = 10

# Threshold for relative distance to join detections
DIST_THRESH = 0.1

# Defines different search windows.
# They are only limited by the y coordinate
# The coordinates are based on the original window
# size and will be adjusted when resizing
Y_START_STOPS = np.array([
    [400, 496],
    [400, 536],
    [488, 624],
    [400, 656],
])

# Stride to use for each search area
STRIDE = np.array([
    [24, 24],
    [16, 16],
    [16, 16],
    [16, 16],
])

# Resize factor for each search area.
IMAGE_SIZE_FACTORS = [
    1.5,
    1,
    1,
    1 / 1.5,
]

# Number of pixels (zeros) to add on each side of the x axis
X_PADDING = [
    24,
    32,
    32,
    32,
]

# The window size has to be the same for all
# search areas since the classifier expects
# a fixed feature size.
XY_WINDOW = (64, 64)

# The algorithm tries to run each search area on
# a separate cpu core. Therefore jobs > number of search areas
# wont't yield any improvements
N_JOBS = 4

# ******* Lane Detection Const ***********

FRAME_SHAPE = (1280, 720)

HIST_STEPS = 10

OFFSET = 250

FRAME_MEMORY = 7

# Source coordinates for perspective transformation
SRC = np.float32([
    (132, 703),
    (540, 466),
    (740, 466),
    (1147, 703)])

# Destination coordinates for perspective transformation
DST = np.float32([
    (SRC[0][0] + OFFSET, 720),
    (SRC[0][0] + OFFSET, 0),
    (SRC[-1][0] - OFFSET, 0),
    (SRC[-1][0] - OFFSET, 720)])

VIDEOS = ["videos/project_video.mp4",
          "videos/project_video_short.mp4",
          "videos/project_video_very_short.mp4",
          "videos/challenge_video.mp4"]
SELECTED_VIDEO = 2


def process_frame(frame, lane_detector, vehicle_detector):
    frame_lane = np.copy(frame)
    frame_lane = lane_detector.process_frame(frame_lane)
    _ = vehicle_detector.process_frame(frame)
    vehicle_detector.draw_info(frame_lane)

    return frame_lane

if __name__ == '__main__':
    sys.path.append("VehicleTracking/")

    with open('models/svm_final.p', 'rb') as f:
        clf = pickle.load(f)

    vehicle_detector = CarDetector(clf,
                                   delete_after=DELETE_AFTER,
                                   xy_window=XY_WINDOW,
                                   stride=STRIDE,
                                   y_start_stops=Y_START_STOPS,
                                   image_size_factors=IMAGE_SIZE_FACTORS,
                                   x_padding=X_PADDING,
                                   dist_thresh=DIST_THRESH,
                                   n_frames=N_FRAMES,
                                   n_jobs=N_JOBS,
                                   auto_draw=False)

    cam_calibration = get_camera_calibration()
    cam_calibrator = CameraCalibrator(FRAME_SHAPE, cam_calibration)
    lane_detector = LaneDetector(SRC, DST, n_frames=FRAME_MEMORY, cam_calibration=cam_calibrator,
                                 transform_offset=OFFSET)

    clip1 = VideoFileClip(VIDEOS[SELECTED_VIDEO])
    project_clip = clip1.fl_image(lambda frame: process_frame(frame, lane_detector, vehicle_detector))

    project_output = VIDEOS[SELECTED_VIDEO][:-4] + '_ann.mp4'
    project_clip.write_videofile(project_output, audio=False)
