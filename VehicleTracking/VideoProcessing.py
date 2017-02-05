import pickle

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

from VehicleTracking.CarDetector import CarDetector

# Numbers of frames to wait before deleting a
# detection object without new detections on
# the heatmap
DELETE_AFTER = 24

# Number of frame for averaging the detections
N_FRAMES = 8

# Defines different search windows.
# They are only limited by the y coordinate
# The coordinates are based on the original window
# size and will be adjusted when resizing
Y_START_STOPS = np.array([
    [400, 400 + 96],
    [400, 400 + 128 + 16],
    [528 - 32, 528 + 128],
    [400, 400 + 256],
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
# wont't yield any implements
N_JOBS = 4


VIDEOS = ["../videos/project_video.mp4",
          "../videos/project_video_short.mp4",
          "../videos/project_video_very_short.mp4",
          "../videos/challenge_video.mp4",
          "../videos/harder_challenge_video.mp4"]
SELECTED_VIDEO = 0

if __name__ == '__main__':
    with open('../models/svm_final.p', 'rb') as f:
        clf = pickle.load(f)

    detector = CarDetector(clf,
                           delete_after=DELETE_AFTER,
                           xy_window=XY_WINDOW,
                           stride=STRIDE,
                           y_start_stops=Y_START_STOPS,
                           image_size_factors=IMAGE_SIZE_FACTORS,
                           x_padding=X_PADDING,
                           dist_thresh=0.1,
                           n_frames=N_FRAMES,
                           n_jobs=N_JOBS)

    clip1 = VideoFileClip(VIDEOS[SELECTED_VIDEO])
    project_clip = clip1.fl_image(detector.process_frame)

    project_output = VIDEOS[SELECTED_VIDEO][:-4] + '_ann.mp4'
    project_clip.write_videofile(project_output, audio=False)
