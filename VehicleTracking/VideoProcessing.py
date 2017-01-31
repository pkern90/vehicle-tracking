import pickle

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

from VehicleTracking.CarDetector import CarDetector

DELETE_AFTER = 24

Y_START_STOPS = np.array([
    [400, 496],
    [400, 496],
    [400, 592],
    [366, 690]
])
STRIDE = np.array([
    [24, 24],
    [16, 16],
    [16, 16],
    [16, 16]
])
IMAGE_SIZE_FACTORS = [
    1.5,
    1,
    1 / 1.5,
    1 / 3
]
XY_WINDOW = (64, 64)

N_JOBS = 4

FRAME_SHAPE = (1280, 720)

VIDEOS = ["../videos/project_video.mp4",
          "../videos/project_video_short.mp4",
          "../videos/project_video_very_short.mp4",
          "../videos/challenge_video.mp4",
          "../videos/harder_challenge_video.mp4"]
SELECTED_VIDEO = 0

if __name__ == '__main__':
    with open('../models/svm_adj.p', 'rb') as f:
        clf = pickle.load(f)

    detector = CarDetector(clf,
                           delete_after=DELETE_AFTER,
                           xy_window=XY_WINDOW,
                           stride=STRIDE,
                           y_start_stops=Y_START_STOPS,
                           image_size_factors=IMAGE_SIZE_FACTORS,
                           n_jobs=N_JOBS)

    clip1 = VideoFileClip(VIDEOS[SELECTED_VIDEO])
    project_clip = clip1.fl_image(detector.process_frame)

    project_output = VIDEOS[SELECTED_VIDEO][:-4] + '_ann.mp4'
    project_clip.write_videofile(project_output, audio=False)
