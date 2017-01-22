import numpy as np
import scipy.cluster.hierarchy as hcluster
from moviepy.video.io.VideoFileClip import VideoFileClip
import pickle

from VehicleTracking.CarDetector import CarDetector
from VehicleTracking.ImageUtils import detect_cars, draw_boxes, center_points, surrounding_box

FRAME_SHAPE = (1280, 720)

VIDEOS = ["../videos/project_video.mp4",
          "../videos/project_video_short.mp4",
          "../videos/challenge_video.mp4",
          "../videos/harder_challenge_video.mp4"]
SELECTED_VIDEO = 1


if __name__ == '__main__':
    # with open('../models/svm_hnm.p', 'rb') as f:
    #     clf = pickle.load(f)

    detector = CarDetector(None)

    clip1 = VideoFileClip(VIDEOS[SELECTED_VIDEO])
    project_clip = clip1.fl_image(detector.process_frame)

    project_output = VIDEOS[SELECTED_VIDEO][:-4] + '_ann.mp4'
    project_clip.write_videofile(project_output, audio=False)
