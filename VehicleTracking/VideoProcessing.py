import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

from CameraCalibration import get_camera_calibration, CameraCalibrator

FRAME_SHAPE = (1280, 720)


VIDEOS = ["../videos/project_video.mp4", "../videos/challenge_video.mp4", "../videos/harder_challenge_video.mp4"]
SELECTED_VIDEO = 1

if __name__ == '__main__':
    cam_calibration = get_camera_calibration()
    cam_calibrator = CameraCalibrator(FRAME_SHAPE, cam_calibration)


    # clip1 = VideoFileClip(VIDEOS[SELECTED_VIDEO])
    # project_clip = clip1.fl_image(ld.process_frame)
    #
    # project_output = VIDEOS[SELECTED_VIDEO][:-4] + '_ann.mp4'
    # project_clip.write_videofile(project_output, audio=False)
