import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

from LaneDetection.CameraCalibration import get_camera_calibration, CameraCalibrator
from LaneDetection.LaneDetector import LaneDetector

FRAME_SHAPE = (1280, 720)
HIST_STEPS = 10
OFFSET = 250
FRAME_MEMORY = 7
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

VIDEOS = ["../videos/project_video.mp4", "../videos/challenge_video.mp4", "../videos/harder_challenge_video.mp4"]
SELECTED_VIDEO = 0

if __name__ == '__main__':
    cam_calibration = get_camera_calibration()
    cam_calibrator = CameraCalibrator(FRAME_SHAPE, cam_calibration)

    ld = LaneDetector(SRC, DST, n_frames=FRAME_MEMORY, cam_calibration=cam_calibrator, transform_offset=OFFSET)

    clip1 = VideoFileClip(VIDEOS[SELECTED_VIDEO])
    project_clip = clip1.fl_image(ld.process_frame)

    project_output = VIDEOS[SELECTED_VIDEO][:-4] + '_ann.mp4'
    project_clip.write_videofile(project_output, audio=False)
