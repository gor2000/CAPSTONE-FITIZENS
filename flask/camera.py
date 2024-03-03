from ultralytics.utils.plotting import Annotator

import utils
import cv2

class VideoCamera(object):
    def __init__(self, model):
        self.model = model
        self.video = None
        self.running = False
        self.initialize_camera()
        self.gym_object = utils.ExcerciseCounter()
        self.gym_object.set_args(
            line_thickness=3,
            view_img=True,
            pose_type='pushup',
            video_source_is_video=False,
            show_skeleton=True
        )
        self.frame = None
        self.frame_count = 0

    def __del__(self):
        if self.video is not None:
            self.video.release()

    def initialize_camera(self):
        if not self.running:
            self.video = cv2.VideoCapture(0)
            self.running = True


    def release_camera(self):
        if self.video is not None:
            self.video.release()
            self.video = None
        self.running = False

    def stop(self):
        self.release_camera()

    def start(self):
        self.initialize_camera()



    def get_frame(self):
        if not self.running or self.video is None:
            return None

        success, self.frame = self.video.read()
        self.frame.flags.writeable = False
        if not success:
            return None
        results = self.model.predict(self.frame, verbose=False)
        self.frame.flags.writeable = True
        self.frame_count += 1
        self.frame = self.gym_object.start_counting(self.frame, results, self.frame_count)
        self.frame = cv2.resize(self.frame, (640, 640), interpolation=cv2.INTER_AREA)
        _, jpeg = cv2.imencode('.jpg', self.frame)
        return jpeg.tobytes()

