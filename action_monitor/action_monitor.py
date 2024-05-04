import math
import cv2

from .pupil import Pupil
from .calibration import Calibration


class ActionMonitor(object):
    """
    This class monitors driver's action / behavior features
    such as:
    - Sight direction
    - Yawn
    """
    def __init__(self, detector):
        self.detector = detector
        self.calibration = Calibration()
        self.frame = None
        self.left_pupil = None
        self.right_pupil = None
        self.fps = 10

        # Action / behavior flags
        self.isDistracted = 0
        self.Yawns = 0

        # Action / behavior flag counters
        self.isDistractedCounter = 0
        self.YawnsCounter = 0

    def update_fps(self, fps):
        """
        Updates FPS for calculating flags

        Arguments:
            fps: Frames per second (based on full frame processing time)
        """
        self.fps = fps

    def track_pupils(self, side):
        """
        Detect and track left or right pupil

        Arguments:
            side: Indicates whether it's the left eye (0) or the right eye (1)
        """
        if side == 0:
            if not self.calibration.is_complete():
                self.calibration.evaluate(self.detector.eye_left.frame, side)

            threshold = self.calibration.threshold(side)
            self.left_pupil = Pupil(self.detector.eye_left.frame, threshold)
        elif side == 1:
            if not self.calibration.is_complete():
                self.calibration.evaluate(self.detector.eye_right.frame, side)

            threshold = self.calibration.threshold(side)
            self.right_pupil = Pupil(self.detector.eye_right.frame, threshold)

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.left_pupil.x)
            int(self.left_pupil.y)
            int(self.right_pupil.x)
            int(self.right_pupil.y)
            return True
        except:
            return False

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.detector.eye_left.origin[0] + self.left_pupil.x
            y = self.detector.eye_left.origin[1] + self.left_pupil.y
            return x, y

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.detector.eye_right.origin[0] + self.right_pupil.x
            y = self.detector.eye_right.origin[1] + self.right_pupil.y
            return x, y

    def horizontal_ratio(self):
        """
        Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.left_pupil.x / (self.detector.eye_left.center[0] * 2 - 10)
            pupil_right = self.right_pupil.x / (self.detector.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.left_pupil.y / (self.detector.eye_left.center[1] * 2 - 10)
            pupil_right = self.right_pupil.y / (self.detector.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.4

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def mouth_aspect_ratio(self):
        """Returns aspect ratio of detected mouth"""
        left = self.detector.mouth.landmark_points[0]
        right = self.detector.mouth.landmark_points[6]
        top = self.detector.mouth.landmark_points[3]
        bottom = self.detector.mouth.landmark_points[9]
        mouth_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        mouth_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = mouth_width / mouth_height
        except ZeroDivisionError:
            ratio = 5.0

        return ratio

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted and text added"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 0, 255)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        if self.is_right():
            eye_text = "Взгляд вправо"
        elif self.is_left():
            eye_text = "Взгляд влево"
        elif self.is_center():
            eye_text = "Взгляд прямо"
        else:
            eye_text = "Зрачки не обнаружены"

        cv2.putText(frame, eye_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (150, 50, 25), 2)

        if self.mouth_aspect_ratio() < 2:
            mouth_text = "Рот открыт"
        else:
            mouth_text = "Рот закрыт"

        cv2.putText(frame, mouth_text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (150, 50, 25), 2)

        return frame

    def _analyze(self):
        """Tracks pupils and updates monitoring parameters"""
        isDistractedCounter = 0
        YawnsCounter = 0
        self.track_pupils(0)
        self.track_pupils(1)

        # IsDistracted flag
        if not self.is_center():
            self.isDistractedCounter += 1.0 / self.fps
            if self.isDistractedCounter > 2.0:
                self.isDistracted = 1
        else:
            isDistractedCounter = self.isDistractedCounter
            self.isDistractedCounter = 0
            self.isDistracted = 0

        # Yawns flag
        if self.mouth_aspect_ratio() < 2:
            self.YawnsCounter += 1.0 / self.fps
            if self.YawnsCounter > 2.0:
                self.Yawns = 1
        else:
            YawnsCounter = self.YawnsCounter
            self.YawnsCounter = 0
            self.Yawns = 0

        return isDistractedCounter, YawnsCounter

    def refresh(self, frame):
        """
        Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        flags = self._analyze()
        return flags
