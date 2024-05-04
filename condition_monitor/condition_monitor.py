import cv2
import math


class ConditionMonitor(object):
    """
        This class monitors driver's condition features
        such as:
        - Eye closure duration
        - No Blinking Detection (Unconsciousness)
        """
    def __init__(self, detector):
        self.detector = detector
        self.frame = None
        self.fps = 10

        # Condition flags
        self.EyesClosed = 0
        self.NoBlinking = 0

        # Condition flag counters
        self.EyesClosedCounter = 0
        self.NoBlinkingCounter = 0

    def update_fps(self, fps):
        """
        Updates FPS for calculating flags

        Arguments:
            fps: Frames per second (based on full frame processing time)
        """
        self.fps = fps

    def eye_aspect_ratio(self, left, right, top, bottom):
        """Returns aspect ratio of detected eye (additional)

        Arguments:
            left: coordinates of left eye point
            right: coordinates of right eye point
            top: coordinates of top eye point
            bottom: coordinates of bottom eye point
        """
        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = 10.0

        return ratio

    def mean_eye_aspect_ratio(self):
        """Returns mean aspect ratio of detected eyes"""
        left = self.detector.eye_left.landmark_points[0]
        right = self.detector.eye_left.landmark_points[3]
        top = self.detector.eye_left.landmark_points[1]
        bottom = self.detector.eye_left.landmark_points[5]
        left_eye_ar = self.eye_aspect_ratio(left, right, top, bottom)

        left = self.detector.eye_right.landmark_points[0]
        right = self.detector.eye_right.landmark_points[3]
        top = self.detector.eye_right.landmark_points[1]
        bottom = self.detector.eye_right.landmark_points[5]
        right_eye_ar = self.eye_aspect_ratio(left, right, top, bottom)

        mean_ratio = (left_eye_ar + right_eye_ar) / 2
        return mean_ratio

    def annotated_frame(self):
        """Returns the main frame with text added"""
        frame = self.frame.copy()

        if self.mean_eye_aspect_ratio() > 5:
            text = "Глаза закрыты"
        else:
            text = "Глаза открыты"

        cv2.putText(frame, text, (50, 150), cv2.FONT_HERSHEY_COMPLEX, 1.0, (150, 50, 25), 2)

        return frame

    def _analyze(self):
        """Updates monitoring parameters"""
        EyesClosedCounter = 0
        NoBlinkingCounter = 0

        # EyesClosed & NoBlinking flag
        if self.mean_eye_aspect_ratio() > 5:
            self.EyesClosedCounter += 1.0 / self.fps
            if self.EyesClosedCounter > 2.0:
                self.EyesClosed = 1

            NoBlinkingCounter = self.NoBlinkingCounter
            self.NoBlinkingCounter = 0
            self.NoBlinking = 0
        else:
            EyesClosedCounter = self.EyesClosedCounter
            self.EyesClosedCounter = 0
            self.EyesClosed = 0

            self.NoBlinkingCounter += 1.0 / self.fps
            if self.NoBlinkingCounter > 20.0:
                self.NoBlinking = 1

        return EyesClosedCounter, NoBlinkingCounter

    def refresh(self, frame):
        """
        Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        flags = self._analyze()
        return flags
