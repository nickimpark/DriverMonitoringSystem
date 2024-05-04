import os
import cv2
import dlib

from .eye import Eye
from .mouth import Mouth


class FaceFeaturesDetector(object):
    """
    This class extracts facial features like eyes (left and right)
    and mouth from an image (frame). Wherein Face Landmark Detector
    from DLib is used.
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.mouth = None

        # Face detector (DLib)
        self._face_detector = dlib.get_frontal_face_detector()

        # Face landmark detector (DLib)
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0)
            self.eye_right = Eye(frame, landmarks, 1)
            self.mouth = Mouth(frame, landmarks)

        except IndexError:
            self.eye_left = None
            self.eye_right = None
            self.mouth = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def annotated_frame(self):
        """Returns the main frame with eyes and mouth highlighted"""
        frame = self.frame.copy()

        color = (0, 255, 0)
        left_eye_hull = cv2.convexHull(self.eye_left.landmark_points)
        cv2.drawContours(frame, [left_eye_hull], -1, color)
        right_eye_hull = cv2.convexHull(self.eye_right.landmark_points)
        cv2.drawContours(frame, [right_eye_hull], -1, color)
        mouth_hull = cv2.convexHull(self.mouth.landmark_points)
        cv2.drawContours(frame, [mouth_hull], -1, color)

        return frame
