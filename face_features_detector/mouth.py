import dlib
import numpy as np
import cv2


class Mouth(object):
    """
    This class creates a new frame to isolate the mouth
    """

    MOUTH_POINTS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

    def __init__(self, original_frame, landmarks):
        self.frame = None
        self.origin = None
        self.center = None
        self.landmark_points = None
        self._analyze(original_frame, landmarks)

    @staticmethod
    def _middle_point(p1: dlib.point, p2: dlib.point):
        """
        Returns the middle point (x,y) between two DLib points.

        Arguments:
            p1 (dlib.point): First point
            p2 (dlib.point): Second point
        """
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return x, y

    def _isolate(self, frame, landmarks, points):
        """Isolate a mouth, to have a frame with mouth only.

        Arguments:
            frame (numpy.ndarray): Frame containing the face
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
            points (list): Points of an eye (from the 68 Multi-PIE landmarks)
        """
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)
        self.landmark_points = region

        # Applying a mask to get only the mouth
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        mouth = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # Cropping on the mouth
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = mouth[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _analyze(self, original_frame, landmarks):
        """Detects and isolates the mouth in a frame.

        Arguments:
            original_frame (numpy.ndarray): Frame from camera / video
            landmarks (dlib.full_object_detection): Facial landmarks for the face region
        """
        points = self.MOUTH_POINTS
        self._isolate(original_frame, landmarks, points)
