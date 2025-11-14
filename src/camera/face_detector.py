"""
Face Detector Module
MediaPipe Face Mesh based face detector + landmarks.
Returns bounding box and list of landmarks in pixel coordinates.

Created by Ali Hamza & Zarmeena Jawad
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional

mp_face_mesh = mp.solutions.face_mesh


class FaceDetector:
    """Wraps MediaPipe Face Mesh for detection and landmarks extraction.

    Methods
    -------
    detect(frame)
        Returns bbox (x,y,w,h) and landmarks list [(x,y), ...] in pixel coords.
    """

    def __init__(self, refine_landmarks: bool = True, max_faces: int = 1):
        """Initialize face detector.
        
        Args:
            refine_landmarks: Whether to use refined landmarks (more accurate)
            max_faces: Maximum number of faces to detect
        """
        self.max_faces = max_faces
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], List[Tuple[int, int]]]:
        """Detect faces and landmarks in BGR frame.

        Args:
            frame: BGR image frame from camera

        Returns:
            bbox: (x, y, w, h) or None if no face detected
            landmarks: list of (x, y) in pixel coords (empty if none)
        """
        if frame is None:
            return None, []

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        try:
            results = self.face_mesh.process(img_rgb)
        except Exception:
            return None, []

        if not results.multi_face_landmarks:
            return None, []

        # Use first face only
        face_lms = results.multi_face_landmarks[0]

        landmarks = []
        x_vals = []
        y_vals = []
        for lm in face_lms.landmark:
            px = int(lm.x * w)
            py = int(lm.y * h)
            landmarks.append((px, py))
            x_vals.append(px)
            y_vals.append(py)

        if not x_vals or not y_vals:
            return None, landmarks

        x_min = max(min(x_vals) - 10, 0)
        y_min = max(min(y_vals) - 10, 0)
        x_max = min(max(x_vals) + 10, w)
        y_max = min(max(y_vals) + 10, h)

        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        return bbox, landmarks

