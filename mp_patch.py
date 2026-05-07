import sys
import mediapipe as mp_real
import cv2
import urllib.request
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Download model if not exists — resolve relative to this file
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_FILE = os.path.join(_SCRIPT_DIR, 'hand_landmarker.task')
if not os.path.exists(TASK_FILE):
    print("[mp_patch] Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        TASK_FILE
    )
    print("[mp_patch] Download complete.")


class LandmarkList:
    def __init__(self, landmarks):
        self.landmark = landmarks


class ProcessResult:
    def __init__(self, multi_hand_landmarks, multi_handedness=None):
        self.multi_hand_landmarks = multi_hand_landmarks
        self.multi_handedness = multi_handedness


class HandednessInfo:
    """Mimics the handedness classification from legacy mediapipe."""
    def __init__(self, label, score):
        self.classification = [type('obj', (object,), {'label': label, 'score': score})()]


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


class Hands:
    def __init__(self, static_image_mode=False, max_num_hands=2,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        base_options = python.BaseOptions(model_asset_path=TASK_FILE)
        running_mode = vision.RunningMode.IMAGE
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            running_mode=running_mode
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, rgb_frame):
        import numpy as np
        rgb_frame = np.ascontiguousarray(rgb_frame)
        mp_image = mp_real.Image(image_format=mp_real.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.hand_landmarks:
            return ProcessResult(None, None)

        multi_hand_landmarks = []
        multi_handedness = []
        for i, hand in enumerate(detection_result.hand_landmarks):
            multi_hand_landmarks.append(LandmarkList(hand))
            # Get handedness info if available
            if detection_result.handedness and i < len(detection_result.handedness):
                h = detection_result.handedness[i]
                if h:
                    label = h[0].category_name if hasattr(h[0], 'category_name') else 'Unknown'
                    score = h[0].score if hasattr(h[0], 'score') else 0.0
                    multi_handedness.append(HandednessInfo(label, score))
                else:
                    multi_handedness.append(HandednessInfo('Unknown', 0.0))
            else:
                multi_handedness.append(HandednessInfo('Unknown', 0.0))

        return ProcessResult(multi_hand_landmarks, multi_handedness)

    def close(self):
        self.detector.close()


class _DrawingUtils:
    # Color palette for different hands
    HAND_COLORS = [
        ((0, 255, 100), (0, 200, 255)),   # Hand 1: green landmarks, cyan connections
        ((255, 100, 0), (255, 200, 0)),    # Hand 2: orange landmarks, yellow connections
    ]

    def draw_landmarks(self, image, landmark_list, connections=None,
                       landmark_drawing_spec=None, connection_drawing_spec=None,
                       hand_index=0):
        if not landmark_list:
            return
        image_rows, image_cols, _ = image.shape
        colors = self.HAND_COLORS[hand_index % len(self.HAND_COLORS)]
        lm_color, conn_color = colors

        # Draw connections
        if connections:
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                start = landmark_list.landmark[start_idx]
                end = landmark_list.landmark[end_idx]
                start_px = (int(start.x * image_cols), int(start.y * image_rows))
                end_px = (int(end.x * image_cols), int(end.y * image_rows))
                cv2.line(image, start_px, end_px, conn_color, 2, cv2.LINE_AA)

        # Draw landmarks
        for lm in landmark_list.landmark:
            px = (int(lm.x * image_cols), int(lm.y * image_rows))
            cv2.circle(image, px, 5, lm_color, -1, cv2.LINE_AA)
            cv2.circle(image, px, 5, (255, 255, 255), 1, cv2.LINE_AA)


class _Solutions:
    class _Hands:
        Hands = Hands
        HAND_CONNECTIONS = HAND_CONNECTIONS

    hands = _Hands()
    drawing_utils = _DrawingUtils()

    class _DrawingStyles:
        def get_default_hand_landmarks_style(self):
            return None
        def get_default_hand_connections_style(self):
            return None

    drawing_styles = _DrawingStyles()


class _MPWrapper:
    solutions = _Solutions()
    Image = mp_real.Image
    ImageFormat = mp_real.ImageFormat


# Overwrite mediapipe module with our wrapper
sys.modules['mediapipe'] = _MPWrapper()
