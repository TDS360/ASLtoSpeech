"""
Shared MediaPipe Hand Landmarker setup and drawing helper.

MediaPipe replaced the old `mp.solutions.hands` interface with a
newer "Tasks" API (`mp.tasks.vision.HandLandmarker`). This file sets
that up once so hand_detection.py, collect_data.py, and
translator.py can all share the same detector and drawing code
instead of repeating it three times.
"""

import os
import time
import cv2
import mediapipe as mp

from paths import hand_landmarker_path
MODEL_PATH = hand_landmarker_path()

# The 21 hand landmarks and how they connect to form a skeleton.
# This is MediaPipe's standard hand topology (wrist = point 0, each
# finger has 4 points running from its base to its tip).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky finger
    (0, 17),                                 # palm base
]


def create_landmarker(num_hands=1):
    """
    Creates a HandLandmarker set up for real-time video, using the
    model file at MODEL_PATH. Raises a clear error with download
    instructions if that file hasn't been downloaded yet.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}\n"
            "Download it by running this from PowerShell, inside your "
            "src folder:\n\n"
            '  Invoke-WebRequest -Uri '
            '"https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
            'hand_landmarker/float16/1/hand_landmarker.task" '
            '-OutFile "..\\models\\hand_landmarker.task"\n'
        )

    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


_last_ts = 0


def detect(landmarker, rgb_frame):
    """
    Runs hand detection on one RGB frame. Returns the raw result --
    result.hand_landmarks is a list of hands, each a list of 21
    landmarks with .x, .y, .z (normalized 0.0-1.0).
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    # detect_for_video requires strictly increasing timestamps; time.time()
    # can jump backwards (NTP sync) or repeat, which crashes MediaPipe.
    global _last_ts
    timestamp_ms = max(int(time.monotonic() * 1000), _last_ts + 1)
    _last_ts = timestamp_ms
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def draw_landmarks(frame, hand_landmarks):
    """
    Draws the hand skeleton directly with OpenCV -- this avoids
    depending on mp.solutions.drawing_utils, which isn't reliably
    available across different MediaPipe versions.
    """
    height, width = frame.shape[:2]
    points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)
