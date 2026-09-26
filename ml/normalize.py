"""
Shared hand-landmark normalization function.

Used by BOTH collect_data.py (when saving training samples) and
translator.py (when making live predictions) later on, so both stages
always process landmarks in exactly the same way. If these two ever
get out of sync, the model will perform worse in ways that are hard
to debug -- so this lives in one shared file instead of being
copy-pasted.
"""


def normalize_landmarks(hand_landmarks):
    """
    Convert MediaPipe's 21 hand landmarks into a normalized list of
    63 numbers (x, y, z for each point):

    - Subtract the wrist position (landmark 0) so it doesn't matter
      WHERE your hand is in the frame.
    - Divide by the wrist-to-middle-finger-base distance so it
      doesn't matter HOW FAR your hand is from the camera.

    Returns a flat list of 63 floats: [x0, y0, z0, x1, y1, z1, ...]

    NOTE: hand_landmarks here is a plain list of 21 landmark objects
    (as returned per-hand from MediaPipe's Tasks API via
    result.hand_landmarks[i]) -- not a .landmark-wrapped object like
    the old mp.solutions.hands API used.
    """
    points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
    wrist_x, wrist_y, wrist_z = points[0]

    # Shift everything so the wrist becomes the origin (0, 0, 0)
    shifted = [(x - wrist_x, y - wrist_y, z - wrist_z) for (x, y, z) in points]

    # Landmark 9 = middle finger MCP joint (base of the middle finger)
    mid_x, mid_y, mid_z = shifted[9]
    scale = (mid_x ** 2 + mid_y ** 2 + mid_z ** 2) ** 0.5
    if scale == 0:
        scale = 1e-6  # avoid divide-by-zero on a bad frame

    normalized = []
    for (x, y, z) in shifted:
        normalized += [x / scale, y / scale, z / scale]

    return normalized