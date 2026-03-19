"""Gesture recognition from 21 hand keypoints.

Recognizes gestures for controlling the label tracker:
  - HOLD:      Hand gripping an object (triggers OCR scan)
  - THUMBS_UP: Confirm / acknowledge alert
  - OPEN_PALM: Pause scanning
  - FIST:      Resume scanning
  - SWIPE:     Dismiss current alert (tracked across frames)

Keypoint layout (MediaPipe / YOLO11n-pose-hands):
  0: wrist
  1-4: thumb (cmc, mcp, ip, tip)
  5-8: index (mcp, pip, dip, tip)
  9-12: middle (mcp, pip, dip, tip)
  13-16: ring (mcp, pip, dip, tip)
  17-20: pinky (mcp, pip, dip, tip)
"""

import numpy as np

# Keypoint indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
FINGER_PIPS = [None, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]


def _visible(kp, idx, threshold=0.3):
    """Check if a keypoint is visible (confidence above threshold)."""
    return kp[idx][2] > threshold


def _dist(kp, a, b):
    """Euclidean distance between two keypoints."""
    return np.linalg.norm(kp[a][:2] - kp[b][:2])


def _finger_extended(kp, tip, pip, mcp, threshold=0.3):
    """Check if a finger is extended (tip farther from wrist than pip)."""
    if not (_visible(kp, tip, threshold) and _visible(kp, mcp, threshold)):
        return False
    tip_dist = _dist(kp, tip, WRIST)
    mcp_dist = _dist(kp, mcp, WRIST)
    return tip_dist > mcp_dist * 1.1


def _finger_curled(kp, tip, pip, mcp, threshold=0.3):
    """Check if a finger is curled (tip closer to wrist than mcp)."""
    if not (_visible(kp, tip, threshold) and _visible(kp, mcp, threshold)):
        return False
    tip_dist = _dist(kp, tip, WRIST)
    mcp_dist = _dist(kp, mcp, WRIST)
    return tip_dist < mcp_dist * 1.1


def _thumb_extended(kp, threshold=0.3):
    """Check if thumb is extended (tip far from index mcp)."""
    if not (_visible(kp, THUMB_TIP, threshold) and _visible(kp, INDEX_MCP, threshold)):
        return False
    thumb_dist = _dist(kp, THUMB_TIP, INDEX_MCP)
    hand_size = _dist(kp, WRIST, MIDDLE_MCP) if _visible(kp, MIDDLE_MCP, threshold) else 100
    return thumb_dist > hand_size * 0.6


def _thumb_up(kp, threshold=0.3):
    """Check if thumb is pointing upward relative to wrist."""
    if not (_visible(kp, THUMB_TIP, threshold) and _visible(kp, WRIST, threshold)):
        return False
    # Thumb tip should be significantly above wrist (lower y = higher in image)
    return kp[THUMB_TIP][1] < kp[WRIST][1] - 30


def recognize(keypoints, conf_threshold=0.3):
    """Classify the hand gesture from 21 keypoints.

    Args:
        keypoints: np.array of shape (21, 3) — x, y, confidence.
        conf_threshold: minimum keypoint confidence.

    Returns:
        One of: "HOLD", "THUMBS_UP", "OPEN_PALM", "FIST", "POINT", "NONE"
    """
    kp = np.asarray(keypoints, dtype=np.float32)

    if not _visible(kp, WRIST, conf_threshold):
        return "NONE"

    # Count extended and curled fingers (excluding thumb)
    fingers = [
        (INDEX_TIP, INDEX_PIP, INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
        (RING_TIP, RING_PIP, RING_MCP),
        (PINKY_TIP, PINKY_PIP, PINKY_MCP),
    ]

    extended = sum(1 for tip, pip, mcp in fingers
                   if _finger_extended(kp, tip, pip, mcp, conf_threshold))
    curled = sum(1 for tip, pip, mcp in fingers
                 if _finger_curled(kp, tip, pip, mcp, conf_threshold))

    thumb_ext = _thumb_extended(kp, conf_threshold)
    thumb_is_up = _thumb_up(kp, conf_threshold)

    # THUMBS_UP: thumb pointing up, all other fingers curled
    if thumb_is_up and thumb_ext and curled >= 3:
        return "THUMBS_UP"

    # OPEN_PALM: all fingers extended + thumb extended
    if extended >= 4 and thumb_ext:
        return "OPEN_PALM"

    # FIST: all fingers curled, thumb not extended
    if curled >= 4 and not thumb_ext:
        return "FIST"

    # POINT: only index extended
    if (extended == 1
            and _finger_extended(kp, INDEX_TIP, INDEX_PIP, INDEX_MCP, conf_threshold)
            and curled >= 2):
        return "POINT"

    # HOLD: some fingers curled (gripping), not a full fist
    if curled >= 2 and extended <= 1:
        return "HOLD"

    # Also detect pinch grip as HOLD
    if (_visible(kp, THUMB_TIP, conf_threshold)
            and _visible(kp, INDEX_TIP, conf_threshold)):
        pinch = _dist(kp, THUMB_TIP, INDEX_TIP)
        hand_size = _dist(kp, WRIST, MIDDLE_MCP) if _visible(kp, MIDDLE_MCP, conf_threshold) else 100
        if hand_size > 0 and pinch < hand_size * 0.4:
            return "HOLD"

    return "NONE"
