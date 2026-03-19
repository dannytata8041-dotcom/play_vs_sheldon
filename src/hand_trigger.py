"""Hand-based OCR trigger using 21 hand keypoints + gesture control.

Uses YOLO11n-pose-hands model for hand detection and keypoint estimation.
Gesture commands:
  HOLD       → triggers OCR scan on the held object region
  THUMBS_UP  → confirm / acknowledge alert
  OPEN_PALM  → pause scanning
  FIST       → resume scanning
  POINT      → (reserved for future use)

Keypoint layout (21 keypoints, MediaPipe convention):
  0: wrist
  1-4: thumb (cmc, mcp, ip, tip)
  5-8: index finger (mcp, pip, dip, tip)
  9-12: middle finger (mcp, pip, dip, tip)
  13-16: ring finger (mcp, pip, dip, tip)
  17-20: pinky finger (mcp, pip, dip, tip)
"""

import logging
import time

import numpy as np

from gestures import recognize

logger = logging.getLogger(__name__)

# Keypoint indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# Stability settings
STABILITY_THRESHOLD_SEC = 0.4
DEDUP_WINDOW_SEC = 60
WRIST_MOVE_THRESHOLD = 30  # pixels — max wrist movement to count as "stable"
GESTURE_COOLDOWN_SEC = 2.0  # prevent rapid gesture re-fires


def get_held_object_region(keypoints, img_shape, conf_threshold=0.3):
    """Estimate the bounding box of what the hand is holding.

    The held object is roughly in the region enclosed by the fingers,
    extending from the fingertips away from the palm.

    Args:
        keypoints: array of shape (21, 3).
        img_shape: (height, width) of the image.
        conf_threshold: minimum keypoint confidence.

    Returns:
        [x1, y1, x2, y2] bounding box or None.
    """
    h, w = img_shape[:2]

    tip_indices = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    points = []
    for idx in tip_indices:
        if keypoints[idx][2] > conf_threshold:
            points.append(keypoints[idx][:2])

    if keypoints[WRIST][2] > conf_threshold:
        points.append(keypoints[WRIST][:2])

    if len(points) < 3:
        return None

    points = np.array(points)
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)

    # Expand the box by 40% to capture the held object
    pad_x = (x2 - x1) * 0.4
    pad_y = (y2 - y1) * 0.4
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))

    return [x1, y1, x2, y2]


class HandTrigger:
    """Triggers OCR when a hand holds an object steady, with gesture control."""

    def __init__(self):
        self._prev_wrist = None
        self._stable_since = None
        self._last_trigger_time = 0
        self._last_gesture_time = 0
        self._last_gesture = "NONE"
        self._paused = False

    @property
    def paused(self):
        return self._paused

    def update(self, keypoints, img_shape):
        """Process hand keypoints for one frame.

        Args:
            keypoints: array of shape (21, 3) — x, y, conf.
            img_shape: (height, width) of the frame.

        Returns:
            dict with keys:
                "gesture": str — the recognized gesture
                "region": [x1, y1, x2, y2] or None — crop box if OCR should trigger
                "command": str or None — "SCAN", "CONFIRM", "PAUSE", "RESUME", None
        """
        now = time.monotonic()
        kp = np.asarray(keypoints, dtype=np.float32)
        gesture = recognize(kp)

        result = {"gesture": gesture, "region": None, "command": None}

        # Process gesture commands (with cooldown)
        if gesture != "NONE" and gesture != self._last_gesture:
            if now - self._last_gesture_time >= GESTURE_COOLDOWN_SEC:
                self._last_gesture_time = now

                if gesture == "OPEN_PALM" and not self._paused:
                    self._paused = True
                    result["command"] = "PAUSE"
                    logger.info("Gesture: OPEN_PALM → scanning PAUSED")

                elif gesture == "FIST" and self._paused:
                    self._paused = False
                    result["command"] = "RESUME"
                    logger.info("Gesture: FIST → scanning RESUMED")

                elif gesture == "THUMBS_UP":
                    result["command"] = "CONFIRM"
                    logger.info("Gesture: THUMBS_UP → confirmed")

        self._last_gesture = gesture

        # Don't trigger OCR when paused
        if self._paused:
            return result

        # Only trigger scan on HOLD gesture
        if gesture != "HOLD":
            self._prev_wrist = None
            self._stable_since = None
            return result

        # Dedup: don't trigger again within window
        if now - self._last_trigger_time < DEDUP_WINDOW_SEC:
            return result

        wrist = kp[WRIST][:2].copy()

        # Check stability via wrist movement
        if self._prev_wrist is not None:
            movement = np.linalg.norm(wrist - self._prev_wrist)
            if movement < WRIST_MOVE_THRESHOLD:
                if self._stable_since is None:
                    self._stable_since = now
                elif now - self._stable_since >= STABILITY_THRESHOLD_SEC:
                    region = get_held_object_region(kp, img_shape)
                    if region is not None:
                        self._last_trigger_time = now
                        self._prev_wrist = None
                        self._stable_since = None
                        result["region"] = region
                        result["command"] = "SCAN"
                        logger.info("Gesture: HOLD (stable) → OCR triggered")
                        return result
            else:
                self._stable_since = None

        self._prev_wrist = wrist
        return result
