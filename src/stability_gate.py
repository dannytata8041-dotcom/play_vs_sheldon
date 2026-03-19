"""Stability gate — sits between raw frame stream and OCR trigger.

Prevents redundant inference on motion frames by requiring a label
bounding box to remain stable for a minimum duration before firing OCR.
"""

import time

STABILITY_THRESHOLD_SEC = 0.4
DEDUP_WINDOW_SEC = 60
IOU_STABILITY_MIN = 0.85


def compute_iou(box_a, box_b):
    """Compute Intersection over Union for two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    if union == 0:
        return 0.0
    return inter / union


class StabilityGate:
    def __init__(self):
        self._prev_box = None
        self._stable_since = None
        self._suppressed_regions = []  # list of (box, expiry_time)

    def _is_suppressed(self, box, now):
        """Check if this region is within the dedup suppression window."""
        self._suppressed_regions = [
            (b, t) for b, t in self._suppressed_regions if t > now
        ]
        for suppressed_box, _ in self._suppressed_regions:
            if compute_iou(box, suppressed_box) >= IOU_STABILITY_MIN:
                return True
        return False

    def update(self, detection):
        """Process a single frame detection result.

        Args:
            detection: dict with 'box' key ([x1,y1,x2,y2]) or None if no label detected.

        Returns:
            box if OCR should be triggered, None otherwise.
        """
        now = time.monotonic()

        if detection is None or "box" not in detection:
            self._prev_box = None
            self._stable_since = None
            return None

        box = detection["box"]

        if self._is_suppressed(box, now):
            return None

        if self._prev_box is not None:
            iou = compute_iou(box, self._prev_box)
            if iou >= IOU_STABILITY_MIN:
                if self._stable_since is None:
                    self._stable_since = now
                elif now - self._stable_since >= STABILITY_THRESHOLD_SEC:
                    # Stable long enough — fire trigger
                    self._suppressed_regions.append(
                        (box, now + DEDUP_WINDOW_SEC)
                    )
                    self._prev_box = None
                    self._stable_since = None
                    return box
            else:
                # Box moved too much, reset
                self._stable_since = None
        else:
            self._stable_since = None

        self._prev_box = box
        return None
