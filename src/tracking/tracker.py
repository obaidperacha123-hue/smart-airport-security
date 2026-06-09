"""
tracker.py  –  Member C
Object Tracking with DeepSORT + Stationary Timer

Responsibilities:
  - Assign persistent IDs to detected luggage bounding boxes using DeepSORT
  - Detect when a bag has been stationary for longer than STATIONARY_THRESHOLD
  - Expose per-track state so the alert engine can decide what to alert

Dependencies:
  pip install deep-sort-realtime opencv-python numpy
"""

import time
import numpy as np
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.structures import Detection

# ── Config ──────────────────────────────────────────────────────────────────
STATIONARY_THRESHOLD_SECONDS = 30   # seconds before a bag is flagged
IOU_MOVE_THRESHOLD           = 0.80 # IoU > this means "bag hasn't moved"
MAX_AGE                      = 30   # frames a track survives without a match
# ─────────────────────────────────────────────────────────────────────────────


def _iou(box_a: list, box_b: list) -> float:
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / float(area_a + area_b - inter)


class BagTracker:
    """
    Wraps DeepSORT and maintains per-track stationarity state.

    Usage
    -----
    tracker = BagTracker()

    # Inside your frame loop:
    detections = [([x1,y1,w,h], confidence, "luggage"), ...]   # YOLO output
    tracks     = tracker.update(frame, detections)

    # Each track dict:
    # {
    #   "track_id"   : int,
    #   "bbox_ltrb"  : [x1,y1,x2,y2],
    #   "stationary" : bool,
    #   "stationary_seconds": float,
    #   "alert"      : bool      # True when stationary > threshold
    # }
    """

    def __init__(self,
                 stationary_threshold: float = STATIONARY_THRESHOLD_SECONDS,
                 iou_move_threshold: float   = IOU_MOVE_THRESHOLD,
                 max_age: int                = MAX_AGE):
        self.stationary_threshold = stationary_threshold
        self.iou_move_threshold   = iou_move_threshold

        self._deepsort = DeepSort(max_age=max_age, embedder=None)

        # track_id -> {"last_bbox": [...], "stationary_since": float | None}
        self._state: dict = defaultdict(lambda: {
            "last_bbox": None,
            "stationary_since": None,
        })

    # ── public API ──────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, detections: list) -> list:
        """
        Parameters
        ----------
        frame       : BGR numpy array (the enhanced CCTV frame)
        detections  : list of ([x1,y1,w,h], confidence, class_name)
                      – the raw output from the YOLOv8 detector

        Returns
        -------
        List of track dicts (see class docstring).
        """
        detections_with_feats = []
        for det in detections:
            bbox, conf, clss = det
            # DeepSORT expects: Detection(tlwh, conference, feature, class_id)
            native_det = Detection(bbox, conf, np.ones(128, dtype=np.float32) * 0.1, clss)
            detections_with_feats.append(native_det)

        raw_tracks = self._deepsort.update_tracks(detections_with_feats, frame=None)
        now        = time.time()
        results    = []

        for t in raw_tracks:
            if not t.is_confirmed():
                continue

            tid  = t.track_id
            ltrb = list(map(int, t.to_ltrb()))   # [x1,y1,x2,y2]
            s    = self._state[tid]

            # ── stationarity check ─────────────────────────────────────────
            if s["last_bbox"] is not None:
                iou = _iou(s["last_bbox"], ltrb)
                if iou >= self.iou_move_threshold:
                    # bag hasn't moved — start / continue the timer
                    if s["stationary_since"] is None:
                        s["stationary_since"] = now
                else:
                    # bag moved — reset timer
                    s["stationary_since"] = None
            s["last_bbox"] = ltrb

            stationary_secs = (
                now - s["stationary_since"]
                if s["stationary_since"] is not None
                else 0.0
            )
            is_stationary = s["stationary_since"] is not None
            is_alert      = stationary_secs >= self.stationary_threshold

            results.append({
                "track_id"           : tid,
                "bbox_ltrb"          : ltrb,
                "stationary"         : is_stationary,
                "stationary_seconds" : round(stationary_secs, 1),
                "alert"              : is_alert,
            })

        return results

    def reset(self) -> None:
        """Clear all track state (call between video files)."""
        self._deepsort = DeepSort(max_age=self._deepsort.max_age, embedder=None)
        self._state.clear()