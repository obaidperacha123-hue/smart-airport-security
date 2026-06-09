"""
tracker.py  –  Member C
Object Tracking with Pure IoU Geometry + Stationary Timer

Responsibilities:
  - Assign persistent IDs to detected luggage bounding boxes using pure IoU tracking
  - Detect when a bag has been stationary for longer than STATIONARY_THRESHOLD
  - Expose per-track state so the alert engine can decide what to alert

Dependencies:
  pip install opencv-python numpy
"""

import time
import numpy as np


STATIONARY_THRESHOLD_SECONDS = 30   
IOU_MOVE_THRESHOLD           = 0.80 
MAX_AGE                      = 30   

def _ltrb(box: list) -> list:
    x1, y1, a, b = box
    if a <= x1 or b <= y1:
        return [int(x1), int(y1), int(x1 + a), int(y1 + b)]
    return [int(x1), int(y1), int(a), int(b)]
 
 
def _iou(box_a: list, box_b: list) -> float:
    b1 = _ltrb(box_a)
    b2 = _ltrb(box_b)
 
    xa = max(b1[0], b2[0])
    ya = max(b1[1], b2[1])
    xb = min(b1[2], b2[2])
    yb = min(b1[3], b2[3])
 
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area_b = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / float(area_a + area_b - inter)


class BagTracker:

    def __init__(self,
                 stationary_threshold: float = STATIONARY_THRESHOLD_SECONDS,
                 iou_move_threshold: float   = IOU_MOVE_THRESHOLD,
                 max_age: int                = MAX_AGE):
        self.stationary_threshold = stationary_threshold
        self.iou_move_threshold   = iou_move_threshold
        self.max_age              = max_age
        
        self.next_id = 1
        # track_id -> {"last_bbox": [x1,y1,x2,y2], "stationary_since": float|None, "age": int}
        self._state: dict = {}

    def update(self, frame: np.ndarray, detections: list) -> list:
        """
        Parameters
        ----------
        frame       : BGR numpy array (the enhanced CCTV frame)
        detections  : list of ([x,y,w,h], confidence, class_name)

        Returns
        -------
        List of track dicts matching expected fields perfectly.
        """
        now = time.time()
        results = []
        
        # 1. Increment age for all existing tracks
        for tid in list(self._state.keys()):
            self._state[tid]["age"] += 1

        matched_tracks = set()
        
        # 2. Match new detections to existing tracking states based on highest IoU
        for det in detections:
            bbox_xywh, conf, clss = det
            # Convert incoming xywh to ltrb layout [x1, y1, x2, y2]
            x, y, w, h = bbox_xywh
            ltrb = [int(x), int(y), int(x + w), int(y + h)]
            
            best_id = None
            best_iou = 0.3  # Minimum baseline IoU threshold for matching
            
            for tid, track in self._state.items():
                if tid in matched_tracks:
                    continue
                overlap = _iou(track["last_bbox"], ltrb)
                if overlap > best_iou:
                    best_iou = overlap
                    best_id = tid
            
            if best_id is not None:
                # Update matched track state
                matched_tracks.add(best_id)
                s = self._state[best_id]
                s["age"] = 0  # Reset survival age
                
                # Check stationarity against previous position
                iou_stationary = _iou(s["last_bbox"], ltrb)
                if iou_stationary >= self.iou_move_threshold:
                    if s["stationary_since"] is None:
                        s["stationary_since"] = now
                else:
                    s["stationary_since"] = None
                
                s["last_bbox"] = ltrb
                track_id = best_id
            else:
                
                track_id = self.next_id
                self.next_id += 1
                self._state[track_id] = {
                    "last_bbox": ltrb,
                    "stationary_since": None,
                    "age": 0
                }
            
            s = self._state[track_id]
            stationary_secs = (now - s["stationary_since"]) if s["stationary_since"] is not None else 0.0
            is_stationary = s["stationary_since"] is not None
            is_alert = stationary_secs >= self.stationary_threshold

            results.append({
                "track_id"           : track_id,
                "bbox_ltrb"          : ltrb,
                "stationary"         : is_stationary,
                "stationary_seconds" : round(stationary_secs, 1),
                "alert"              : is_alert,
            })

        self._state = {tid: s for tid, s in self._state.items() if s["age"] <= self.max_age}

        return results

    def reset(self) -> None:
        self._state.clear()
        self.next_id = 1
