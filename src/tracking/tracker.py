"""
tracker.py  –  Member C
Object Tracking with IoU Geometry + Stationary Timer + MOG2 Background Subtraction
 
CV tasks implemented in this file
----------------------------------
  1. Object tracking        – greedy IoU-based multi-object tracker assigns
                              persistent IDs to detected luggage bounding boxes
  2. Background modelling   – MOG2 Gaussian Mixture Model separates foreground
     & moving-object           (moving objects) from background; the foreground
     detection                 mask is exposed per-frame for optional use by
                              Member D's server and Member E's frontend overlay
 
Responsibilities
----------------
  - Initialise a per-scene background model using OpenCV's MOG2
  - Assign persistent track IDs to luggage detections via IoU matching
  - Time how long each bag has been stationary (IoU between consecutive
    frames stays above IOU_MOVE_THRESHOLD)
  - Fire an alert flag once a bag exceeds STATIONARY_THRESHOLD_SECONDS
 
Why IoU matching instead of DeepSORT
--------------------------------------
  DeepSORT requires a separate appearance re-ID network (~150 MB) and
  Kalman-filter state prediction.  For a fixed overhead CCTV camera the
  scene is stable and bags move slowly, so greedy IoU matching is
  sufficient and keeps the tracker dependency-free.
 
Dependencies
------------
  pip install opencv-python numpy
"""
 
import time
import cv2
import numpy as np
 
 
# ── Config ────────────────────────────────────────────────────────────────────
STATIONARY_THRESHOLD_SECONDS = 30    # seconds without movement → UNATTENDED alert
IOU_MOVE_THRESHOLD           = 0.80  # IoU >= this  →  bag considered "not moved"
MAX_AGE                      = 30    # frames before an unmatched track is pruned
                                     # assumes ~10 FPS  →  3 s of tolerance
                                     # increase to 90 for 30 FPS cameras
MOG2_HISTORY                 = 500   # frames MOG2 uses to build background model
MOG2_VAR_THRESHOLD           = 40    # pixel variance threshold for MOG2
MOG2_DETECT_SHADOWS          = True  # mark shadow pixels grey (value 127)
# ─────────────────────────────────────────────────────────────────────────────
 
 
# ── Geometry helpers ──────────────────────────────────────────────────────────
 
def _to_ltrb(box: list) -> list:
    """
    Normalise a bounding box to [x1, y1, x2, y2] (left-top-right-bottom).
 
    Accepts either:
      [x1, y1, x2, y2]  – already ltrb  (x2 > x1)
      [x,  y,  w,  h]   – xywh format   (w is a size, so x+w > x)
    """
    x1, y1, a, b = box
    if a <= x1 or b <= y1:           # third/fourth values are width/height
        return [int(x1), int(y1), int(x1 + a), int(y1 + b)]
    return [int(x1), int(y1), int(a), int(b)]
 
 
def _iou(box_a: list, box_b: list) -> float:
    """Intersection-over-Union of two bounding boxes (xywh or ltrb)."""
    b1 = _to_ltrb(box_a)
    b2 = _to_ltrb(box_b)
 
    xa = max(b1[0], b2[0]);  ya = max(b1[1], b2[1])
    xb = min(b1[2], b2[2]);  yb = min(b1[3], b2[3])
 
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
 
    area_a = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area_b = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union  = area_a + area_b - inter
    return inter / float(union) if union > 0 else 0.0
 
 
# ─────────────────────────────────────────────────────────────────────────────
 
class BagTracker:
    """
    Multi-object luggage tracker with background modelling.
 
    Integrates two CV techniques:
      • MOG2 background subtraction  – per-frame foreground mask
      • Greedy IoU tracker            – persistent bag IDs + stationary timer
 
    Typical call sequence
    ---------------------
    tracker = BagTracker()
 
    # once per frame:
    bag_tracks, fg_mask = tracker.update(frame, detections)
 
    bag_tracks — list of dicts:
    {
      "track_id"           : int,
      "bbox_ltrb"          : [x1, y1, x2, y2],
      "stationary"         : bool,
      "stationary_seconds" : float,
      "alert"              : bool,   # True when stationary >= threshold
      "fg_ratio"           : float,  # fraction of bbox covered by foreground
    }
 
    fg_mask — uint8 numpy array (same H×W as frame):
      255 = foreground  |  127 = shadow  |  0 = background
    """
 
    def __init__(self,
                 stationary_threshold: float = STATIONARY_THRESHOLD_SECONDS,
                 iou_move_threshold:   float = IOU_MOVE_THRESHOLD,
                 max_age:              int   = MAX_AGE,
                 mog2_history:         int   = MOG2_HISTORY,
                 mog2_var_threshold:   float = MOG2_VAR_THRESHOLD,
                 mog2_detect_shadows:  bool  = MOG2_DETECT_SHADOWS):
 
        self.stationary_threshold = stationary_threshold
        self.iou_move_threshold   = iou_move_threshold
        self.max_age              = max_age

        # Store MOG2 params so reset() can recreate with the same settings
        self._mog2_history        = mog2_history
        self._mog2_var_threshold  = mog2_var_threshold
        self._mog2_detect_shadows = mog2_detect_shadows
 
        # ── MOG2 background subtractor ────────────────────────────────────
        # BackgroundSubtractorMOG2 fits a Gaussian mixture model per pixel.
        # It automatically adapts to slow lighting changes (ideal for CCTV).
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history        = mog2_history,
            varThreshold   = mog2_var_threshold,
            detectShadows  = mog2_detect_shadows,
        )
 
        # 3×3 elliptical kernel for morphological cleanup of the fg mask
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
 
        self.next_id = 1
        # track_id -> {
        #   "last_bbox"       : [x1, y1, x2, y2],
        #   "stationary_since": float | None,
        #   "age"             : int
        # }
        self._state: dict = {}
 
    # ── Public API ────────────────────────────────────────────────────────────
 
    def update(self, frame: np.ndarray, detections: list) -> tuple:
        """
        Process one frame: run MOG2, match detections to tracks, update timers.
 
        Parameters
        ----------
        frame      : BGR numpy array — the enhanced CCTV frame from Member B
        detections : list of  ([x, y, w, h], confidence, class_name)
                     format returned by Member B's LuggageDetector
 
        Returns
        -------
        (bag_tracks, fg_mask)
          bag_tracks : list of track dicts (see class docstring)
          fg_mask    : uint8 H×W foreground mask from MOG2
        """
        now = time.time()
 
        # ── Step 1: Background subtraction (MOG2) ────────────────────────
        # Apply MOG2 to the raw frame.  Returns a mask where:
        #   255 = foreground pixel (recently appeared / moving)
        #   127 = shadow pixel (darker than background but same structure)
        #     0 = background pixel
        fg_mask_raw = self._bg_subtractor.apply(frame)
 
        # Clean up noise with morphological opening (erode then dilate).
        # This removes small spurious foreground blobs (compression artefacts,
        # minor lighting flicker) while keeping large moving objects intact.
        fg_mask = cv2.morphologyEx(
            fg_mask_raw, cv2.MORPH_OPEN, self._morph_kernel
        )
 
        # ── Step 2: Age all existing tracks ──────────────────────────────
        for tid in list(self._state.keys()):
            self._state[tid]["age"] += 1
 
        matched_tracks: set = set()
        results: list       = []
 
        # ── Step 3: Match detections to tracks (greedy IoU) ──────────────
        for det in detections:
            bbox_xywh, _conf, _cls = det
            x, y, w, h = bbox_xywh
            ltrb = [int(x), int(y), int(x + w), int(y + h)]
 
            best_id  = None
            best_iou = 0.3      # minimum IoU to accept a match
 
            for tid, track in self._state.items():
                if tid in matched_tracks:
                    continue
                overlap = _iou(track["last_bbox"], ltrb)
                if overlap > best_iou:
                    best_iou = overlap
                    best_id  = tid
 
            if best_id is not None:
                # ── Update existing track ─────────────────────────────
                matched_tracks.add(best_id)
                s        = self._state[best_id]
                s["age"] = 0
 
                iou_same = _iou(s["last_bbox"], ltrb)
                if iou_same >= self.iou_move_threshold:
                    if s["stationary_since"] is None:
                        s["stationary_since"] = now
                else:
                    s["stationary_since"] = None   # bag moved → reset timer
 
                s["last_bbox"] = ltrb
                track_id = best_id
            else:
                # ── New track ──────────────────────────────────────────
                track_id = self.next_id
                self.next_id += 1
                self._state[track_id] = {
                    "last_bbox"       : ltrb,
                    "stationary_since": None,
                    "age"             : 0,
                }
 
            # ── Compute fg_ratio for this bounding box ────────────────
            # Fraction of the bbox area covered by foreground pixels.
            # A high fg_ratio (> 0.3) confirms the object is genuinely
            # present in the foreground and not a ghost detection.
            x1, y1, x2, y2 = ltrb
            # Clamp to frame dimensions
            fh, fw = frame.shape[:2]
            rx1 = max(0, x1);  ry1 = max(0, y1)
            rx2 = min(fw, x2); ry2 = min(fh, y2)
            roi = fg_mask[ry1:ry2, rx1:rx2]
            if roi.size > 0:
                fg_ratio = float(np.count_nonzero(roi)) / roi.size
            else:
                fg_ratio = 0.0
 
            s = self._state[track_id]
            stationary_secs = (
                (now - s["stationary_since"])
                if s["stationary_since"] is not None else 0.0
            )
 
            results.append({
                "track_id"           : track_id,
                "bbox_ltrb"          : ltrb,
                "stationary"         : s["stationary_since"] is not None,
                "stationary_seconds" : round(stationary_secs, 1),
                "alert"              : stationary_secs >= self.stationary_threshold,
                "fg_ratio"           : round(fg_ratio, 3),
            })
 
        # ── Step 4: Prune stale tracks ────────────────────────────────────
        self._state = {
            tid: s
            for tid, s in self._state.items()
            if s["age"] <= self.max_age
        }
 
        return results, fg_mask
 
    def reset(self) -> None:
        """Clear all tracks, reset ID counter, and reinitialise the MOG2 model."""
        self._state.clear()
        self.next_id = 1
        # Use stored instance params, not module-level constants, so custom
        # values passed to __init__ are preserved after reset.
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history      = self._mog2_history,
            varThreshold = self._mog2_var_threshold,
            detectShadows= self._mog2_detect_shadows,
        )
