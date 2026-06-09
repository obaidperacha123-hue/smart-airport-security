# Member C — Tracking, Face Recognition & Alert Engine

## Overview

This module implements **three CV tasks** for the Smart Airport Unattended Luggage Detection system:

| CV Task | File | Technique |
|---|---|---|
| Object Tracking | `tracker.py` | Greedy IoU multi-object tracker + persistent IDs |
| Background Modelling | `tracker.py` | MOG2 Gaussian Mixture Model foreground/background separation |
| Face Detection & Recognition | `face_recognizer.py` | InsightFace buffalo_sc — ArcFace 512-d embeddings + cosine similarity |

The `alert_engine.py` combines the outputs of both into structured `Alert` objects that Member D's FastAPI server forwards to Member E's React frontend.

---

## File Structure

```
src/tracking/
├── __init__.py          # Public exports: BagTracker, FaceRecognizer, AlertEngine, Alert
├── tracker.py           # BagTracker — IoU tracker + MOG2 + stationary timer
├── face_recognizer.py   # FaceRecognizer — InsightFace ArcFace detection & matching
└── alert_engine.py      # AlertEngine — unified alert logic and deduplication

test_tracking.py         # Standalone smoke-test (webcam or video file)
data/
└── face_db.pkl          # Enrolled face database (auto-created on first enrolment)
```

---

## Dependencies

All dependencies are in the shared `requirements.txt` at the repo root.

```bash
pip install -r requirements.txt
```

To install Member C's dependencies individually:

```bash
pip install opencv-python numpy insightface onnxruntime
```

> **Note:** InsightFace will auto-download the `buffalo_sc` ONNX model bundle (~85 MB) on first run into `~/.insightface/models/`. This requires an internet connection the first time only.

---

## Quick Start — Smoke Test

```bash
# From the repo root:
python test_tracking.py              # uses webcam (device 0)
python test_tracking.py sample.mp4  # uses a video file
python test_tracking.py 1           # uses webcam device 1
```

### Keyboard shortcuts during the test

| Key | Action |
|---|---|
| `Q` | Quit |
| `A` | Print all currently active alerts to the terminal |
| `E` | Enrol your own face as `TestOwner` from the live webcam frame |

---

## How to Trigger Each Alert Type

### UNATTENDED_BAG
A stub bag is placed in the centre of the frame and barely moves. After **10 seconds** (threshold reduced from 30 s for quick testing), the bounding box turns red and the alert fires.

### OWNER_LEFT
1. Press **E** while your face is clearly visible to enrol yourself as `TestOwner`.
2. The system associates your face with the nearest bag.
3. Walk out of frame (or cover your face) for more than **8 seconds**.
4. An `OWNER_LEFT` alert fires once the timer expires.

### ACCESS_VIOLATION
Point the camera at any face that has **not** been enrolled in the database. An `ACCESS_VIOLATION` alert fires immediately for each new unknown face detected.

---

## How It Fits Into the System

```
Member B (detector.py)
    │  detections: [([x, y, w, h], conf, label), ...]
    ▼
BagTracker.update(frame, detections)
    │  bag_tracks: [{track_id, bbox_ltrb, stationary, stationary_seconds, alert, fg_ratio}]
    │  fg_mask:    uint8 H×W foreground mask (255=FG, 127=shadow, 0=BG)
    ▼
FaceRecognizer.process_frame(frame, bag_tracks)
    │  face_results:  [{bbox_ltrb, identity, similarity, embedding}]
    │  absent_alerts: [{track_id, owner, absent_seconds}]   ← via get_owner_absent_alerts()
    ▼
AlertEngine.process(bag_tracks, face_results, absent_alerts)
    │  new Alert objects  →  forwarded to Member D's FastAPI /ws WebSocket
    ▼
Member D (server.py)  →  Member E (React frontend)
```

---

## API Reference

### BagTracker

```python
from src.tracking import BagTracker

tracker = BagTracker(
    stationary_threshold=30,   # seconds before UNATTENDED_BAG alert fires
    iou_move_threshold=0.80,   # IoU >= this → bag considered "not moved"
    max_age=30,                # frames before unmatched track is pruned
)

# Call once per frame — returns (bag_tracks, fg_mask)
bag_tracks, fg_mask = tracker.update(frame, detections)

# bag_tracks is a list of dicts:
# {
#   "track_id"           : int,
#   "bbox_ltrb"          : [x1, y1, x2, y2],
#   "stationary"         : bool,
#   "stationary_seconds" : float,
#   "alert"              : bool,   # True when stationary >= threshold
#   "fg_ratio"           : float,  # fraction of bbox area covered by foreground
# }

tracker.reset()   # clear all tracks and reinitialise MOG2 (call on scene change)
```

`detections` format matches Member B's `LuggageDetector` output:
```python
[([x, y, w, h], confidence_float, class_name_str), ...]
```

---

### FaceRecognizer

```python
from src.tracking import FaceRecognizer

face_rec = FaceRecognizer(
    db_path="data/face_db.pkl",
    recognition_threshold=0.50,   # cosine similarity cutoff for a valid match
    owner_absent_threshold=20.0,  # seconds before OWNER_LEFT alert fires
    association_radius_px=300,    # max pixel distance face↔bag to link them
    face_skip_frames=1,           # set to 3 for ~3× speedup on slow hardware
)
face_rec.load_db()   # load enrolled database from disk

# Call once per frame
face_results  = face_rec.process_frame(frame, bag_tracks)
absent_alerts = face_rec.get_owner_absent_alerts()

# face_results is a list of dicts:
# {
#   "bbox_ltrb"  : [x1, y1, x2, y2],
#   "identity"   : str | None,     # matched name, or None = unknown
#   "similarity" : float,          # best cosine similarity (0–1)
#   "embedding"  : np.ndarray,     # 512-d ArcFace vector
# }

# absent_alerts is a list of dicts:
# {
#   "track_id"      : int,
#   "owner"         : str,
#   "absent_seconds": float,
# }
```

#### Enrolling a Person

Called automatically by Member D's `POST /enrol` endpoint. Can also be used standalone:

```python
import cv2
from src.tracking import FaceRecognizer

face_rec = FaceRecognizer()
face_rec.load_db()

img = cv2.imread("path/to/person_photo.jpg")
success = face_rec.enrol("Alice", img)   # saves DB automatically
print("Enrolled:", success)              # False if no face detected in image

# Bulk enrolment (avoids N disk writes):
for img_path in photo_list:
    face_rec.enrol("Alice", cv2.imread(img_path), auto_save=False)
face_rec.save_db()   # single write at the end
```

---

### AlertEngine

```python
from src.tracking import AlertEngine, Alert

engine = AlertEngine(detect_unknown_faces=True)

# Call once per frame
new_alerts = engine.process(bag_tracks, face_results, absent_alerts)
# Returns a list of new Alert objects raised this frame (may be empty).
# Alerts already active are updated in-place, not re-raised.

# Query active alerts (returns list of dicts)
active = engine.active_alerts()

# Acknowledge (dismiss) an alert from the frontend
final_state = engine.acknowledge(track_id, alert_type)
# Returns the final Alert dict with acknowledged=True, or None if not found.
# For OWNER_LEFT, also call face_rec.reset_ownership(track_id) so the
# owner can be re-associated if they return.

# Clear everything (e.g. end of shift / scene reset)
engine.clear_all()
```

#### Alert Object Fields

```python
@dataclass
class Alert:
    alert_type        : Literal["UNATTENDED_BAG", "OWNER_LEFT", "ACCESS_VIOLATION"]
    track_id          : int          # negative for ACCESS_VIOLATION pseudo-IDs
    timestamp         : float        # time.time() when alert was raised
    owner             : str | None   # set for OWNER_LEFT alerts
    absent_seconds    : float        # set for OWNER_LEFT alerts
    stationary_seconds: float        # set for UNATTENDED_BAG alerts
    bbox_ltrb         : list         # [x1, y1, x2, y2]
    acknowledged      : bool         # True after acknowledge() is called
```

---

## Configuration Constants

All thresholds are defined at the top of each file and can be changed without touching the logic:

| File | Constant | Default | Meaning |
|---|---|---|---|
| `tracker.py` | `STATIONARY_THRESHOLD_SECONDS` | `30` | Seconds stationary before UNATTENDED_BAG alert |
| `tracker.py` | `IOU_MOVE_THRESHOLD` | `0.80` | IoU above which bag counts as "not moved" |
| `tracker.py` | `MAX_AGE` | `30` | Frames of grace before dropping an unmatched track (~3 s at 10 FPS) |
| `tracker.py` | `MOG2_HISTORY` | `500` | Frames MOG2 uses to build background model |
| `tracker.py` | `MOG2_VAR_THRESHOLD` | `40` | Pixel variance threshold for MOG2 foreground detection |
| `face_recognizer.py` | `RECOGNITION_THRESHOLD` | `0.50` | Cosine similarity cutoff for a valid face match |
| `face_recognizer.py` | `OWNER_ABSENT_THRESHOLD` | `20.0` | Seconds owner absent before OWNER_LEFT alert |
| `face_recognizer.py` | `ASSOCIATION_RADIUS_PX` | `300` | Max pixel distance face↔bag to link them |

---

## Alert Types Summary

| Type | Trigger | track_id |
|---|---|---|
| `UNATTENDED_BAG` | Bag stationary ≥ `STATIONARY_THRESHOLD_SECONDS` | Positive integer (real bag ID) |
| `OWNER_LEFT` | Identified owner absent ≥ `OWNER_ABSENT_THRESHOLD` | Positive integer (real bag ID) |
| `ACCESS_VIOLATION` | Unknown face detected in frame | Negative integer (pseudo-ID) |

---

## Known Issues & Notes

1. **README threshold mismatch:** The `README_member_c.md` configuration table lists `RECOGNITION_THRESHOLD = 0.45`, but the actual code uses `0.50`. The code value is correct — it was intentionally raised to reduce false positives. The README table needs updating.

2. **`active_track_ids` unused variable:** In `alert_engine.py` line 106, `active_track_ids` is computed but never used. It was likely intended for cleaning up stale `UNATTENDED_BAG` alerts when a bag disappears from the scene (analogous to the `OWNER_LEFT` cleanup in Step 4). Currently, acknowledged or auto-cleared `UNATTENDED_BAG` alerts are never pruned unless `acknowledge()` is called. This is safe but means stale UNATTENDED_BAG entries persist until the frontend acknowledges them. Member D should call `engine.acknowledge(track_id, "UNATTENDED_BAG")` when a bag disappears from `bag_tracks`.

3. **`ACCESS_VIOLATION` alerts are never auto-cleaned:** Unlike `OWNER_LEFT`, `ACCESS_VIOLATION` alerts stay in the active table until explicitly `acknowledge()`-d. The spatial deduplication prevents re-firing for the same position, but if an unknown person moves across the frame they will accumulate multiple alerts. This is intentional per the design notes.

4. **Frame-skip on skipped frames still refreshes `last_seen`:** The skip-frame logic in `FaceRecognizer.process_frame()` reuses cached `face_results` but still calls the ownership timestamp refresh. This is correct — it prevents the absent-owner timer from firing spuriously during skipped frames.

---

## Performance Notes

- InsightFace `buffalo_sc` on CPU: typically **8–15 FPS** at 640×640 depending on hardware.
- To stay at or above 10 FPS system-wide, set `face_skip_frames=3` — face recognition runs every 3 frames while tracking runs every frame.
- MOG2 and IoU matching are negligible cost compared to the neural inference.
- The `fg_ratio` field in each track dict allows Member D / Member E to optionally filter ghost detections (objects with `fg_ratio < 0.1` are likely false positives from the detector).
