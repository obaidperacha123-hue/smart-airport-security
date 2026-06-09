# Member C — Tracking & Face Recognition

## Files

```
src/tracking/
├── __init__.py          # public exports
├── tracker.py           # BagTracker  (DeepSORT + stationary timer)
├── face_recognizer.py   # FaceRecognizer (InsightFace ArcFace)
└── alert_engine.py      # AlertEngine (unattended bag + owner left scene)

test_tracking.py         # standalone smoke-test (webcam or video file)
```

## Dependencies

These are already in the repo's `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

If you need them individually:

```bash
pip install deep-sort-realtime insightface onnxruntime opencv-python numpy
```

InsightFace will auto-download the `buffalo_sc` ONNX model bundle (~85 MB)
on first run into `~/.insightface/models/`.

## Quick Test

```bash
# From the repo root:
python test_tracking.py              # uses webcam
python test_tracking.py sample.mp4  # uses video file
```

Press **Q** to quit · Press **A** to print active alerts to the terminal.

## How It Fits Into the System

```
Member B (detector.py)
    │  detections: [([x,y,w,h], conf, label), ...]
    ▼
BagTracker.update(frame, detections)
    │  bag_tracks: [{track_id, bbox_ltrb, stationary, stationary_seconds, alert}]
    ▼
FaceRecognizer.process_frame(frame, bag_tracks)
    │  face_results, absent_alerts
    ▼
AlertEngine.process(bag_tracks, face_results, absent_alerts)
    │  new Alert objects  →  forwarded to Member D's FastAPI /ws WebSocket
    ▼
Member D (server.py)  →  Member E (React frontend)
```

## Alert Types

| Type | Trigger |
|---|---|
| `UNATTENDED_BAG` | Bag stationary > 30 s (configurable) |
| `OWNER_LEFT` | Identified owner absent > 20 s (configurable) |
| `ACCESS_VIOLATION` | Unknown face detected in frame |

## Enrolling a Person (for testing)

```python
import cv2
from src.tracking import FaceRecognizer

fr = FaceRecognizer()
fr.load_db()
img = cv2.imread("path/to/person_photo.jpg")
success = fr.enrol("Alice", img)
print("Enrolled:", success)
```

This is called automatically by Member D's `POST /enrol` endpoint.

## Configuration

All thresholds are constants at the top of each file:

| File | Constant | Default | Meaning |
|---|---|---|---|
| tracker.py | `STATIONARY_THRESHOLD_SECONDS` | 30 | Seconds before unattended alert |
| tracker.py | `IOU_MOVE_THRESHOLD` | 0.80 | IoU above which bag is "not moved" |
| face_recognizer.py | `RECOGNITION_THRESHOLD` | 0.45 | Cosine similarity for a match |
| face_recognizer.py | `OWNER_ABSENT_THRESHOLD` | 20 | Seconds owner can be absent |
| face_recognizer.py | `ASSOCIATION_RADIUS_PX` | 300 | Max px distance face↔bag to link them |
