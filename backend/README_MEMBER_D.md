# Member D — FastAPI Backend

## Files Owned

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app, all route handlers |
| `pipeline.py` | Orchestrates B's and C's CV modules per-frame |
| `face_db.py` | Pickle-backed ArcFace embedding store |
| `schemas.py` | Pydantic models shared across the whole backend |

---

## Quick Start

```bash
# From the /backend directory
pip install fastapi uvicorn[standard] opencv-python-headless numpy

# Run the dev server
python main.py
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

The server **starts successfully even without B's or C's modules committed yet**.  
Each missing module logs a `⚠` warning and falls back to a pass-through.

---

## Endpoints

### `GET /health`
Liveness probe. Returns pipeline readiness + face DB count + active WebSocket count.

```json
{
  "status": "ok",
  "pipeline_ready": true,
  "face_db_count": 3,
  "active_ws_connections": 1
}
```

---

### `GET /ws` — WebSocket Live Feed

**Client → Server (per frame):**
```json
{ "frame_b64": "<base64-encoded JPEG string>" }
```

**Server → Client (`FrameResult`):**
```json
{
  "frame_id": 42,
  "timestamp": 1718000000.123,
  "tracks": [
    { "track_id": 3, "class_name": "backpack", "confidence": 0.91,
      "bbox": {"x1": 120, "y1": 80, "x2": 260, "y2": 320},
      "stationary_seconds": 35.2, "is_alert": true }
  ],
  "faces": [
    { "name": "alice", "confidence": 0.87,
      "bbox": {"x1": 400, "y1": 50, "x2": 480, "y2": 160},
      "is_authorized": true }
  ],
  "alerts": [
    { "alert_id": "uuid4", "alert_type": "unattended_bag", "track_id": 3,
      "timestamp": 1718000035.0, "duration_seconds": 35.2,
      "message": "Backpack (track 3) stationary for 35s",
      "thumbnail_b64": "<base64 crop>" }
  ],
  "enhanced_frame_b64": "<base64 JPEG — sent every 5th frame, else null>"
}
```

**Member E integration:**  
Connect with `new WebSocket("ws://localhost:8000/ws")`, send base64 frames from `getUserMedia` canvas, draw `tracks` and `faces` as bounding boxes, push `alerts` to the alert dashboard.

---

### `POST /upload`

Upload a `.jpg`, `.png`, `.mp4`, `.avi`, or `.mov` file.  
Videos are downsampled to ~5 fps for processing.

**Form field:** `file` (multipart/form-data)

**Response (`UploadResponse`):**
```json
{
  "job_id": "uuid4",
  "filename": "terminal2_cam3.mp4",
  "total_frames": 1800,
  "processed_frames": 360,
  "duration_seconds": 14.7,
  "alerts": [ ... ],
  "summary": {
    "unique_tracked_objects": 5,
    "unique_persons_identified": 2,
    "alert_breakdown": { "unattended_bag": 2 },
    "frames_with_alerts": 18,
    "total_frames_processed": 360
  }
}
```

---

### `POST /enrol`

Enrol airport staff into the face recognition database.

**Form fields:**
- `name` (string) — full name, e.g. `"Alice Mahmoud"`
- `images` (files) — one or more `.jpg` / `.png` reference photos

**Response (`EnrolResponse`):**
```json
{
  "success": true,
  "name": "Alice Mahmoud",
  "embeddings_count": 3,
  "message": "Enrolled 3 image(s) for 'Alice Mahmoud'."
}
```

---

### `GET /enrol`
List all enrolled persons → `{ "persons": {"alice mahmoud": 3}, "total_persons": 1 }`

### `DELETE /enrol/{name}`
Remove a person → `{ "success": true, "message": "..." }`

---

## Integration Contract with Members B & C

### What D expects from **Member B** (`enhancement.py`, `detection.py`)

```python
# enhancement.py
def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """CLAHE + bilateral filter. Returns same-shape BGR array."""

# detection.py
def detect_objects(frame: np.ndarray) -> list[dict]:
    """YOLOv8 inference. Each dict must contain:
       class_name: str     — 'backpack' | 'handbag' | 'suitcase'
       confidence: float   — 0.0–1.0
       bbox: list[float]   — [x1, y1, x2, y2] in pixel coords
    """
```

### What D expects from **Member C** (`tracker.py`, `face_recognition_module.py`, `alert_logic.py`)

```python
# tracker.py
class DeepSORTTracker: ...

def update_tracks(tracker: DeepSORTTracker, detections: list, frame: np.ndarray) -> list[dict]:
    """Each dict must contain:
       id: int                   — persistent track ID
       class_name: str
       confidence: float
       bbox: list[float]         — [x1, y1, x2, y2]
       stationary_seconds: float — seconds the object hasn't moved
    """

# face_recognition_module.py
class InsightFaceMatcher:
    def detect_and_match(self, frame: np.ndarray, face_db: FaceDatabase) -> list[dict]:
        """Each dict: name, confidence, bbox [x1,y1,x2,y2], is_authorized (bool)"""

    def extract_embedding(self, image: np.ndarray) -> np.ndarray | None:
        """Returns 512-dim ArcFace embedding, or None if no face found."""

# alert_logic.py
class AlertEngine:
    def __init__(self, stationary_threshold_seconds: float): ...

    def evaluate(
        self,
        tracks: list[TrackedObject],
        faces: list[FaceMatch],
        frame: np.ndarray,
    ) -> list[AlertEvent]: ...
```

> **Important:** If a module is missing, `pipeline.py` silently falls back — the server never crashes. This means the whole team can run the backend independently while modules are still in development.

---

## Face DB

- Stored at `data/face_db.pkl` (auto-created on first enrolment)
- Schema: `{ "name_lowercase": [embedding_array, ...] }`
- Matching: cosine similarity with default threshold `0.40`
- Auto-saved after every enrolment or deletion

---

## File Layout

```
backend/
├── main.py                  ← Member D  (this file — FastAPI server)
├── pipeline.py              ← Member D  (CV module orchestrator)
├── face_db.py               ← Member D  (face DB manager)
├── schemas.py               ← Member D  (Pydantic models)
│
├── enhancement.py           ← Member B  (CLAHE + bilateral)
├── detection.py             ← Member B  (YOLOv8)
├── morphology.py            ← Member B  (optional mask cleanup)
│
├── tracker.py               ← Member C  (DeepSORT + stationary timer)
├── face_recognition_module.py ← Member C (InsightFace)
├── alert_logic.py           ← Member C  (alert engine)
│
├── requirements.txt         ← Member A
└── data/
    └── face_db.pkl          ← auto-generated on first enrolment
```
