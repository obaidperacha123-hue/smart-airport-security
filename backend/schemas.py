"""
schemas.py — Pydantic models for every request/response in the API.
Member D owns this file.  Members B & C use these same models so that
pipeline output is always serialisable through the WebSocket / REST endpoints.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ── Primitives ────────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """Pixel-space bounding box (top-left / bottom-right corners)."""
    x1: float
    y1: float
    x2: float
    y2: float


# ── CV pipeline outputs ───────────────────────────────────────────────────────

class TrackedObject(BaseModel):
    """One DeepSORT track (single bag/backpack/suitcase across frames)."""
    track_id: int
    class_name: str          # "backpack" | "handbag" | "suitcase"
    confidence: float = 0.0
    bbox: BoundingBox
    stationary_seconds: float = 0.0
    is_alert: bool = False   # True when stationary_seconds >= threshold


class FaceMatch(BaseModel):
    """InsightFace detection + DB look-up result for one face in a frame."""
    name: str                # enrolled name or "Unknown"
    confidence: float        # ArcFace cosine similarity (0–1)
    bbox: BoundingBox
    is_authorized: bool      # True when name found in face DB


# ── Alert system ──────────────────────────────────────────────────────────────

class AlertType(str, Enum):
    UNATTENDED_BAG    = "unattended_bag"     # bag stationary > threshold, owner gone
    OWNER_LEFT_SCENE  = "owner_left_scene"   # matched person no longer in frame
    UNAUTHORIZED_ACCESS = "unauthorized_access"  # unrecognised face in restricted zone


class AlertEvent(BaseModel):
    """A single security alert emitted by the alert engine."""
    alert_id: str
    alert_type: AlertType
    track_id: Optional[int] = None   # luggage track that triggered the alert
    timestamp: float                 # Unix epoch
    duration_seconds: float = 0.0
    message: str
    thumbnail_b64: Optional[str] = None   # base64 JPEG crop of the offending region


# ── Per-frame result (WebSocket payload) ──────────────────────────────────────

class FrameResult(BaseModel):
    """
    Everything the frontend needs to render bounding boxes, face labels,
    and alert cards for a single processed frame.
    Sent as JSON over the WebSocket on every frame (or batch of frames).
    """
    frame_id: int
    timestamp: float
    tracks: List[TrackedObject] = []
    faces: List[FaceMatch] = []
    alerts: List[AlertEvent] = []
    # Enhanced preview frame — only sent every N frames to save bandwidth
    enhanced_frame_b64: Optional[str] = None


# ── REST endpoint responses ───────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    face_db_count: int
    active_ws_connections: int


class EnrolResponse(BaseModel):
    success: bool
    name: str
    embeddings_count: int   # total embeddings stored for this person after enrolment
    message: str


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    total_frames: int
    processed_frames: int
    duration_seconds: float
    alerts: List[AlertEvent]
    summary: Dict[str, Any]  # aggregated stats: unique tracks, alert breakdown, etc.
