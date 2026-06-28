"""
pipeline.py — CV Pipeline Orchestrator.
Member D owns this file.

This module wires together:
  Member B's  enhancement.py, detection.py, morphology.py
  Member C's  tracker.py, face_recognition_module.py, alert_logic.py

Every import is wrapped in a try/except so the FastAPI server starts and
responds correctly even when a teammate's module hasn't been pushed yet.
When a module is missing, its step is a no-op pass-through, and a warning
is logged once at startup.

Integration contract (actual module APIs)
─────────────────────────────────────────────────
Member B:
  enhancement.py      → enhance_frame(frame: np.ndarray) -> np.ndarray
  detection.py        → detect_luggage(frame: np.ndarray) -> List[dict]
                          dict keys: "bbox" [x1,y1,x2,y2], "conf", "class", "label"

Member C:
  tracker.py          → class BagTracker
                        .update(frame, detections) -> List[dict]
                          detections format: [([x,y,w,h], conf, class_name), ...]
                          output dict keys: "track_id", "bbox_ltrb" [x1,y1,x2,y2],
                                           "stationary_seconds", "alert"
  face_recognition_module.py
                      → class FaceRecognizer
                        .load_db() -> None
                        .process_frame(frame, bag_tracks) -> List[dict]
                          output dict keys: "bbox_ltrb", "identity", "similarity"
                        .get_owner_absent_alerts() -> List[dict]
  alert_logic.py      → class AlertEngine
                        .process(bag_tracks, face_results, absent_alerts) -> List[Alert]
                          Alert is a dataclass (converted to AlertEvent here)
"""

import base64
import logging
import time
import uuid
from typing import List, Optional

import cv2
import numpy as np

from face_db import FaceDatabase
from schemas import (
    AlertEvent,
    AlertType,
    BoundingBox,
    FaceMatch,
    FrameResult,
    TrackedObject,
)

logger = logging.getLogger(__name__)

# ── Member B — import with graceful fallback ──────────────────────────────────

try:
    from enhancement import enhance_frame
    _ENHANCE_OK = True
    logger.info("✓  enhancement   (Member B) loaded")
except ImportError:
    _ENHANCE_OK = False
    logger.warning("⚠  enhancement   (Member B) not found — frames pass through unenhanced")

try:
    from detection import detect_luggage
    _DETECT_OK = True
    logger.info("✓  detection     (Member B) loaded")
except ImportError:
    _DETECT_OK = False
    logger.warning("⚠  detection     (Member B) not found — no bounding boxes will be produced")

# ── Member C — import with graceful fallback ──────────────────────────────────

try:
    from tracker import BagTracker
    _TRACK_OK = True
    logger.info("✓  tracker       (Member C) loaded")
except ImportError:
    _TRACK_OK = False
    logger.warning("⚠  tracker       (Member C) not found — no persistent track IDs")

try:
    from face_recognition_module import FaceRecognizer
    _FACE_OK = True
    logger.info("✓  face_recog    (Member C) loaded")
except ImportError:
    _FACE_OK = False
    logger.warning("⚠  face_recog    (Member C) not found — face matching disabled")

try:
    from alert_logic import AlertEngine
    _ALERT_OK = True
    logger.info("✓  alert_engine  (Member C) loaded")
except ImportError:
    _ALERT_OK = False
    logger.warning("⚠  alert_engine  (Member C) not found — using built-in fallback alerts")

# Threshold (seconds) before a stationary bag triggers an alert
STATIONARY_THRESHOLD = 10.0


class CVPipeline:
    """
    Runs the four CV capabilities in sequence for each frame.

    Usage:
        pipeline = CVPipeline(face_db)
        result: FrameResult = pipeline.process_frame(frame_bgr, frame_id)
    """

    def __init__(self, face_db: FaceDatabase) -> None:
        self.face_db = face_db
        self._frame_counter = 0

        # Stateful CV modules (instantiated once, reused across frames)
        self.tracker = BagTracker(stationary_threshold=STATIONARY_THRESHOLD) if _TRACK_OK else None
        if _FACE_OK:
            self.face_matcher = FaceRecognizer()
            self.face_matcher.load_db()
        else:
            self.face_matcher = None
        self.alert_engine = AlertEngine() if _ALERT_OK else None

        logger.info("CVPipeline ready")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1  —  Image Enhancement  (Member B)
    # ─────────────────────────────────────────────────────────────────────────
    def _enhance(self, frame: np.ndarray) -> np.ndarray:
        """CLAHE on L-channel + bilateral filter to de-noise."""
        if _ENHANCE_OK:
            try:
                return enhance_frame(frame)
            except Exception as exc:
                logger.error(f"enhance_frame failed: {exc}")
        return frame  # pass-through

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2  —  Object Detection  (Member B)
    # ─────────────────────────────────────────────────────────────────────────
    def _detect(self, frame: np.ndarray) -> list:
        """
        YOLOv8 inference → raw detection dicts.
        Returns [] when module is unavailable.
        """
        if _DETECT_OK:
            try:
                return detect_luggage(frame)
            except Exception as exc:
                logger.error(f"detect_objects failed: {exc}")
        return []

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3  —  Object Tracking  (Member C)
    # ─────────────────────────────────────────────────────────────────────────
    def _track(self, detections: list, frame: np.ndarray):
        """
        BagTracker assigns persistent IDs and accumulates a stationary timer.
        Returns (List[TrackedObject], raw_tracks) so _match_faces and _check_alerts
        can pass the raw dicts directly to their respective modules.
        """
        if _TRACK_OK and self.tracker is not None:
            try:
                # BagTracker.update() expects list of ([x,y,w,h], conf, class_name) tuples.
                # detect_luggage() returns dicts with xyxy bbox, so convert here.
                tracker_input = []
                for d in detections:
                    x1, y1, x2, y2 = d["bbox"]
                    tracker_input.append(([x1, y1, x2 - x1, y2 - y1], d["conf"], d["label"]))

                raw_tracks = self.tracker.update(frame, tracker_input)
                # BagTracker output keys: "track_id", "bbox_ltrb", "stationary_seconds", "alert"
                pydantic_tracks = [
                    TrackedObject(
                        track_id=t["track_id"],
                        class_name="object",
                        confidence=0.0,
                        bbox=BoundingBox(
                            x1=t["bbox_ltrb"][0], y1=t["bbox_ltrb"][1],
                            x2=t["bbox_ltrb"][2], y2=t["bbox_ltrb"][3],
                        ),
                        stationary_seconds=t.get("stationary_seconds", 0.0),
                        is_alert=t.get("alert", False),
                    )
                    for t in raw_tracks
                ]
                return pydantic_tracks, raw_tracks
            except Exception as exc:
                logger.error(f"update_tracks failed: {exc}")

        # Fallback: treat each detection as a pseudo-track with no persistent ID.
        # detection.py keys: "bbox" [x1,y1,x2,y2], "conf", "label"
        raw_tracks = [
            {
                "track_id": -1,
                "bbox_ltrb": d["bbox"],
                "stationary_seconds": 0.0,
                "alert": False,
            }
            for d in detections
        ]
        pydantic_tracks = [
            TrackedObject(
                track_id=-1,
                class_name=d.get("label", "object"),
                confidence=d.get("conf", 0.0),
                bbox=BoundingBox(
                    x1=d["bbox"][0], y1=d["bbox"][1],
                    x2=d["bbox"][2], y2=d["bbox"][3],
                ),
                stationary_seconds=0.0,
                is_alert=False,
            )
            for d in detections
        ]
        return pydantic_tracks, raw_tracks

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4  —  Face Recognition  (Member C)
    # ─────────────────────────────────────────────────────────────────────────
    def _match_faces(self, frame: np.ndarray, raw_tracks: list):
        """
        InsightFace detects faces, then matches each embedding against the
        enrolled face DB. Returns (List[FaceMatch], raw_faces) so _check_alerts
        can forward the raw dicts to AlertEngine.
        FaceRecognizer output keys: "bbox_ltrb", "identity", "similarity"
        """
        if _FACE_OK and self.face_matcher is not None:
            try:
                raw_faces = self.face_matcher.process_frame(frame, raw_tracks)
                pydantic_faces = [
                    FaceMatch(
                        name=f.get("identity") or "Unknown",
                        confidence=f.get("similarity", 0.0),
                        bbox=BoundingBox(
                            x1=f["bbox_ltrb"][0], y1=f["bbox_ltrb"][1],
                            x2=f["bbox_ltrb"][2], y2=f["bbox_ltrb"][3],
                        ),
                        is_authorized=f.get("identity") is not None,
                    )
                    for f in raw_faces
                ]
                return pydantic_faces, raw_faces
            except Exception as exc:
                logger.error(f"face process_frame failed: {exc}")
        return [], []

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5  —  Alert Logic  (Member C, with D's fallback)
    # ─────────────────────────────────────────────────────────────────────────
    def _check_alerts(
        self,
        tracks: List[TrackedObject],
        raw_tracks: list,
        raw_faces: list,
        frame: np.ndarray,
    ) -> List[AlertEvent]:
        """
        Delegate to Member C's AlertEngine when available.
        AlertEngine.process() expects raw track/face dicts, not Pydantic models.
        Its Alert dataclass return values are converted to AlertEvent here.
        Built-in fallback: raise UNATTENDED_BAG when any track is stationary
        beyond the threshold — covers the core demo scenario.
        """
        _ALERT_TYPE_MAP = {
            "UNATTENDED_BAG": AlertType.UNATTENDED_BAG,
            "OWNER_LEFT": AlertType.OWNER_LEFT_SCENE,
            "ACCESS_VIOLATION": AlertType.UNAUTHORIZED_ACCESS,
        }
        if _ALERT_OK and self.alert_engine is not None:
            try:
                absent_alerts = (
                    self.face_matcher.get_owner_absent_alerts()
                    if _FACE_OK and self.face_matcher is not None
                    else []
                )
                raw_alerts = self.alert_engine.process(raw_tracks, raw_faces, absent_alerts)
                result: List[AlertEvent] = []
                for a in raw_alerts:
                    bbox_ltrb = getattr(a, "bbox_ltrb", [])
                    thumbnail = None
                    if len(bbox_ltrb) == 4:
                        thumbnail = _crop_b64(
                            frame,
                            BoundingBox(
                                x1=bbox_ltrb[0], y1=bbox_ltrb[1],
                                x2=bbox_ltrb[2], y2=bbox_ltrb[3],
                            ),
                        )
                    duration = getattr(a, "stationary_seconds", 0.0) or getattr(a, "absent_seconds", 0.0)
                    result.append(
                        AlertEvent(
                            alert_id=str(uuid.uuid4()),
                            alert_type=_ALERT_TYPE_MAP.get(a.alert_type, AlertType.UNATTENDED_BAG),
                            track_id=getattr(a, "track_id", None),
                            timestamp=getattr(a, "timestamp", time.time()),
                            duration_seconds=duration,
                            message=f"{a.alert_type} — track {getattr(a, 'track_id', '?')}",
                            thumbnail_b64=thumbnail,
                        )
                    )
                return result
            except Exception as exc:
                logger.error(f"alert_engine.process failed: {exc}")

        # ── Built-in fallback ──
        alerts: List[AlertEvent] = []
        for t in tracks:
            if t.stationary_seconds >= STATIONARY_THRESHOLD:
                alerts.append(
                    AlertEvent(
                        alert_id=str(uuid.uuid4()),
                        alert_type=AlertType.UNATTENDED_BAG,
                        track_id=t.track_id,
                        timestamp=time.time(),
                        duration_seconds=t.stationary_seconds,
                        message=(
                            f"{t.class_name.capitalize()} (track {t.track_id}) "
                            f"stationary for {t.stationary_seconds:.0f}s"
                        ),
                        thumbnail_b64=_crop_b64(frame, t.bbox),
                    )
                )
        return alerts

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────
    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: Optional[int] = None,
    ) -> FrameResult:
        """
        Full pipeline: raw frame → enhance → detect → track → faces → alerts.
        Returns a FrameResult ready to be JSON-serialised over the WebSocket.

        Called from both the /ws WebSocket handler and the /upload processor.
        Runs synchronously (blocking) — caller is responsible for offloading to
        an executor so the async event loop isn't blocked.
        """
        if frame_id is None:
            self._frame_counter += 1
            frame_id = self._frame_counter

        enhanced           = self._enhance(frame)
        detections         = self._detect(enhanced)
        tracks, raw_tracks = self._track(detections, enhanced)
        faces, raw_faces   = self._match_faces(enhanced, raw_tracks)
        alerts             = self._check_alerts(tracks, raw_tracks, raw_faces, enhanced)

        # Send an enhanced preview frame every 5 frames to reduce bandwidth
        annotated = enhanced.copy()
        for t in tracks:
            x1, y1, x2, y2 = int(t.bbox.x1), int(t.bbox.y1), int(t.bbox.x2), int(t.bbox.y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"luggage {t.confidence:.2f}" if t.confidence > 0 else "luggage"
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for f in faces:
            x1, y1, x2, y2 = int(f.bbox.x1), int(f.bbox.y1), int(f.bbox.x2), int(f.bbox.y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f.name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        preview_b64 = _frame_to_b64(annotated)
        

        return FrameResult(
            frame_id=frame_id,
            timestamp=time.time(),
            tracks=tracks,
            faces=faces,
            alerts=alerts,
            enhanced_frame_b64=preview_b64,
        )

    def sync_face_db(self) -> None:
        """
        Reload FaceRecognizer's in-memory enrolled DB from disk.
        Call this after face_db.enrol() so the recognizer picks up new faces
        immediately without requiring a server restart.
        """
        if _FACE_OK and self.face_matcher is not None:
            self.face_matcher.load_db()

    def get_embedding_for_enrol(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a single ArcFace embedding from a reference enrolment image.
        Returns None if no face is detected or if the module is unavailable.
        Called from the /enrol endpoint.
        """
        if _FACE_OK and self.face_matcher is not None:
            try:
                # FaceRecognizer exposes the InsightFace app via ._app
                faces = self.face_matcher._app.get(image)
                if faces:
                    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    return face.normed_embedding
            except Exception as exc:
                logger.error(f"extract_embedding failed: {exc}")
        logger.warning("Face recognition unavailable — cannot extract embedding for enrolment")
        return None


# ── Module-level helpers (also used by main.py) ───────────────────────────────

def _crop_b64(frame: np.ndarray, bbox: BoundingBox) -> Optional[str]:
    """Crop a bounding box from a frame and return as base64-encoded JPEG."""
    try:
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(w, int(bbox.x2))
        y2 = min(h, int(bbox.y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode()
    except Exception:
        return None


def _frame_to_b64(frame: np.ndarray, quality: int = 70) -> str:
    """Encode a full frame as a base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()
