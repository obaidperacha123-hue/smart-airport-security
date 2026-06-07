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

Integration contract (what D expects from B & C)
─────────────────────────────────────────────────
Member B:
  enhancement.py      → enhance_frame(frame: np.ndarray) -> np.ndarray
  detection.py        → detect_objects(frame: np.ndarray) -> List[dict]
                          dict keys: "class_name", "confidence", "bbox" [x1,y1,x2,y2]
  morphology.py       → clean_mask(mask: np.ndarray) -> np.ndarray   (optional)

Member C:
  tracker.py          → class DeepSORTTracker
                        update_tracks(tracker, detections, frame) -> List[dict]
                          dict keys: "id", "class_name", "confidence",
                                     "bbox" [x1,y1,x2,y2], "stationary_seconds"
  face_recognition_module.py
                      → class InsightFaceMatcher
                        .detect_and_match(frame, face_db) -> List[dict]
                          dict keys: "name", "confidence", "bbox", "is_authorized"
                        .extract_embedding(image) -> np.ndarray | None
  alert_logic.py      → class AlertEngine(stationary_threshold_seconds)
                        .evaluate(tracks, faces, frame) -> List[AlertEvent]
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
    from detection import detect_objects
    _DETECT_OK = True
    logger.info("✓  detection     (Member B) loaded")
except ImportError:
    _DETECT_OK = False
    logger.warning("⚠  detection     (Member B) not found — no bounding boxes will be produced")

try:
    from morphology import clean_mask
    _MORPH_OK = True
    logger.info("✓  morphology    (Member B) loaded")
except ImportError:
    _MORPH_OK = False
    # Morphology is optional (5th capability) — no warning needed

# ── Member C — import with graceful fallback ──────────────────────────────────

try:
    from tracker import DeepSORTTracker, update_tracks
    _TRACK_OK = True
    logger.info("✓  tracker       (Member C) loaded")
except ImportError:
    _TRACK_OK = False
    logger.warning("⚠  tracker       (Member C) not found — no persistent track IDs")

try:
    from face_recognition_module import InsightFaceMatcher
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
STATIONARY_THRESHOLD = 30.0


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
        self.tracker = DeepSORTTracker() if _TRACK_OK else None
        self.face_matcher = InsightFaceMatcher() if _FACE_OK else None
        self.alert_engine = AlertEngine(
            stationary_threshold_seconds=STATIONARY_THRESHOLD
        ) if _ALERT_OK else None

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
                return detect_objects(frame)
            except Exception as exc:
                logger.error(f"detect_objects failed: {exc}")
        return []

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3  —  Object Tracking  (Member C)
    # ─────────────────────────────────────────────────────────────────────────
    def _track(self, detections: list, frame: np.ndarray) -> List[TrackedObject]:
        """
        DeepSORT assigns persistent track IDs and accumulates a stationary timer.
        Converts raw dicts → TrackedObject Pydantic models.
        """
        if _TRACK_OK and self.tracker is not None:
            try:
                raw = update_tracks(self.tracker, detections, frame)
                return [
                    TrackedObject(
                        track_id=t["id"],
                        class_name=t.get("class_name", "object"),
                        confidence=t.get("confidence", 0.0),
                        bbox=BoundingBox(
                            x1=t["bbox"][0], y1=t["bbox"][1],
                            x2=t["bbox"][2], y2=t["bbox"][3],
                        ),
                        stationary_seconds=t.get("stationary_seconds", 0.0),
                        is_alert=t.get("stationary_seconds", 0.0) >= STATIONARY_THRESHOLD,
                    )
                    for t in raw
                ]
            except Exception as exc:
                logger.error(f"update_tracks failed: {exc}")

        # Fallback: treat each raw detection as a pseudo-track (no persistent ID)
        return [
            TrackedObject(
                track_id=-1,
                class_name=d.get("class_name", "object"),
                confidence=d.get("confidence", 0.0),
                bbox=BoundingBox(
                    x1=d["bbox"][0], y1=d["bbox"][1],
                    x2=d["bbox"][2], y2=d["bbox"][3],
                ),
                stationary_seconds=0.0,
                is_alert=False,
            )
            for d in detections
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4  —  Face Recognition  (Member C)
    # ─────────────────────────────────────────────────────────────────────────
    def _match_faces(self, frame: np.ndarray) -> List[FaceMatch]:
        """
        InsightFace detects faces, then matches each embedding against the
        enrolled face DB.  Returns [] when module is unavailable.
        """
        if _FACE_OK and self.face_matcher is not None:
            try:
                raw = self.face_matcher.detect_and_match(frame, self.face_db)
                return [
                    FaceMatch(
                        name=f.get("name", "Unknown"),
                        confidence=f.get("confidence", 0.0),
                        bbox=BoundingBox(
                            x1=f["bbox"][0], y1=f["bbox"][1],
                            x2=f["bbox"][2], y2=f["bbox"][3],
                        ),
                        is_authorized=f.get("is_authorized", False),
                    )
                    for f in raw
                ]
            except Exception as exc:
                logger.error(f"face detect_and_match failed: {exc}")
        return []

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5  —  Alert Logic  (Member C, with D's fallback)
    # ─────────────────────────────────────────────────────────────────────────
    def _check_alerts(
        self,
        tracks: List[TrackedObject],
        faces: List[FaceMatch],
        frame: np.ndarray,
    ) -> List[AlertEvent]:
        """
        Delegate to Member C's AlertEngine when available.
        Built-in fallback: raise UNATTENDED_BAG when any track is stationary
        beyond the threshold — covers the core demo scenario.
        """
        if _ALERT_OK and self.alert_engine is not None:
            try:
                return self.alert_engine.evaluate(tracks, faces, frame)
            except Exception as exc:
                logger.error(f"alert_engine.evaluate failed: {exc}")

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

        enhanced  = self._enhance(frame)
        detections = self._detect(enhanced)
        tracks    = self._track(detections, enhanced)
        faces     = self._match_faces(enhanced)
        alerts    = self._check_alerts(tracks, faces, enhanced)

        # Send an enhanced preview frame every 5 frames to reduce bandwidth
        preview_b64 = _frame_to_b64(enhanced) if (frame_id % 5 == 0) else None

        return FrameResult(
            frame_id=frame_id,
            timestamp=time.time(),
            tracks=tracks,
            faces=faces,
            alerts=alerts,
            enhanced_frame_b64=preview_b64,
        )

    def get_embedding_for_enrol(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a single ArcFace embedding from a reference enrolment image.
        Returns None if no face is detected or if the module is unavailable.
        Called from the /enrol endpoint.
        """
        if _FACE_OK and self.face_matcher is not None:
            try:
                return self.face_matcher.extract_embedding(image)
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
