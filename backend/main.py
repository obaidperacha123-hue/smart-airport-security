"""
main.py — FastAPI backend for Smart Airport Security System.
Member D owns this file.

Endpoints
─────────────────────────────────────────────────────────────────────────────
  GET  /health          Liveness probe — component readiness check
  GET  /ws              WebSocket — live browser camera frame stream
  POST /upload          Batch-process an uploaded image or video file
  POST /enrol           Enrol a new person into the face DB (multipart form)
  GET  /enrol           List all enrolled persons (admin utility)
  DELETE /enrol/{name}  Remove a person from the face DB

WebSocket protocol
─────────────────────────────────────────────────────────────────────────────
  Client → Server (JSON):
    { "frame_b64": "<JPEG base64 string>" }

  Server → Client (JSON, FrameResult schema):
    {
      "frame_id": 42,
      "timestamp": 1718000000.123,
      "tracks":  [ { "track_id": 3, "class_name": "backpack", "bbox": {...},
                     "stationary_seconds": 12.5, "is_alert": false } ],
      "faces":   [ { "name": "alice", "confidence": 0.82, "bbox": {...},
                     "is_authorized": true } ],
      "alerts":  [ { "alert_id": "uuid", "alert_type": "unattended_bag", ... } ],
      "enhanced_frame_b64": "<JPEG base64 | null>"
    }
"""

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware

from face_db import FaceDatabase
from pipeline import CVPipeline
from schemas import (
    AlertEvent,
    EnrolResponse,
    FrameResult,
    HealthResponse,
    UploadResponse,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Airport Security API",
    description=(
        "Computer Vision pipeline: CLAHE enhancement → YOLOv8 detection → "
        "DeepSORT tracking → InsightFace recognition → alert engine"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared application state ──────────────────────────────────────────────────
# Initialised in startup_event — never None after that.
face_db: Optional[FaceDatabase] = None
pipeline: Optional[CVPipeline] = None

# Track open WebSocket sessions so /health can report connection count.
active_connections: Dict[str, WebSocket] = {}


# ── Lifecycle hooks ───────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    global face_db, pipeline
    logger.info("═══ Smart Airport Security backend starting up ═══")
    face_db = FaceDatabase()
    pipeline = CVPipeline(face_db)
    logger.info(
        f"Face DB: {face_db.count} person(s) enrolled | "
        f"Pipeline modules: ready"
    )
    logger.info("Backend ready — listening for connections")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("Shutting down — persisting face DB…")
    if face_db:
        face_db.save()


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Liveness / readiness probe",
)
async def health_check() -> HealthResponse:
    """
    Returns server status and the readiness of each component.
    Member E's frontend polls this to show a 'pipeline ready' indicator.
    """
    return HealthResponse(
        status="ok",
        pipeline_ready=pipeline is not None,
        face_db_count=face_db.count if face_db else 0,
        active_ws_connections=len(active_connections),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /ws  — WebSocket live feed
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_live_feed(websocket: WebSocket) -> None:
    """
    Live camera streaming endpoint.

    Member E's React frontend captures camera frames via getUserMedia,
    encodes each frame as a base64 JPEG string, and sends it here.
    This handler decodes the frame, runs the full CV pipeline in a
    thread-pool executor (non-blocking), and streams FrameResult JSON back.

    The pipeline runs at whatever rate the client sends frames.
    For a 30 fps source, consider throttling on the frontend to 10–15 fps.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    active_connections[session_id] = websocket
    logger.info(
        f"WS connected  session={session_id}  "
        f"total={len(active_connections)}"
    )

    loop = asyncio.get_event_loop()
    frame_id = 0

    try:
        while True:
            # ── Receive ──────────────────────────────────────────────────────
            raw_message = await websocket.receive_text()

            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Payload must be valid JSON"})
                continue

            frame_b64: Optional[str] = payload.get("frame_b64")
            if not frame_b64:
                await websocket.send_json({"error": "Missing required field: frame_b64"})
                continue

            # ── Decode base64 → BGR numpy frame ───────────────────────────────
            try:
                img_bytes = base64.b64decode(frame_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("cv2.imdecode returned None — corrupt frame?")
            except Exception as exc:
                await websocket.send_json({"error": f"Frame decode error: {exc}"})
                continue

            # ── Run CV pipeline in executor (keeps event loop free) ────────────
            frame_id += 1
            try:
                result: FrameResult = await loop.run_in_executor(
                    None,
                    pipeline.process_frame,
                    frame,
                    frame_id,
                )
                await websocket.send_text(result.model_dump_json())
            except Exception as exc:
                logger.error(
                    f"Pipeline error  session={session_id}  frame={frame_id}: {exc}",
                    exc_info=True,
                )
                await websocket.send_json({"error": f"Pipeline error: {exc}"})

    except WebSocketDisconnect:
        logger.info(f"WS disconnected  session={session_id}")
    except Exception as exc:
        logger.error(f"WS error  session={session_id}: {exc}", exc_info=True)
    finally:
        active_connections.pop(session_id, None)
        logger.info(
            f"WS cleaned up  session={session_id}  "
            f"remaining={len(active_connections)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# POST /upload  — batch-process a video or image file
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".mkv"}
# Process ~5 fps of a video regardless of source frame rate to keep response times reasonable
VIDEO_PROCESS_FPS = 5


@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Processing"],
    summary="Batch-process an image or video file",
)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a CCTV image (.jpg / .png) or video (.mp4 / .avi / .mov) for
    full CV pipeline processing.

    The server:
      1. Reads every Nth frame (videos are downsampled to ~5 fps)
      2. Runs enhancement → detection → tracking → face recognition → alerts
      3. Returns all generated alert events plus an aggregated summary

    Used by Member E's 'Upload view' drag-and-drop panel.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not yet initialised")

    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Accepted: {ALLOWED_EXTENSIONS}",
        )

    job_id = str(uuid.uuid4())
    start_ts = time.time()
    content = await file.read()
    logger.info(
        f"Upload received  job={job_id}  file={filename}  "
        f"size={len(content) / 1024:.1f} KB"
    )

    # ── Image ─────────────────────────────────────────────────────────────────
    if suffix in {".jpg", ".jpeg", ".png"}:
        result, total = await _process_image_bytes(content, job_id)
        return UploadResponse(
            job_id=job_id,
            filename=filename,
            total_frames=total,
            processed_frames=total,
            duration_seconds=round(time.time() - start_ts, 3),
            alerts=result.alerts,
            summary=_build_summary([result]),
        )

    # ── Video ─────────────────────────────────────────────────────────────────
    results, total_frames, processed_frames = await _process_video_bytes(
        content, suffix, job_id
    )
    all_alerts: List[AlertEvent] = [a for r in results for a in r.alerts]

    return UploadResponse(
        job_id=job_id,
        filename=filename,
        total_frames=total_frames,
        processed_frames=processed_frames,
        duration_seconds=round(time.time() - start_ts, 3),
        alerts=all_alerts,
        summary=_build_summary(results),
    )


async def _process_image_bytes(
    content: bytes,
    job_id: str,
) -> tuple[FrameResult, int]:
    """Decode image bytes and run the pipeline once."""
    nparr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode image file")

    loop = asyncio.get_event_loop()
    result: FrameResult = await loop.run_in_executor(
        None, pipeline.process_frame, frame, 1
    )
    logger.info(f"Image processed  job={job_id}  alerts={len(result.alerts)}")
    return result, 1


async def _process_video_bytes(
    content: bytes,
    suffix: str,
    job_id: str,
) -> tuple[List[FrameResult], int, int]:
    """
    Write the video to a temp file, iterate frames with OpenCV VideoCapture,
    run the pipeline on every Nth frame (targeting VIDEO_PROCESS_FPS).
    Yields to the event loop every 10 processed frames so the server stays
    responsive to other requests during long video processing.
    """
    # Write to a named temp file — OpenCV VideoCapture needs a real filesystem path
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(tmp_fd, content)
        os.close(tmp_fd)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=422, detail="Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_skip = max(1, int(round(source_fps / VIDEO_PROCESS_FPS)))

        logger.info(
            f"Video  job={job_id}  total={total_frames}  "
            f"source_fps={source_fps:.1f}  frame_skip={frame_skip}"
        )

        loop = asyncio.get_event_loop()
        results: List[FrameResult] = []
        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            result: FrameResult = await loop.run_in_executor(
                None, pipeline.process_frame, frame, frame_idx
            )
            results.append(result)
            processed += 1

            # Yield every 10 processed frames so the event loop breathes
            if processed % 10 == 0:
                await asyncio.sleep(0)

        cap.release()
        logger.info(
            f"Video done  job={job_id}  processed={processed}  "
            f"alerts={sum(len(r.alerts) for r in results)}"
        )
        return results, total_frames, processed

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _build_summary(results: List[FrameResult]) -> dict:
    """Aggregate stats across all processed frames for the UploadResponse."""
    unique_track_ids: set = set()
    unique_persons: set = set()
    alert_counts: dict = {}

    for r in results:
        for t in r.tracks:
            if t.track_id != -1:
                unique_track_ids.add(t.track_id)
        for f in r.faces:
            if f.name.lower() not in ("unknown", ""):
                unique_persons.add(f.name)
        for a in r.alerts:
            alert_counts[a.alert_type] = alert_counts.get(a.alert_type, 0) + 1

    return {
        "unique_tracked_objects": len(unique_track_ids),
        "unique_persons_identified": len(unique_persons),
        "alert_breakdown": alert_counts,
        "frames_with_alerts": sum(1 for r in results if r.alerts),
        "total_frames_processed": len(results),
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /enrol  — enrol a person into the face database
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/enrol",
    response_model=EnrolResponse,
    tags=["Administration"],
    summary="Enrol an authorised person into the face DB",
)
async def enrol_person(
    name: str = Form(..., description="Full name of the person (e.g. 'Alice Mahmoud')"),
    images: List[UploadFile] = File(
        ...,
        description="One or more reference face images (.jpg / .png). "
                    "More images → better matching under CCTV conditions.",
    ),
) -> EnrolResponse:
    """
    Registers airport staff / authorised personnel in the face recognition DB.

    - Accepts 1–N reference images per person.
    - InsightFace extracts a 512-dim ArcFace embedding per image.
    - All embeddings are stored under the same name in the pickle DB.
    - The DB is written to disk after every successful enrolment.

    Used by Member E's 'Admin / Enrolment Panel'.
    """
    if pipeline is None or face_db is None:
        raise HTTPException(status_code=503, detail="Pipeline not yet initialised")

    name = name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="'name' field must not be empty")

    loop = asyncio.get_event_loop()
    enrolled_count = 0
    warnings: List[str] = []

    for img_upload in images:
        ext = Path(img_upload.filename or "").suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png"}:
            warnings.append(
                f"{img_upload.filename}: skipped — unsupported format (use .jpg/.png)"
            )
            continue

        img_bytes = await img_upload.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        ref_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if ref_image is None:
            warnings.append(f"{img_upload.filename}: could not decode image")
            continue

        # Extract embedding in executor — InsightFace inference is CPU-heavy
        embedding = await loop.run_in_executor(
            None, pipeline.get_embedding_for_enrol, ref_image
        )

        if embedding is None:
            warnings.append(
                f"{img_upload.filename}: no face detected — "
                "ensure the image shows a clear, front-facing face"
            )
            continue

        face_db.enrol(name.lower(), embedding)
        enrolled_count += 1

    if enrolled_count == 0:
        raise HTTPException(
            status_code=422,
            detail=f"No faces could be enrolled. Details: {warnings}",
        )

    total_embs = len(face_db.embeddings.get(name.lower(), []))
    message = f"Enrolled {enrolled_count} image(s) for '{name}'."
    if warnings:
        message += f" Warnings: {'; '.join(warnings)}"

    logger.info(
        f"Enrolment complete: '{name}'  "
        f"new={enrolled_count}  total_embeddings={total_embs}"
    )

    return EnrolResponse(
        success=True,
        name=name,
        embeddings_count=total_embs,
        message=message,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /enrol  — list enrolled persons
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/enrol",
    tags=["Administration"],
    summary="List all enrolled persons",
)
async def list_enrolled() -> dict:
    """
    Returns the list of enrolled persons and their embedding counts.
    Used by Member E's admin panel to display who is currently in the DB.
    """
    if face_db is None:
        raise HTTPException(status_code=503, detail="Face DB not initialised")
    return {
        "persons": face_db.list_persons(),
        "total_persons": face_db.count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /enrol/{name}  — remove a person
# ─────────────────────────────────────────────────────────────────────────────

@app.delete(
    "/enrol/{name}",
    tags=["Administration"],
    summary="Remove a person from the face DB",
)
async def remove_person(name: str) -> dict:
    """Delete all embeddings for the given person from the face database."""
    if face_db is None:
        raise HTTPException(status_code=503, detail="Face DB not initialised")
    removed = face_db.remove(name.lower())
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' not found in face DB",
        )
    return {"success": True, "message": f"'{name}' removed from face DB"}


# ─────────────────────────────────────────────────────────────────────────────
# Dev entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,          # hot-reload during development
        log_level="info",
    )
