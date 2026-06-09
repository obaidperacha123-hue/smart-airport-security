"""
face_recognizer.py  –  Member C
Face Recognition with InsightFace (ArcFace embeddings)

CV task implemented in this file
----------------------------------
  3. Face detection    – InsightFace buffalo_sc ONNX model locates face regions
  4. Face recognition  – ArcFace 512-d embeddings + cosine similarity matching
                         against an enrolled personnel database

Responsibilities
----------------
  - Detect all faces in each frame using InsightFace
  - Match each detected face against the enrolled database via average
    cosine similarity across all reference embeddings for each person
  - Associate known faces with nearby bag track IDs (proximity-based)
  - Fire "owner left scene" alerts when a bag's known owner has not
    appeared in any frame for longer than OWNER_ABSENT_THRESHOLD seconds

Performance note
----------------
  InsightFace buffalo_sc on CPU typically achieves 8–15 FPS at 640×640
  depending on hardware.  To maintain >= 10 FPS at the system level,
  face recognition can be run every N frames (face_skip_frames parameter)
  while tracking runs every frame.  The last known face results are cached
  and reused on skipped frames.

Dependencies
------------
  pip install insightface onnxruntime numpy
  InsightFace auto-downloads buffalo_sc (~85 MB) on first run into
  ~/.insightface/models/
"""

import time
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

from insightface.app import FaceAnalysis


# ── Config ────────────────────────────────────────────────────────────────────
FACE_DB_PATH           = Path("data/face_db.pkl")
RECOGNITION_THRESHOLD  = 0.50   # cosine similarity cutoff for a valid match.
                                 # Raised from 0.45 → 0.50 to reduce false
                                 # positives in busy multi-person scenes.
                                 # ArcFace literature recommends 0.50–0.60
                                 # for 1:N identification tasks.
OWNER_ABSENT_THRESHOLD = 20.0   # seconds before OWNER_LEFT alert fires
ASSOCIATION_RADIUS_PX  = 300    # max pixel distance (face centre → bag centre)
                                 # to link a recognised face with a bag track
# ─────────────────────────────────────────────────────────────────────────────


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two embedding vectors (L2-safe)."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def _box_centre(ltrb: list) -> tuple:
    """Return (cx, cy) of an [x1, y1, x2, y2] bounding box."""
    return ((ltrb[0] + ltrb[2]) / 2, (ltrb[1] + ltrb[3]) / 2)


def _pixel_dist(p1: tuple, p2: tuple) -> float:
    """Euclidean pixel distance between two (x, y) points."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


# ─────────────────────────────────────────────────────────────────────────────

class FaceRecognizer:
    """
    Frame-by-frame face detection, recognition, and owner-tracking.

    Typical call sequence
    ---------------------
    fr = FaceRecognizer()
    fr.load_db()

    # once per frame:
    face_results  = fr.process_frame(frame, bag_tracks)
    absent_alerts = fr.get_owner_absent_alerts()

    face_result dict fields:
    {
      "bbox_ltrb"  : [x1, y1, x2, y2],
      "identity"   : str | None,     # matched name, or None = unknown
      "similarity" : float,          # best cosine similarity (0–1)
      "embedding"  : np.ndarray,     # 512-d ArcFace vector
    }

    absent_alert dict fields:
    {
      "track_id"      : int,
      "owner"         : str,
      "absent_seconds": float,
    }
    """

    def __init__(self,
                 db_path:                Path  = FACE_DB_PATH,
                 recognition_threshold:  float = RECOGNITION_THRESHOLD,
                 owner_absent_threshold: float = OWNER_ABSENT_THRESHOLD,
                 association_radius_px:  float = ASSOCIATION_RADIUS_PX,
                 face_skip_frames:       int   = 1):
        """
        Parameters
        ----------
        face_skip_frames : run face recognition every N frames to maintain
                           >= 10 FPS on slower hardware.  Set to 1 (default)
                           to process every frame.  Set to 3 for ~3x speedup.
                           Tracking still runs every frame regardless.
        """
        self.db_path                = db_path
        self.recognition_threshold  = recognition_threshold
        self.owner_absent_threshold = owner_absent_threshold
        self.association_radius_px  = association_radius_px
        self.face_skip_frames       = max(1, int(face_skip_frames))

        # InsightFace — buffalo_sc is the lightweight CPU-compatible ONNX bundle.
        # Swap providers list to ["CUDAExecutionProvider"] if a GPU is available.
        self._app = FaceAnalysis(
            name      = "buffalo_sc",
            providers = ["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=(640, 640))

        # name -> list[np.ndarray]  (one or more 512-d reference embeddings per person)
        self._enrolled: dict[str, list] = {}

        # Dirty flag: avoids redundant disk writes during bulk enrolment.
        # Set True by enrol(), cleared by save_db().
        self._dirty: bool = False

        # bag track_id -> {"owner": str | None, "last_seen": float | None}
        self._ownership: dict = defaultdict(lambda: {
            "owner"    : None,
            "last_seen": None,
        })

        # Frame-skip state
        self._frame_count:    int  = 0
        self._cached_results: list = []   # last face_results from a processed frame

    # ── Database management ───────────────────────────────────────────────────

    def load_db(self) -> None:
        """Load the enrolled face database from disk (safe when no file exists)."""
        if self.db_path.exists():
            with open(self.db_path, "rb") as f:
                self._enrolled = pickle.load(f)
            print(f"[FaceRecognizer] Loaded {len(self._enrolled)} identities "
                  f"from {self.db_path}")
        else:
            print(f"[FaceRecognizer] No database at {self.db_path}. "
                  "Starting with empty database.")
        self._dirty = False

    def save_db(self) -> None:
        """Persist the enrolled database to disk."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(self._enrolled, f)
        self._dirty = False
        print(f"[FaceRecognizer] Saved {len(self._enrolled)} identities "
              f"to {self.db_path}")

    def enrol(self, name: str, image_bgr: np.ndarray,
              auto_save: bool = True) -> bool:
        """
        Add a reference face from a BGR image.

        Parameters
        ----------
        name      : display name or employee ID for this person
        image_bgr : BGR frame or photo containing the face
        auto_save : True (default) saves the DB after each enrolment.
                    Pass False when enrolling many images in a loop and
                    call save_db() once at the end to avoid N disk writes.

        Returns True if a face was found and enrolled, False otherwise.
        """
        faces = self._app.get(image_bgr)
        if not faces:
            print(f"[FaceRecognizer] enrol('{name}'): no face detected.")
            return False

        # If multiple faces in the image, use the largest (closest to camera)
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        self._enrolled.setdefault(name, []).append(face.normed_embedding)
        self._dirty = True
        print(f"[FaceRecognizer] Enrolled '{name}' "
              f"({len(self._enrolled[name])} embedding(s) total).")

        if auto_save:
            self.save_db()
        return True

    # ── Per-frame processing ──────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, bag_tracks: list) -> list:
        """
        Detect and recognise faces, then update bag ownership state.

        Respects face_skip_frames: if this frame is a skip frame, returns
        the cached results from the last processed frame without running
        inference.  Ownership timestamps are still updated on skip frames
        so the "absent" timer does not fire spuriously.

        Parameters
        ----------
        frame      : enhanced BGR frame (output of Member B's pipeline)
        bag_tracks : list of track dicts from BagTracker.update()

        Returns
        -------
        List of face result dicts (one per detected face).
        """
        self._frame_count += 1
        now = time.time()

        # ── Frame-skip logic ──────────────────────────────────────────────
        if self._frame_count % self.face_skip_frames != 0:
            # Reuse cached results; still refresh last_seen timestamps
            # for any identity that was visible in the last processed frame
            for fr in self._cached_results:
                if fr["identity"] is not None:
                    for tid, rec in self._ownership.items():
                        if rec["owner"] == fr["identity"]:
                            rec["last_seen"] = now
            return self._cached_results

        # ── Run face detection + recognition ──────────────────────────────
        faces        = self._app.get(frame)
        face_results = []

        for face in faces:
            bbox = list(map(int, face.bbox))    # float → int pixel coords
            emb  = face.normed_embedding
            identity, similarity = self._match(emb)

            face_results.append({
                "bbox_ltrb" : bbox,
                "identity"  : identity,
                "similarity": round(similarity, 3),
                "embedding" : emb,
            })

            # Only associate KNOWN faces with bag tracks
            if identity is None:
                continue

            # Find the nearest bag within ASSOCIATION_RADIUS_PX
            face_centre         = _box_centre(bbox)
            best_tid, best_dist = None, float("inf")

            for track in bag_tracks:
                dist = _pixel_dist(face_centre, _box_centre(track["bbox_ltrb"]))
                if dist < best_dist and dist < self.association_radius_px:
                    best_dist = dist
                    best_tid  = track["track_id"]

            if best_tid is not None:
                rec = self._ownership[best_tid]
                if rec["owner"] is None or rec["owner"] == identity:
                    rec["owner"]     = identity
                    rec["last_seen"] = now

            # Refresh last_seen for ALL bags this person owns
            # (handles walking slightly away from the bag but still visible)
            for tid, rec in self._ownership.items():
                if rec["owner"] == identity:
                    rec["last_seen"] = now

        self._cached_results = face_results
        return face_results

    def get_owner_absent_alerts(self) -> list:
        """
        Return track IDs whose known owner has been absent for longer than
        owner_absent_threshold seconds.

        Only fires for tracks that have a confirmed owner (i.e., the system
        previously saw that person standing near the bag).

        Returns
        -------
        List of dicts:
          [{"track_id": int, "owner": str, "absent_seconds": float}, ...]
        """
        now    = time.time()
        alerts = []
        for tid, rec in self._ownership.items():
            if rec["owner"] is None or rec["last_seen"] is None:
                continue
            absent_secs = now - rec["last_seen"]
            if absent_secs >= self.owner_absent_threshold:
                alerts.append({
                    "track_id"      : tid,
                    "owner"         : rec["owner"],
                    "absent_seconds": round(absent_secs, 1),
                })
        return alerts

    def reset_ownership(self, track_id: int) -> None:
        """
        Clear the ownership record for one track.
        Call this after acknowledging an OWNER_LEFT alert so the owner
        can be re-associated if they return to the scene.
        """
        if track_id in self._ownership:
            del self._ownership[track_id]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _match(self, embedding: np.ndarray) -> tuple:
        """
        Find the best matching enrolled identity.

        Computes the average cosine similarity across all reference embeddings
        for each enrolled person and returns the best match if it exceeds
        RECOGNITION_THRESHOLD.

        Returns (name, similarity) or (None, best_sim) if below threshold.
        """
        best_name, best_sim = None, -1.0
        for name, refs in self._enrolled.items():
            sims = [_cosine_sim(embedding, r) for r in refs]
            avg  = float(np.mean(sims))
            if avg > best_sim:
                best_sim  = avg
                best_name = name

        if best_sim >= self.recognition_threshold:
            return best_name, best_sim
        return None, best_sim
