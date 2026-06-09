"""
face_recognizer.py  –  Member C
Face Recognition with InsightFace (ArcFace embeddings)

Responsibilities:
  - Detect faces in every frame
  - Compare each face to the enrolled personnel database
  - Track which person (by face ID) is associated with which bag track ID
  - Raise "owner left scene" alerts when the owner of a stationary bag
    has not been seen for longer than OWNER_ABSENT_THRESHOLD seconds

Dependencies:
  pip install insightface onnxruntime numpy
"""

import time
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

import insightface
from insightface.app import FaceAnalysis

# ── Config ───────────────────────────────────────────────────────────────────
FACE_DB_PATH              = Path("data/face_db.pkl")
RECOGNITION_THRESHOLD     = 0.45    # cosine similarity; below = unknown
OWNER_ABSENT_THRESHOLD    = 20.0    # seconds owner can be absent before alert
ASSOCIATION_RADIUS_PX     = 300     # max pixel dist (face→bag centre) to associate
# ─────────────────────────────────────────────────────────────────────────────


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def _box_centre(ltrb: list) -> tuple:
    return ((ltrb[0] + ltrb[2]) / 2, (ltrb[1] + ltrb[3]) / 2)


def _pixel_dist(p1: tuple, p2: tuple) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class FaceRecognizer:
    """
    Detects and identifies faces, then drives the owner-left-scene logic.

    Typical call sequence
    ---------------------
    fr = FaceRecognizer()
    fr.load_db()                        # load enrolled faces

    # per frame:
    face_results = fr.process_frame(frame, bag_tracks)

    # Each face_result dict:
    # {
    #   "bbox_ltrb"  : [x1,y1,x2,y2],
    #   "identity"   : str | None,      # matched name or None
    #   "similarity" : float,
    #   "embedding"  : np.ndarray,      # 512-d ArcFace vector
    # }

    # Owner-left-scene alerts (list of track_ids with absent owners):
    absent_alerts = fr.get_owner_absent_alerts()
    """

    def __init__(self,
                 db_path: Path                   = FACE_DB_PATH,
                 recognition_threshold: float    = RECOGNITION_THRESHOLD,
                 owner_absent_threshold: float   = OWNER_ABSENT_THRESHOLD,
                 association_radius_px: float    = ASSOCIATION_RADIUS_PX):

        self.db_path                 = db_path
        self.recognition_threshold   = recognition_threshold
        self.owner_absent_threshold  = owner_absent_threshold
        self.association_radius_px   = association_radius_px

        # InsightFace – buffalo_sc is the lightweight ONNX model bundle
        self._app = FaceAnalysis(
            name="buffalo_sc",
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=(640, 640))

        # name -> list[np.ndarray]  (multiple reference embeddings per person)
        self._enrolled: dict[str, list] = {}

        # bag track_id -> {"owner": str|None, "last_seen": float|None}
        self._ownership: dict = defaultdict(lambda: {
            "owner": None,
            "last_seen": None,
        })

    # ── database management ─────────────────────────────────────────────────

    def load_db(self) -> None:
        """Load the face database from disk (if it exists)."""
        if self.db_path.exists():
            with open(self.db_path, "rb") as f:
                self._enrolled = pickle.load(f)
            print(f"[FaceRecognizer] Loaded {len(self._enrolled)} identities "
                  f"from {self.db_path}")
        else:
            print(f"[FaceRecognizer] No database found at {self.db_path}. "
                  "Starting empty.")

    def save_db(self) -> None:
        """Persist the database to disk."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(self._enrolled, f)
        print(f"[FaceRecognizer] Saved {len(self._enrolled)} identities.")

    def enrol(self, name: str, image_bgr: np.ndarray) -> bool:
        """
        Add a reference face from a BGR image.
        Returns True if a face was detected and enrolled.
        """
        faces = self._app.get(image_bgr)
        if not faces:
            return False
        # Use the largest detected face
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        emb  = face.normed_embedding
        self._enrolled.setdefault(name, []).append(emb)
        self.save_db()
        return True

    # ── per-frame processing ────────────────────────────────────────────────

    def process_frame(self,
                      frame: np.ndarray,
                      bag_tracks: list) -> list:
        """
        Parameters
        ----------
        frame      : enhanced BGR frame
        bag_tracks : list of track dicts from BagTracker.update()

        Returns
        -------
        List of face result dicts (one per detected face).
        Also updates internal ownership state.
        """
        faces       = self._app.get(frame)
        now         = time.time()
        face_results = []

        for face in faces:
            bbox = list(map(int, face.bbox))      # [x1,y1,x2,y2]
            emb  = face.normed_embedding
            identity, similarity = self._match(emb)

            face_results.append({
                "bbox_ltrb"  : bbox,
                "identity"   : identity,
                "similarity" : round(similarity, 3),
                "embedding"  : emb,
            })

            # ── associate face with the nearest bag ────────────────────────
            if identity is None:
                continue
            face_centre = _box_centre(bbox)
            best_tid, best_dist = None, float("inf")
            for track in bag_tracks:
                dist = _pixel_dist(face_centre, _box_centre(track["bbox_ltrb"]))
                if dist < best_dist and dist < self.association_radius_px:
                    best_dist, best_tid = dist, track["track_id"]

            if best_tid is not None:
                rec = self._ownership[best_tid]
                if rec["owner"] is None or rec["owner"] == identity:
                    rec["owner"]     = identity
                    rec["last_seen"] = now

            # ── mark owner as currently visible (any bag they own) ─────────
            for tid, rec in self._ownership.items():
                if rec["owner"] == identity:
                    rec["last_seen"] = now

        return face_results

    def get_owner_absent_alerts(self) -> list:
        """
        Returns list of track_ids whose known owner has not been seen
        for longer than OWNER_ABSENT_THRESHOLD seconds.

        Only fires for bags that actually have an assigned owner (i.e. we
        saw them with the bag at some point).
        """
        now    = time.time()
        alerts = []
        for tid, rec in self._ownership.items():
            if rec["owner"] is None or rec["last_seen"] is None:
                continue
            absent_secs = now - rec["last_seen"]
            if absent_secs >= self.owner_absent_threshold:
                alerts.append({
                    "track_id"     : tid,
                    "owner"        : rec["owner"],
                    "absent_seconds": round(absent_secs, 1),
                })
        return alerts

    def reset_ownership(self, track_id: int) -> None:
        """Call when an alert has been acknowledged for a track."""
        if track_id in self._ownership:
            del self._ownership[track_id]

    # ── private helpers ─────────────────────────────────────────────────────

    def _match(self, embedding: np.ndarray) -> tuple:
        """
        Returns (name, similarity) for the best matching enrolled identity,
        or (None, 0.0) if below threshold.
        """
        best_name, best_sim = None, -1.0
        for name, refs in self._enrolled.items():
            sims = [_cosine_sim(embedding, r) for r in refs]
            avg  = float(np.mean(sims))
            if avg > best_sim:
                best_sim, best_name = avg, name
        if best_sim >= self.recognition_threshold:
            return best_name, best_sim
        return None, best_sim
