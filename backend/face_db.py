"""
face_db.py — Persistent face-embedding database for the enrolment system.

Member D owns this file.
Member C's InsightFaceMatcher calls face_db.match() to identify people.
The /enrol endpoint calls face_db.enrol() to register new staff.

Storage:  data/face_db.pkl
Schema:   { name_lowercase: List[np.ndarray (512-dim ArcFace embedding)] }

Matching: cosine similarity — faster and more robust than Euclidean for
          high-dimensional ArcFace vectors.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default path — override via constructor for testing
FACE_DB_PATH = Path("data/face_db.pkl")


class FaceDatabase:
    """
    Thread-safe*-ish face embedding store.
    (* FastAPI runs async; heavy CPU calls go to executor threads, but writes
       are always from the main async context, so no real concurrency issue.)
    """

    def __init__(self, db_path: Path = FACE_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Core store: name → list of 512-dim ArcFace embeddings
        self.embeddings: Dict[str, List[np.ndarray]] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load the database from disk on startup."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "rb") as fh:
                    self.embeddings = pickle.load(fh)
                logger.info(
                    f"Face DB loaded from {self.db_path}: "
                    f"{self.count} person(s), "
                    f"{sum(len(v) for v in self.embeddings.values())} total embeddings"
                )
            except Exception as exc:
                logger.error(f"Failed to load face DB ({exc}), starting fresh")
                self.embeddings = {}
        else:
            logger.info("No face DB found on disk — starting empty")

    def save(self) -> None:
        """Persist the database to disk."""
        try:
            with open(self.db_path, "wb") as fh:
                pickle.dump(self.embeddings, fh)
            logger.info(
                f"Face DB saved → {self.db_path} "
                f"({self.count} persons)"
            )
        except Exception as exc:
            logger.error(f"Failed to save face DB: {exc}")

    # ── Write operations ──────────────────────────────────────────────────────

    def enrol(self, name: str, embedding: np.ndarray) -> None:
        """
        Add one ArcFace embedding for a person.
        Multiple calls with the same name accumulate embeddings —
        more embeddings → more robust matching from different angles.
        """
        key = name.strip().lower()
        if key not in self.embeddings:
            self.embeddings[key] = []
        self.embeddings[key].append(embedding.astype(np.float32))
        self.save()
        logger.info(
            f"Enrolled '{key}': "
            f"{len(self.embeddings[key])} embedding(s) total"
        )

    def remove(self, name: str) -> bool:
        """Delete all embeddings for a person. Returns True if found."""
        key = name.strip().lower()
        if key in self.embeddings:
            del self.embeddings[key]
            self.save()
            logger.info(f"Removed '{key}' from face DB")
            return True
        return False

    def clear(self) -> None:
        """Wipe the entire database (useful for testing)."""
        self.embeddings.clear()
        self.save()
        logger.warning("Face DB cleared")

    # ── Read / match operations ───────────────────────────────────────────────

    def match(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.40,
    ) -> Optional[Tuple[str, float]]:
        """
        Find the closest enrolled person to a query embedding.

        Returns:
            (name, similarity_score)  — if best match >= threshold
            None                       — if DB is empty or no match above threshold

        ArcFace embeddings are L2-normalised internally; cosine similarity is
        therefore equivalent to dot product of normalised vectors.
        The default threshold of 0.40 is conservative for CCTV conditions
        (partial occlusion, compression artefacts, off-angle faces).
        """
        if not self.embeddings:
            return None

        # Normalise query once
        q = query_embedding.astype(np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)

        best_name: Optional[str] = None
        best_sim: float = -1.0

        for name, emb_list in self.embeddings.items():
            for emb in emb_list:
                e_norm = emb / (np.linalg.norm(emb) + 1e-8)
                sim = float(np.dot(q_norm, e_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_name = name

        if best_sim >= threshold and best_name is not None:
            return best_name, round(best_sim, 4)
        return None

    # ── Utility ───────────────────────────────────────────────────────────────

    def list_persons(self) -> Dict[str, int]:
        """Return {name: num_embeddings} for the admin panel."""
        return {name: len(embs) for name, embs in self.embeddings.items()}

    @property
    def count(self) -> int:
        """Number of distinct persons enrolled."""
        return len(self.embeddings)
