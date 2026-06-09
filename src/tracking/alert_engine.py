"""
alert_engine.py  –  Member C
Unified Alert Engine

Combines signals from BagTracker and FaceRecognizer to produce structured
Alert objects that Member D's FastAPI server can forward to the React frontend.

Alert types
-----------
  UNATTENDED_BAG   – bag stationary > threshold (regardless of whether owner known)
  OWNER_LEFT       – bag's identified owner has been absent > threshold seconds
  ACCESS_VIOLATION – face detected that is not in the enrolled personnel database

Design notes
------------
- Alerts are deduplicated: the same (track_id, alert_type) pair is only raised
  ONCE until it has been acknowledged.  Subsequent frames update the in-place
  counters (stationary_seconds, absent_seconds) so the frontend always shows
  fresh values without generating duplicate events.
- acknowledge() removes the alert from _active AND returns its final state as
  a dict — so Member D can log it before discarding.  (Bug fix: previously the
  acknowledged=True flag was set then immediately lost when the object was deleted.)
- ACCESS_VIOLATION uses a monotonic negative counter for track IDs so they
  never clash with real bag track IDs (which are always positive integers).
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Literal


AlertType = Literal["UNATTENDED_BAG", "OWNER_LEFT", "ACCESS_VIOLATION"]


@dataclass
class Alert:
    alert_type        : AlertType
    track_id          : int
    timestamp         : float      = field(default_factory=time.time)
    owner             : str | None = None
    absent_seconds    : float      = 0.0
    stationary_seconds: float      = 0.0
    bbox_ltrb         : list       = field(default_factory=list)
    acknowledged      : bool       = False

    def to_dict(self) -> dict:
        return asdict(self)


class AlertEngine:
    """
    Processes one frame's worth of tracking + face-recognition output and
    produces new Alert objects when conditions are met.

    Usage
    -----
    engine = AlertEngine()

    # once per frame:
    new_alerts = engine.process(bag_tracks, face_results, absent_alerts)

    # to acknowledge (dismiss) an alert from the frontend:
    final_state = engine.acknowledge(track_id, alert_type)

    # to query what is currently active:
    active = engine.active_alerts()
    """

    def __init__(self, detect_unknown_faces: bool = True):
        """
        Parameters
        ----------
        detect_unknown_faces : set False to disable ACCESS_VIOLATION alerts
                               (e.g. in public areas where unknown faces are expected)
        """
        self.detect_unknown_faces = detect_unknown_faces
        # (track_id, alert_type) -> Alert
        self._active: dict[tuple, Alert] = {}
        # Monotonic counter for ACCESS_VIOLATION pseudo track IDs.
        # Always negative so they never collide with real bag track IDs.
        self._violation_counter = 0

    # ── Main entry point ──────────────────────────────────────────────────────

    def process(self,
                bag_tracks:    list,
                face_results:  list,
                absent_alerts: list) -> list:
        """
        Call once per frame with the outputs of BagTracker and FaceRecognizer.

        Parameters
        ----------
        bag_tracks    : output of BagTracker.update()
        face_results  : output of FaceRecognizer.process_frame()
        absent_alerts : output of FaceRecognizer.get_owner_absent_alerts()

        Returns
        -------
        List of *new* Alert objects raised this frame (may be empty).
        Alerts that were already active are updated in-place, not re-raised.
        """
        new_alerts: list[Alert] = []

        # Build a quick-lookup set of active bag track IDs for cleanup below
        active_track_ids = {t["track_id"] for t in bag_tracks}

        # ── 1. UNATTENDED_BAG ─────────────────────────────────────────────
        for track in bag_tracks:
            tid = track["track_id"]
            if track["alert"]:
                key = (tid, "UNATTENDED_BAG")
                if key not in self._active:
                    alert = Alert(
                        alert_type         = "UNATTENDED_BAG",
                        track_id           = tid,
                        stationary_seconds = track["stationary_seconds"],
                        bbox_ltrb          = track["bbox_ltrb"],
                    )
                    self._active[key] = alert
                    new_alerts.append(alert)
                else:
                    # Update live counter without re-raising
                    self._active[key].stationary_seconds = track["stationary_seconds"]
                    self._active[key].bbox_ltrb          = track["bbox_ltrb"]

        # ── 2. OWNER_LEFT ─────────────────────────────────────────────────
        # Build a set of track IDs that currently have an absent-owner alert
        absent_tids = {a["track_id"] for a in absent_alerts}

        for absent in absent_alerts:
            tid = absent["track_id"]
            key = (tid, "OWNER_LEFT")
            if key not in self._active:
                alert = Alert(
                    alert_type     = "OWNER_LEFT",
                    track_id       = tid,
                    owner          = absent["owner"],
                    absent_seconds = absent["absent_seconds"],
                )
                # Attach the bag's current bounding box if still tracked
                for t in bag_tracks:
                    if t["track_id"] == tid:
                        alert.bbox_ltrb = t["bbox_ltrb"]
                        break
                self._active[key] = alert
                new_alerts.append(alert)
            else:
                # Update live counter without re-raising
                self._active[key].absent_seconds = absent["absent_seconds"]

        # ── 3. ACCESS_VIOLATION ───────────────────────────────────────────
        # One alert per unique unknown-face bounding box position per session.
        # A face moving slightly across frames generates a new alert (acceptable
        # behaviour — it means the unknown person is still present and moving).
        if self.detect_unknown_faces:
            for fr in face_results:
                if fr["identity"] is None:
                    bbox_key = tuple(fr["bbox_ltrb"])
                    # Check for a spatially identical active violation
                    already_active = any(
                        tuple(a.bbox_ltrb) == bbox_key
                        for (_, atype), a in self._active.items()
                        if atype == "ACCESS_VIOLATION"
                    )
                    if not already_active:
                        self._violation_counter += 1
                        pseudo_tid = -self._violation_counter   # always negative
                        key = (pseudo_tid, "ACCESS_VIOLATION")
                        alert = Alert(
                            alert_type = "ACCESS_VIOLATION",
                            track_id   = pseudo_tid,
                            bbox_ltrb  = list(bbox_key),
                        )
                        self._active[key] = alert
                        new_alerts.append(alert)

        # ── 4. Cleanup stale OWNER_LEFT alerts ────────────────────────────
        # If the absent-owner alert is no longer being reported by FaceRecognizer
        # (e.g. the owner came back, or the bag was removed), clear it automatically.
        # We iterate over a snapshot of keys to safely modify _active inside the loop.
        for key in list(self._active.keys()):
            tid, atype = key
            if atype == "OWNER_LEFT" and tid not in absent_tids:
                del self._active[key]

        return new_alerts

    # ── Alert management ──────────────────────────────────────────────────────

    def acknowledge(self, track_id: int, alert_type: AlertType) -> dict | None:
        """
        Dismiss an active alert.

        Sets acknowledged=True on the Alert object, removes it from the active
        table, and returns its final state as a dict so Member D can log it.

        For OWNER_LEFT alerts: the caller should also call
        face_rec.reset_ownership(track_id) so the owner can be re-associated
        with their bag if they return to the scene.

        Returns
        -------
        The final alert dict (with acknowledged=True), or None if the alert
        was not found (e.g. already auto-cleared by step 4 above).
        """
        key = (track_id, alert_type)
        if key not in self._active:
            return None
        alert = self._active.pop(key)    # remove from active table
        alert.acknowledged = True        # mark AFTER removing, on the live object
        return alert.to_dict()           # return final state for Member D to log

    def active_alerts(self) -> list:
        """Return all current un-acknowledged alerts as a list of dicts."""
        return [a.to_dict() for a in self._active.values()]

    def clear_all(self) -> None:
        """Remove all active alerts (e.g. on scene reset or end of shift)."""
        self._active.clear()
