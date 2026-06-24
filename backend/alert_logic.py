"""
alert_engine.py  –  Member C
Unified Alert Engine

Combines signals from BagTracker and FaceRecognizer to produce
structured alert objects that the FastAPI server (Member D) can
forward to the frontend.

Alert types
-----------
  UNATTENDED_BAG  – bag stationary > threshold (no owner known OR owner absent)
  OWNER_LEFT      – bag's identified owner has left the scene
  ACCESS_VIOLATION– face not in enrolled database (optional, toggle via flag)
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Literal

AlertType = Literal["UNATTENDED_BAG", "OWNER_LEFT", "ACCESS_VIOLATION"]


@dataclass
class Alert:
    alert_type      : AlertType
    track_id        : int
    timestamp       : float     = field(default_factory=time.time)
    owner           : str | None = None
    absent_seconds  : float      = 0.0
    stationary_seconds: float    = 0.0
    bbox_ltrb       : list       = field(default_factory=list)
    acknowledged    : bool       = False

    def to_dict(self) -> dict:
        return asdict(self)


class AlertEngine:
    """
    Call process() once per frame with the outputs of BagTracker and
    FaceRecognizer.  It deduplicates alerts (won't re-raise the same
    alert for the same track until it has been cleared).

    Usage
    -----
    engine = AlertEngine()

    alerts = engine.process(bag_tracks, face_results, absent_alerts)
    # alerts is a (possibly empty) list of new Alert objects

    engine.acknowledge(track_id, alert_type)  # clear an alert
    active = engine.active_alerts()           # all un-acknowledged alerts
    """

    def __init__(self, detect_unknown_faces: bool = True):
        self.detect_unknown_faces = detect_unknown_faces
        # (track_id, alert_type) -> Alert
        self._active: dict[tuple, Alert] = {}

    # ── main entry point ─────────────────────────────────────────────────────

    def process(self,
                bag_tracks:    list,
                face_results:  list,
                absent_alerts: list) -> list:
        """
        Parameters
        ----------
        bag_tracks    : output of BagTracker.update()
        face_results  : output of FaceRecognizer.process_frame()
        absent_alerts : output of FaceRecognizer.get_owner_absent_alerts()

        Returns
        -------
        List of *new* Alert objects raised this frame.
        """
        new_alerts: list[Alert] = []

        # 1. Unattended bag (stationary + no owner info or owner absent)
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
                    # update stationary_seconds in place
                    self._active[key].stationary_seconds = track["stationary_seconds"]

        # 2. Owner left scene
        for absent in absent_alerts:
            tid = absent["track_id"]
            key = (tid, "OWNER_LEFT")
            if key not in self._active:
                alert = Alert(
                    alert_type      = "OWNER_LEFT",
                    track_id        = tid,
                    owner           = absent["owner"],
                    absent_seconds  = absent["absent_seconds"],
                )
                # find bbox from bag_tracks
                for t in bag_tracks:
                    if t["track_id"] == tid:
                        alert.bbox_ltrb = t["bbox_ltrb"]
                        break
                self._active[key] = alert
                new_alerts.append(alert)
            else:
                self._active[key].absent_seconds = absent["absent_seconds"]

        # 3. Unknown face / access violation
        if self.detect_unknown_faces:
            for fr in face_results:
                if fr["identity"] is None:
                    # Round bbox to nearest 30px so small jitter doesn't create duplicate alerts
                    pseudo_tid = hash(tuple(round(v / 30) for v in fr["bbox_ltrb"]))
                    key = (pseudo_tid, "ACCESS_VIOLATION")
                    if key not in self._active:
                        alert = Alert(
                            alert_type = "ACCESS_VIOLATION",
                            track_id   = pseudo_tid,
                            bbox_ltrb  = fr["bbox_ltrb"],
                        )
                        self._active[key] = alert
                        new_alerts.append(alert)

        return new_alerts

    # ── alert management ─────────────────────────────────────────────────────

    def acknowledge(self, track_id: int, alert_type: AlertType) -> None:
        key = (track_id, alert_type)
        if key in self._active:
            self._active[key].acknowledged = True
            del self._active[key]

    def active_alerts(self) -> list:
        return [a.to_dict() for a in self._active.values()]

    def clear_all(self) -> None:
        self._active.clear()
