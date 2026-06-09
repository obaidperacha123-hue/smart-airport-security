"""
test_tracking.py  –  Member C
Standalone smoke-test for the full tracking + face pipeline.

Runs on a webcam feed (default) or any video file passed as sys.argv[1].

Usage
-----
  python test_tracking.py                     # webcam
  python test_tracking.py path/to/vid.mp4     # video file

Keyboard shortcuts
------------------
  Q  — quit
  A  — print current active alerts to the terminal
  E  — enrol your own face as "TestOwner" from the live webcam frame
       (lets you test OWNER_LEFT: show face → walk out of frame → wait 8 s)

How to trigger every alert type during the demo
-----------------------------------------------
  UNATTENDED_BAG  — the stub bag appears in the frame and never moves.
                    After 10 seconds (stationary_threshold=10 for quick testing)
                    the box turns RED and the alert fires automatically.

  OWNER_LEFT      — press E while your face is visible to enrol yourself.
                    The system will associate you with the bag.
                    Walk out of webcam view (or cover your face) for > 8 s.
                    The OWNER_LEFT alert fires once the timer expires.

  ACCESS_VIOLATION— point the camera at any face that is NOT enrolled in the DB.
                    An ACCESS_VIOLATION alert fires immediately for each new
                    unknown face position detected.

NOTE
----
  This script bypasses YOLOv8 and uses a stub detection so it runs without
  the full model stack.  In production, detections come from Member B's
  LuggageDetector (detector.py).
"""

import sys
import cv2
import numpy as np
from pathlib import Path


#
# parents[0] = directory containing this file (repo root)
# We insert that onto sys.path so  `from src.tracking import ...`  resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.tracking import BagTracker, FaceRecognizer, AlertEngine


# ── Stub bag detection ────────────────────────────────────────────────────────

_frame_counter = 0   # used to gently oscillate the bag so you can see movement


def _stub_detect_bags(frame: np.ndarray) -> list:

    global _frame_counter
    _frame_counter += 1

    h, w   = frame.shape[:2]
    cx, cy = w // 2, h // 2
    bw, bh = 120, 90

    drift = int(2 * np.sin(_frame_counter / 50.0))

    x1 = cx - bw // 2 + drift
    y1 = cy - bh // 2
    return [([x1, y1, bw, bh], 0.88, "suitcase")]


# ── Draw utilities ────────────────────────────────────────────────────────────

def _draw_bag_tracks(frame: np.ndarray, bag_tracks: list) -> None:
    for t in bag_tracks:
        x1, y1, x2, y2 = t["bbox_ltrb"]
        colour = (0, 0, 255) if t["alert"] else (0, 220, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = (f"Bag ID:{t['track_id']}  {t['stationary_seconds']:.0f}s"
                 + ("  ALERT" if t["alert"] else ""))
        cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def _draw_face_results(frame: np.ndarray, face_results: list) -> None:
    for f in face_results:
        x1, y1, x2, y2 = f["bbox_ltrb"]
        colour = (255, 165, 0) if f["identity"] else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f["identity"] or f"unknown ({f['similarity']:.2f})"
        cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, engine: AlertEngine) -> None:
    """Overlay active alert count and key hints."""
    n = len(engine.active_alerts())
    colour = (0, 0, 255) if n > 0 else (180, 180, 180)
    cv2.putText(frame, f"Active alerts: {n}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)
    cv2.putText(frame, "Q=quit  A=print alerts  E=enrol face", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    source = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].isdigit()) \
             else (sys.argv[1] if len(sys.argv) > 1 else 0)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    # Use shorter thresholds for quick smoke-testing.
    # In production these are STATIONARY_THRESHOLD_SECONDS=30, OWNER_ABSENT_THRESHOLD=20.
    tracker  = BagTracker(stationary_threshold=10)    # 10 s → alert in quick test
    face_rec = FaceRecognizer(owner_absent_threshold=8)  # 8 s → owner-left alert
    face_rec.load_db()
    engine   = AlertEngine(detect_unknown_faces=True)

    print("\n=== Smart Airport — Tracking Smoke-Test ===")
    print("  Q = quit      A = print active alerts      E = enrol your face")
    print("  UNATTENDED_BAG fires after 10 s (threshold reduced for testing)")
    print("  OWNER_LEFT fires 8 s after your face leaves the frame")
    print("  ACCESS_VIOLATION fires immediately for any unknown face\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        # ── Pipeline ──────────────────────────────────────────────────────
        detections    = _stub_detect_bags(frame)
        bag_tracks, fg_mask = tracker.update(frame, detections)
        face_results  = face_rec.process_frame(frame, bag_tracks)
        absent_alerts = face_rec.get_owner_absent_alerts()
        new_alerts    = engine.process(bag_tracks, face_results, absent_alerts)

        for alert in new_alerts:
            print(f"[ALERT] {alert}")

        # ── Draw ──────────────────────────────────────────────────────────
        _draw_bag_tracks(frame, bag_tracks)
        _draw_face_results(frame, face_results)
        _draw_hud(frame, engine)

        cv2.imshow("Smart Airport — Tracking Test", frame)

        # ── Keys ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("a"):
            alerts = engine.active_alerts()
            print(f"\n[Active alerts ({len(alerts)})]")
            for a in alerts:
                print(" ", a)
            print()

        elif key == ord("e"):
            print("[ENROL] Capturing frame for enrolment as 'TestOwner' …")
            success = face_rec.enrol("TestOwner", frame)
            if success:
                print("[ENROL] Enrolled successfully. Walk into frame to associate "
                      "with the bag, then leave to trigger OWNER_LEFT.")
            else:
                print("[ENROL] No face detected. Move closer to the camera and try again.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Test finished.")


if __name__ == "__main__":
    main()
