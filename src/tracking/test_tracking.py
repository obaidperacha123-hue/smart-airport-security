"""
test_tracking.py  –  Member C
Standalone smoke-test — runs the full tracking + face pipeline on
a webcam feed (or a video file if you pass a path as sys.argv[1]).

Usage
-----
  python test_tracking.py                  # webcam
  python test_tracking.py path/to/vid.mp4  # video file

Press  q  to quit.  Press  a  to print current active alerts.

NOTE: This test script bypasses YOLOv8 and uses a simple Haar cascade
      for person/face detection so it runs without the full model stack.
      In production the detections come from Member B's detector.py.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# ── allow running from repo root without installing the package ───────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.tracking import BagTracker, FaceRecognizer, AlertEngine


# ── tiny stub: replace with real YOLOv8 detections in production ─────────────
def _stub_detect_bags(frame: np.ndarray) -> list:
    """
    Returns fake detections in DeepSORT format: ([x,y,w,h], conf, label)
    In production Member B's LuggageDetector returns this format.
    """
    h, w = frame.shape[:2]
    # single static "bag" in the centre for demo purposes
    cx, cy = w // 2, h // 2
    bw, bh = 100, 80
    return [([cx - bw//2, cy - bh//2, bw, bh], 0.85, "suitcase")]


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open source: {source}")
        sys.exit(1)

    tracker  = BagTracker(stationary_threshold=10)  # 10 s for quick testing
    face_rec = FaceRecognizer(owner_absent_threshold=8)
    face_rec.load_db()
    engine   = AlertEngine()

    print("Running… press Q to quit, A to print alerts")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Bag detection (stub → replace with real YOLO detections)
        detections = _stub_detect_bags(frame)

        # 2. Tracking
        bag_tracks = tracker.update(frame, detections)

        # 3. Face recognition
        face_results  = face_rec.process_frame(frame, bag_tracks)
        absent_alerts = face_rec.get_owner_absent_alerts()

        # 4. Alert engine
        new_alerts = engine.process(bag_tracks, face_results, absent_alerts)
        for a in new_alerts:
            print(f"[ALERT] {a}")

        # ── draw bounding boxes ───────────────────────────────────────────
        for t in bag_tracks:
            x1, y1, x2, y2 = t["bbox_ltrb"]
            colour = (0, 0, 255) if t["alert"] else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            label = (f"ID:{t['track_id']}  "
                     f"{t['stationary_seconds']:.0f}s"
                     + ("  ⚠ ALERT" if t["alert"] else ""))
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        for f in face_results:
            x1, y1, x2, y2 = f["bbox_ltrb"]
            label = f["identity"] or "unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

        cv2.imshow("Smart Airport – Tracking Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            print("Active alerts:", engine.active_alerts())

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
