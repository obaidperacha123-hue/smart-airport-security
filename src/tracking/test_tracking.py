"""
test_tracking.py  –  Member C
Standalone smoke-test — runs the full tracking + face pipeline on
a webcam feed (or a video file if you pass a path as sys.argv[1]).

Usage
-----
  python test_tracking.py                  # webcam
  python test_tracking.py path/to/vid.mp4  # video file

Press  q  to quit.  Press  a  to print current active alerts.
"""

import sys
import cv2
import numpy as np
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents))
from src.tracking import BagTracker, FaceRecognizer, AlertEngine


def _stub_detect_bags(frame: np.ndarray) -> list:
    h, w = frame.shape[:2]
    bx, by = int(w * 0.75), int(h * 0.75)
    bw, bh = 80, 60
    return [([bx, by, bw, bh], 0.85, "suitcase")]


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open source: {source}")
        sys.exit(1)

    tracker  = BagTracker(stationary_threshold=10)  
    face_rec = FaceRecognizer(owner_absent_threshold=8)
    face_rec.load_db()
    engine   = AlertEngine()

    print("Running… press Q to quit, A to print alerts")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Bag detection (stub layout)
        detections = _stub_detect_bags(frame)

        # 2. Tracking
        bag_tracks = tracker.update(frame, detections)
        
        # Normalize track IDs immediately before they enter any other teammate's subsystem module
        for track in bag_tracks:
            if abs(track["track_id"]) > 10000:
                track["track_id"] = int(abs(track["track_id"]) % 10000)

        # 3. Face recognition processing
        face_results  = face_rec.process_frame(frame, bag_tracks)
        
        # Clean face log reference IDs if they are linked to the tracking hashes
        for face in face_results:
            if "track_id" in face and face["track_id"] is not None:
                if abs(face["track_id"]) > 10000:
                    face["track_id"] = int(abs(face["track_id"]) % 10000)

        absent_alerts = face_rec.get_owner_absent_alerts()

        # 4. Alert engine pipeline processing execution
        new_alerts = engine.process(bag_tracks, face_results, absent_alerts)
        for a in new_alerts:
            # Clean up the output string directly if it still holds raw string structures
            log_msg = str(a)
            print(f"[ALERT] {log_msg}")

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
