import cv2
import numpy as np
from pathlib import Path

# COCO class IDs for luggage items (YOLOv8 pretrained baseline)
# These same IDs are used after fine-tuning since we keep the same class list.
LUGGAGE_CLASSES = [24, 26, 28]   # 24=backpack, 26=handbag, 28=suitcase
CLASS_NAMES     = {24: "backpack", 26: "handbag", 28: "suitcase"}
COLOURS         = {24: (0, 128, 255), 26: (0, 200, 100), 28: (255, 80, 0)}

# Path to model weights,  place the fine tuned file here after training
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "yolov8n_luggage.pt"


def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    """
    Load the YOLOv8 model. Falls back to the COCO pretrained weights if the
    fine tuned file doesn't exist yet.

    Returns:
        Ultralytics YOLO model instance.
    """
    from ultralytics import YOLO

    if model_path.exists():
        print(f"[detect] Loaded fine-tuned model: {model_path}")
        return YOLO(str(model_path))
    else:
        print("[detect] Fine-tuned weights not found — falling back to yolov8n.pt (COCO).")
        print(f"         Expected path: {model_path}")
        return YOLO("yolov8n.pt")


# Module level model instance (loaded once, reused per frame)
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def detect_luggage(frame: np.ndarray,
                   conf_threshold: float = 0.4) -> list[dict]:
    """
    Run YOLOv8 detection on a single enhanced BGR frame.

    Args:
        frame:          Enhanced BGR frame from enhance.py.
        conf_threshold: Minimum confidence to keep a detection (default 0.4).

    Returns:
        List of detection dicts, each with:
            bbox   — [x1, y1, x2, y2] in pixel coordinates
            conf   — confidence score (0–1)
            class  — COCO class ID (24, 26, or 28)
            label  — human-readable string ("backpack", "handbag", "suitcase")
    """
    model = _get_model()
    results = model(frame, conf=conf_threshold, classes=LUGGAGE_CLASSES, verbose=False)

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "bbox":  box.xyxy[0].tolist(),   
            "conf":  round(float(box.conf[0]), 3),
            "class": cls_id,
            "label": CLASS_NAMES.get(cls_id, "luggage"),
        })

    return detections


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draw bounding boxes and labels onto a copy of the frame.
    Args:
        frame:      BGR frame.
        detections: Output of detect_luggage().

    Returns:
        Annotated BGR frame (copy — original is not modified).
    """
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        colour = COLOURS.get(d["class"], (200, 200, 200))
        label  = f"{d['label']} {d['conf']:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        cv2.rectangle(out, (x1, y1 - 22), (x1 + len(label) * 9, y1), colour, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out



# Standalone test — run: python detect.py
# Processes a test image through the full enhance → detect pipeline.
if __name__ == "__main__":
    import sys
    from enhance import enhance_frame

    test_path = "test_frame.jpg"
    if not __import__("os").path.exists(test_path):
        print(f"Place a test image at '{test_path}' and re-run.")
        sys.exit(1)

    original  = cv2.imread(test_path)
    enhanced  = enhance_frame(original)
    detections = detect_luggage(enhanced)

    print(f"=== Detections ({len(detections)} found) ===")
    for d in detections:
        print(f"  {d['label']:10s}  conf={d['conf']}  bbox={[round(v) for v in d['bbox']]}")

    annotated = draw_detections(enhanced, detections)
    cv2.imwrite("output_detections.jpg", annotated)
    print("Saved: output_detections.jpg")