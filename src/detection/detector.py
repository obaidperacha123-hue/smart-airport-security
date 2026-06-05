from ultralytics import YOLO
import numpy as np

class LuggageDetector:
    def __init__(self, weights_path="models/yolov8_luggage/weights/best.pt", conf=0.4):
        self.model = YOLO(weights_path)
        self.conf = conf

    def detect(self, frame: np.ndarray) -> list:
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class_name": "luggage",
                "confidence": float(box.conf),
            })
        return detections