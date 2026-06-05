from ultralytics import YOLO
import numpy as np

class LuggageDetector:
    LUGGAGE_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}

    def __init__(self, weights_path: str = "models/yolov8_luggage/weights/best.pt", conf: float = 0.4):
        self.model = YOLO(weights_path)
        self.conf = conf

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Args:
            frame: BGR numpy array (from OpenCV)
        Returns:
            List of dicts: {bbox: [x1,y1,x2,y2], class_name: str, confidence: float}
        """
        results = self.model.predict(frame, conf=self.conf, classes=list(self.LUGGAGE_CLASSES.keys()), verbose=False)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class_name": self.LUGGAGE_CLASSES[int(box.cls)],
                "confidence": float(box.conf),
            })
        return detections