from ultralytics import YOLO

# Load YOLOv8n pretrained on COCO
model = YOLO("yolov8n.pt")

# Fine-tune on airport luggage dataset
results = model.train(
    data="data/luggage_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="yolov8_luggage",
    project="models",
    hsv_v=0.4,
    degrees=10,
    flipud=0.1,
    mosaic=1.0,
)

print("Training complete! Best weights saved to:", results.save_dir)
print("Training complete. Best weights saved to:", results.save_dir)