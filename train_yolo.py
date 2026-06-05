from ultralytics import YOLO

# Load YOLOv8n pretrained on COCO (includes backpack/handbag/suitcase)
model = YOLO("yolov8n.pt")

# Fine-tune on airport luggage dataset
results = model.train(
    data="data/luggage_dataset/data.yaml",  # path to your dataset yaml
    epochs=50,
    imgsz=640,
    batch=16,
    name="yolov8_luggage",
    project="models",
    # Augmentation to simulate CCTV conditions:
    hsv_v=0.4,        # brightness variation
    degrees=10,       # slight rotation (overhead camera)
    flipud=0.1,       # vertical flip (overhead CCTV)
    mosaic=1.0,
)

print("Training complete. Best weights saved to:", results.save_dir)