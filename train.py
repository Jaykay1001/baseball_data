from ultralytics import YOLO

model = YOLO("yolo11l.pt")
model.train(
    data=r"data.yaml",
    epochs=100,
    imgsz=640,
    batch=16
)