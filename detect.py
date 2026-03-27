from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="mein_video.mp4",
    save=True,
    conf=0.25,
    imgsz=1280,
    device=0
)