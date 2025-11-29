import ultralytics
from ultralytics import YOLO
from pathlib import Path


ultralytics.checks()
model = YOLO("yolov8s.pt")
root = Path("./face_skin_yolo").resolve()  # root dataset path

model.train(
    data=str((root / "data.yaml")),
    epochs=100,
    imgsz=640,
    batch=16,
    project="runs",
    name="notebook_train",
    seed=0,
)

model.export(format="onnx", keras=True)
