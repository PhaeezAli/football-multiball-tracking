"""Fine-tune YOLOv11 on the football dataset.

NOTE: yolo11m at imgsz=1280 does NOT fit an 18GB M3 Pro. batch=8 hard-OOMs
(peaks 21.7/22.6 GiB); batch=4 allocates 16.4 GiB and thrashes swap to a
standstill (2 of 65 iters in 48 min). Train on CUDA instead —
see train_football_colab.ipynb. To train locally, use yolo11s.pt.
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
DATA = ROOT / "dataset" / "yolo" / "data.yaml"

model = YOLO("yolo11m.pt")   # yolo11s.pt if you must run this on MPS — see module docstring
model.train(
    data=str(DATA),
    epochs=60,
    imgsz=1280,       # balls are ~20px; lowering this is not an option
    batch=4,

    device="mps",
    patience=15,
    cos_lr=True,
    mosaic=1.0,
    mixup=0.1,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=5, translate=0.1, scale=0.5, fliplr=0.5,
    project=str(ROOT / "runs"),
    name="football_yolo11m",
    exist_ok=True,
)
print("Best weights at runs/football_yolo11m/weights/best.pt")
