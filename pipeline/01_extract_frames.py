"""Extract frames from test videos for labeling.

test.mp4 (306f) and test2.mp4 (269f) sampled at stride=2 ~= 287 frames.
test2.mp4 is 4K — downscale to 1920w to keep label/training tractable.
"""
import cv2
from pathlib import Path

OUT = Path(__file__).parent.parent / "dataset" / "images"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ("vdo/test.mp4",  2, 1280),
    ("vdo/test2.mp4", 2, 1920),
]

ROOT = Path(__file__).parent.parent
total = 0
for rel, stride, max_w in SOURCES:
    path = ROOT / rel
    cap = cv2.VideoCapture(str(path))
    stem = path.stem
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % stride == 0:
            h, w = frame.shape[:2]
            if w > max_w:
                scale = max_w / w
                frame = cv2.resize(frame, (max_w, int(h * scale)))
            out_path = OUT / f"{stem}_{i:05d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            total += 1
        i += 1
    cap.release()
    print(f"{rel}: extracted {i // stride + (1 if i % stride else 0)} frames")

print(f"\nTotal: {total} frames in {OUT}")
