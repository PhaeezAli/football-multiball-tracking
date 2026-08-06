"""Render bbox overlays on a sample of cleaned labels for visual sanity check."""
import random
from pathlib import Path
import cv2

ROOT = Path(__file__).parent.parent
SRC = ROOT / "dataset" / "cleaned"
OUT = ROOT / "dataset" / "preview"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(0)
imgs = sorted((SRC / "images").glob("*.jpg"))
sample = random.sample(imgs, min(16, len(imgs)))

for img_path in sample:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    lbl = SRC / "labels" / (img_path.stem + ".txt")
    n = 0
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = parts
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            n += 1
    cv2.putText(img, f"{img_path.stem}  n={n}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(str(OUT / img_path.name), img)

print(f"wrote {len(sample)} previews to {OUT}")
