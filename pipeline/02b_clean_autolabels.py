"""Clean GroundingDINO autolabels.

Two filters:
  1. Drop boxes that are too large to be a football (width or height > MAX_REL).
  2. NMS-dedupe overlapping boxes that are the same physical ball.

Output: dataset/cleaned/{images,labels} ready for Roboflow upload.
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).parent.parent
SRC = ROOT / "dataset" / "autolabeled"
DST = ROOT / "dataset" / "cleaned"

MAX_REL = 0.15      # any box wider/taller than 15% of frame is not a ball
IOU_THRESH = 0.30   # dedupe boxes whose IoU > this

(DST / "images").mkdir(parents=True, exist_ok=True)
(DST / "labels").mkdir(parents=True, exist_ok=True)


def to_xyxy(b):
    cx, cy, w, h = b
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a, b):
    ax1, ay1, ax2, ay2 = to_xyxy(a)
    bx1, by1, bx2, by2 = to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def nms(boxes):
    # boxes: list of (cx,cy,w,h), prefer smaller boxes (tighter on the ball)
    boxes = sorted(boxes, key=lambda b: b[2] * b[3])
    keep = []
    for b in boxes:
        if all(iou(b, k) < IOU_THRESH for k in keep):
            keep.append(b)
    return keep


stats = {"in": 0, "dropped_huge": 0, "dropped_nms": 0, "out": 0}

for split in ("train", "valid"):
    for lbl_path in sorted((SRC / split / "labels").glob("*.txt")):
        raw_boxes = []
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            stats["in"] += 1
            if w > MAX_REL or h > MAX_REL:
                stats["dropped_huge"] += 1
                continue
            raw_boxes.append((cx, cy, w, h))
        kept = nms(raw_boxes)
        stats["dropped_nms"] += len(raw_boxes) - len(kept)
        stats["out"] += len(kept)

        # Write cleaned label
        out_lbl = DST / "labels" / lbl_path.name
        out_lbl.write_text("\n".join(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for b in kept))

        # Copy matching image
        img_path = SRC / split / "images" / (lbl_path.stem + ".jpg")
        if img_path.exists():
            shutil.copy2(img_path, DST / "images" / img_path.name)

print(f"in={stats['in']}  dropped_huge={stats['dropped_huge']}  "
      f"dropped_nms={stats['dropped_nms']}  out={stats['out']}")
print(f"cleaned -> {DST}")
