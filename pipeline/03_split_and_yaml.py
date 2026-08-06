"""After Roboflow cleanup, drop the export ZIP into dataset/cleaned/.
This script splits the cleaned images into train/val (90/10) and writes data.yaml.

If you skip Roboflow and trust autolabels as-is, point CLEANED_DIR at autolabeled/.
"""
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
CLEANED_DIR = ROOT / "dataset" / "cleaned"   # expects images/ and labels/ inside
OUT = ROOT / "dataset" / "yolo"
VAL_RATIO = 0.10
SEED = 42

random.seed(SEED)

src_imgs = CLEANED_DIR / "images"
src_lbls = CLEANED_DIR / "labels"
assert src_imgs.exists(), f"missing {src_imgs} -- export from Roboflow first"

for split in ("train", "val"):
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

imgs = sorted([p for p in src_imgs.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
random.shuffle(imgs)
n_val = max(1, int(len(imgs) * VAL_RATIO))
val_set = set(imgs[:n_val])

for img in imgs:
    split = "val" if img in val_set else "train"
    shutil.copy2(img, OUT / "images" / split / img.name)
    lbl = src_lbls / (img.stem + ".txt")
    if lbl.exists():
        shutil.copy2(lbl, OUT / "labels" / split / lbl.name)
    else:
        # negative sample — write empty label file so YOLO treats it as background
        (OUT / "labels" / split / (img.stem + ".txt")).touch()

yaml = f"""path: {OUT}
train: images/train
val: images/val
names:
  0: football
"""
(OUT / "data.yaml").write_text(yaml)
print(f"train={len(imgs)-n_val}  val={n_val}")
print(f"data.yaml -> {OUT / 'data.yaml'}")
