"""Auto-label football detections with GroundingDINO.

Output: YOLO-format labels in dataset/labels/ (single class 'football' = 0)
Also writes dataset/data.yaml ready for ultralytics training after cleanup.
"""
from pathlib import Path
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

ROOT = Path(__file__).parent.parent
IMG_DIR = ROOT / "dataset" / "images"
LBL_DIR = ROOT / "dataset" / "labels"
LBL_DIR.mkdir(parents=True, exist_ok=True)

# Ontology: caption -> class_name. GroundingDINO uses the caption as the text prompt.
ontology = CaptionOntology({
    "soccer ball on the ground": "football",
})

base_model = GroundingDINO(
    ontology=ontology,
    box_threshold=0.25,
    text_threshold=0.20,
)

# Label the whole folder. autodistill writes labels alongside in YOLO format.
out = ROOT / "dataset"
base_model.label(
    input_folder=str(IMG_DIR),
    output_folder=str(out / "autolabeled"),
    extension=".jpg",
)

print(f"Labels written to {out / 'autolabeled'}")
print("Inspect, then upload to Roboflow for cleanup.")
