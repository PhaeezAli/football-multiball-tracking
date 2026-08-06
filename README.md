# Multi-Ball Football Tracking & Possession Analysis

Tracks several footballs simultaneously through occlusion, works out which one is
actually in play, and labels per-frame possession state from player pose.

The hard part isn't tracking a ball — it's tracking **four**, three of which sit
still as regional markers while a player drills with the fourth, and keeping IDs
stable when they overlap or briefly vanish.

## Results

A single-class ball detector distilled from Grounding DINO's zero-shot labels:

| metric | value |
|---|---|
| mAP50 | 0.966 |
| mAP50-95 | 0.853 |
| precision | 0.933 |
| recall | 0.971 |

YOLO11m, 1280 px, 60 epochs, 260 train / 28 val frames.

Measured against auto-generated labels, not human annotation — see
[Label noise](#label-noise-the-student-beat-the-teacher) below, which is the more
interesting result.

End-to-end on a 306-frame clip: **3.80 ball detections/frame** (above the 3.64
average in the ground-truth labels), 14 successful track re-links.

## How it works

```
video ─┬─ YOLO11m ball detector ── BoT-SORT ──┬─ track stitching ─┬─ state machine ─ annotated
       └─ YOLO pose (multi-person) ── ankles ─┘   (re-ID)          │                   video +
                                                                    └─ zone of control   stats
```

**Detection + tracking.** A purpose-trained single-class detector feeds BoT-SORT.
Off-the-shelf COCO `sports ball` was the starting point and loses balls
constantly — on a 6-frame sample it found 20 balls where the trained detector
found 24.

**Track stitching.** BoT-SORT issues a fresh ID when a ball is occluded and
reappears. A re-identification layer keeps recently-lost tracks in memory for
30 frames and re-links a new detection to a lost one within 100 px, preserving
ID identity across occlusion.

**Possession via zone of control.** Multi-person pose gives ankle keypoints; a
ball within 120 px of any ankle is in that player's zone of control.

**State machine.** Distinguishes the in-play ball from stationary decoys —
a ball stationary for 30+ consecutive frames is reclassified as a marker and
drawn as a box rather than tracked as the action ball.

## Label noise: the student beat the teacher

Ball count cannot physically change frame to frame, so any variation in labels
per frame is annotation error. Grounding DINO's zero-shot labels:

| balls labeled | test.mp4 | test2.mp4 |
|---|---|---|
| 2 | 1.3% | 11.9% |
| 3 | 21.6% | 28.9% |
| 4 | **75.2%** | **57.8%** |
| 5 | 2.0% | 1.5% |

Roughly a quarter of frames in one video and over 40% in the other are
mislabeled — usually a dropped ball, occasionally a hallucinated one.

The trained detector **recovers balls its own training labels omit**, at high
confidence (0.83 on a basketball Grounding DINO skipped entirely). So the
reported precision of 0.933 is understated: some "false positives" are label
errors, not model errors. Recall (0.971) and mAP50-95 (0.853) are the figures to
trust.

## Pipeline

| script | does |
|---|---|
| `pipeline/01_extract_frames.py` | sample frames from source video, downscale 4K |
| `pipeline/02_autolabel.py` | zero-shot label with Grounding DINO |
| `pipeline/02b_clean_autolabels.py` | drop implausibly large boxes, NMS-dedupe |
| `pipeline/02c_preview_labels.py` | render overlays for visual QA |
| `pipeline/03_split_and_yaml.py` | train/val split + data.yaml |
| `pipeline/04_train.py` | fine-tune YOLO11m |
| `pipeline/05_analyze.py` | end-to-end tracking + possession analysis |
| `notebooks/train_football_colab.ipynb` | the training run, on a CUDA GPU |
| `notebooks/football_analysis_colab.ipynb` | earlier Colab version of the analysis pipeline |
| `legacy/` | initial prototypes, superseded by `pipeline/05_analyze.py` |

## Running it

```bash
pip install ultralytics opencv-python

# analysis with the trained detector (weights from Releases -> pipeline/football_best.pt)
python pipeline/05_analyze.py input.mp4 output.mp4
```

Rebuilding the dataset from scratch additionally needs
`autodistill autodistill-grounding-dino`, then run `pipeline/01` → `04` in order.

### A note on hardware

YOLO11m at 1280 px does **not** fit an 18 GB Apple Silicon machine. `batch=8`
hard-OOMs on MPS at 21.7/22.6 GiB; `batch=4` allocates 16.4 GiB and thrashes
swap to a standstill. Train on CUDA — `notebooks/train_football_colab.ipynb` does this on a
free T4 in about an hour at `batch=4`. Inference on MPS is fine.

## Not included

Source footage and the frames extracted from it are not redistributed here. Point
`05_analyze.py` at your own video.
