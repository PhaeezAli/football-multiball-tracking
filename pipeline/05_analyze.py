"""End-to-end multi-ball tracking + possession analysis using our own detector.

Two changes vs the Colab notebook this is derived from:
  1. Detector is our fine-tuned football_best.pt (single class 0) instead of
     yolov8x + COCO class 32 'sports ball'.
  2. Pose comes from yolov8n-pose.pt instead of mediapipe. The py3.13 arm64
     mediapipe wheels are Tasks-only (no mp.solutions), and YOLO pose is
     multi-person whereas mp.solutions.pose only ever returns one player.

Usage:  python pipeline/05_analyze.py [input.mp4] [output.mp4]
"""
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent

BALL_WEIGHTS = ROOT / "pipeline" / "football_best.pt"
POSE_WEIGHTS = ROOT / "yolov8n-pose.pt"

CONF = 0.3
DEVICE = "mps"

# --- analysis thresholds (carried over from the notebook) ---
HISTORY_LENGTH = 60
STATIONARY_SPEED_THRESHOLD = 2.0    # px/frame
POSSESSION_PROXIMITY_THRESHOLD = 120  # px from an ankle
MARKER_STATIONARY_FRAMES = 30       # stationary this long => a marker, not the action ball

# --- track stitching ---
REACQUIRE_THRESHOLD = 100
MAX_LOST_FRAMES = 30

L_ANKLE, R_ANKLE = 15, 16   # COCO keypoint indices
KP_CONF = 0.5

STATE_COLORS = {
    "IN_POSSESSION": (0, 0, 255),
    "STATIONARY": (255, 0, 0),
    "ROLLING_AWAY": (0, 165, 255),
}


class Ball:
    def __init__(self, track_id, pos):
        self.id = track_id
        self.positions = deque([np.array(pos, dtype=float)], maxlen=HISTORY_LENGTH)
        self.state = "STATIONARY"
        self.velocity = np.zeros(2)
        self.frames_stationary = 0

    def update(self, pos):
        self.positions.append(np.array(pos, dtype=float))
        if len(self.positions) > 5:
            self.velocity = (self.positions[-1] - self.positions[-5]) / 4.0
        elif len(self.positions) > 1:
            self.velocity = self.positions[-1] - self.positions[-2]
        self.frames_stationary = (
            self.frames_stationary + 1 if self.speed < STATIONARY_SPEED_THRESHOLD else 0
        )

    @property
    def speed(self):
        return float(np.linalg.norm(self.velocity))

    @property
    def pos(self):
        return self.positions[-1]

    @property
    def is_marker(self):
        return self.frames_stationary > MARKER_STATIONARY_FRAMES


def ankles(pose_result):
    """Every visible ankle across every detected player."""
    out = []
    kps = pose_result.keypoints
    if kps is None or kps.xy is None:
        return out
    xy = kps.xy.cpu().numpy()
    conf = kps.conf.cpu().numpy() if kps.conf is not None else None
    for p in range(xy.shape[0]):
        for idx in (L_ANKLE, R_ANKLE):
            if conf is None or conf[p, idx] > KP_CONF:
                out.append(xy[p, idx])
    return out


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "vdo" / "test.mp4"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "vdo" / "analysis_own_detector.mp4"

    detector = YOLO(str(BALL_WEIGHTS))
    poser = YOLO(str(POSE_WEIGHTS))

    cap = cv2.VideoCapture(str(src))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"{src.name}: {w}x{h} @ {fps}fps -> {dst.name}")

    balls, lost = {}, {}
    frame_i = 0
    stats = {"frames": 0, "detections": 0, "relinks": 0,
             "IN_POSSESSION": 0, "STATIONARY": 0, "ROLLING_AWAY": 0,
             "markers": 0, "players": 0}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        stats["frames"] += 1

        pose_res = poser.predict(frame, conf=0.4, device=DEVICE, verbose=False)[0]
        feet = ankles(pose_res)
        stats["players"] += len(pose_res.boxes)

        res = detector.track(frame, persist=True, conf=CONF, classes=[0],
                             tracker="botsort.yaml", device=DEVICE, verbose=False)[0]

        ids, boxes = [], {}
        if res.boxes.id is not None:
            ids = res.boxes.id.int().cpu().tolist()
            xyxy = res.boxes.xyxy.cpu().numpy()
            boxes = {t: b for t, b in zip(ids, xyxy)}
            stats["detections"] += len(ids)

            # stitch: a brand-new id appearing near a recently-lost one is the same ball
            for new_id in [i for i in ids if i not in balls]:
                x1, y1, x2, y2 = boxes[new_id]
                c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                best, best_d = None, REACQUIRE_THRESHOLD
                for lost_id, d in lost.items():
                    dist = np.linalg.norm(c - d["pos"])
                    if dist < best_d:
                        best, best_d = lost_id, dist
                if best is not None:
                    balls[best] = Ball(best, lost[best]["pos"])
                    del lost[best]
                    ids[ids.index(new_id)] = best
                    boxes[best] = boxes.pop(new_id)
                    stats["relinks"] += 1

            for t in ids:
                x1, y1, x2, y2 = boxes[t]
                c = ((x1 + x2) / 2, (y1 + y2) / 2)
                if t in balls:
                    balls[t].update(c)
                else:
                    balls[t] = Ball(t, c)

        # retire tracks that vanished this frame
        for gone in set(balls) - set(ids):
            lost[gone] = {"pos": balls[gone].pos, "frame": frame_i}
            del balls[gone]
        for lost_id in [k for k, v in lost.items() if frame_i - v["frame"] > MAX_LOST_FRAMES]:
            del lost[lost_id]

        # possession state
        for b in balls.values():
            if b.speed < STATIONARY_SPEED_THRESHOLD:
                b.state = "STATIONARY"
            elif feet and min(np.linalg.norm(b.pos - f) for f in feet) < POSSESSION_PROXIMITY_THRESHOLD:
                b.state = "IN_POSSESSION"
            else:
                b.state = "ROLLING_AWAY"
            stats[b.state] += 1
            if b.is_marker:
                stats["markers"] += 1

        # --- draw ---
        annotated = pose_res.plot(boxes=False)
        for f in feet:
            cv2.circle(annotated, (int(f[0]), int(f[1])), POSSESSION_PROXIMITY_THRESHOLD,
                       (255, 255, 0), 1, cv2.LINE_AA)
        for b in balls.values():
            p = (int(b.pos[0]), int(b.pos[1]))
            if b.is_marker:
                box = boxes.get(b.id)
                if box is not None:
                    x1, y1, x2, y2 = (int(v) for v in box)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"ID:{b.id} (marker)", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                col = STATE_COLORS[b.state]
                cv2.circle(annotated, p, 10, col, -1)
                cv2.putText(annotated, f"ID:{b.id} | {b.state} | {b.speed:.1f}px/f",
                            (p[0] + 15, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        cv2.putText(annotated, f"frame {frame_i}  balls {len(balls)}  players {len(pose_res.boxes)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(annotated)

    cap.release()
    writer.release()

    f = max(stats["frames"], 1)
    print(f"\n--- {stats['frames']} frames ---")
    print(f"ball detections   : {stats['detections']} ({stats['detections']/f:.2f}/frame)")
    print(f"player detections : {stats['players']} ({stats['players']/f:.2f}/frame)")
    print(f"track re-links    : {stats['relinks']}")
    tot = sum(stats[k] for k in ("IN_POSSESSION", "STATIONARY", "ROLLING_AWAY")) or 1
    for k in ("IN_POSSESSION", "STATIONARY", "ROLLING_AWAY"):
        print(f"{k:17} : {stats[k]:5d} ball-frames ({100*stats[k]/tot:.1f}%)")
    print(f"marker ball-frames: {stats['markers']}")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
