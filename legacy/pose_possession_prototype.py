from ultralytics import YOLO
import cv2
from collections import defaultdict, deque
import numpy as np

# --- CONFIGURATION ---
MODEL_NAME = 'yolov8x.pt'
VIDEO_PATH = 'test.mp4'  # Or 'test2.mp4'
CONFIDENCE_THRESHOLD = 0.3
HISTORY_LENGTH = 30

# --- NEW STATE-BASED LOGIC THRESHOLDS ---
STATIONARY_SPEED_THRESHOLD = 2 # Speed (pixels/frame) below which a ball is considered stationary
MOVING_SPEED_THRESHOLD = 5     # Speed above which a ball is considered moving

# The class ID for "sports ball" in the COCO dataset is 32
SPORTS_BALL_CLASS_ID = 32

# --- NEW BALL CLASS TO STORE STATE ---
class Ball:
    def __init__(self, track_id):
        self.id = track_id
        self.positions = deque(maxlen=HISTORY_LENGTH)
        self.state = 'STATIONARY' # Initial state
        self.speed = 0

    def update_position(self, centroid):
        self.positions.append(centroid)
        if len(self.positions) > 1:
            # Calculate speed based on the last two positions
            self.speed = np.linalg.norm(np.array(self.positions[-1]) - np.array(self.positions[-2]))

    def update_state(self):
        if self.speed > MOVING_SPEED_THRESHOLD:
            self.state = 'MOVING'
        elif self.speed < STATIONARY_SPEED_THRESHOLD:
            self.state = 'STATIONARY'
        # Note: If speed is between the two thresholds, the state remains unchanged, providing stability.

    @property
    def last_known_position(self):
        return self.positions[-1] if self.positions else None

# --- INITIALIZATION ---
model = YOLO(MODEL_NAME)
tracked_balls = {} # Dictionary to store our Ball objects, keyed by track_id

cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = "state_based_output.mp4"
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))


# --- MAIN PROCESSING LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. DETECTION & TRACKING
    results = model.track(
        frame,
        persist=True,
        conf=CONFIDENCE_THRESHOLD,
        classes=[SPORTS_BALL_CLASS_ID],
        tracker='botsort.yaml',
        verbose=False # Quieter output
    )

    annotated_frame = frame.copy() # We will do all annotations manually

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        current_frame_ball_ids = set()

        # 2. UPDATE BALL STATES
        for i, track_id in enumerate(track_ids):
            current_frame_ball_ids.add(track_id)
            if track_id not in tracked_balls:
                tracked_balls[track_id] = Ball(track_id)

            x1, y1, x2, y2 = boxes[i]
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            # Update the ball's position and state
            ball = tracked_balls[track_id]
            ball.update_position(centroid)
            ball.update_state()

        # Clean up old tracks that have disappeared
        for track_id in list(tracked_balls.keys()):
            if track_id not in current_frame_ball_ids:
                del tracked_balls[track_id]

        # 3. ANNOTATE FRAME BASED ON STATE
        for i, track_id in enumerate(track_ids):
            ball = tracked_balls.get(track_id)
            if ball:
                x1, y1, x2, y2 = map(int, boxes[i])
                
                if ball.state == 'MOVING':
                    # Draw a red dot for the action ball
                    cv2.circle(annotated_frame, ball.last_known_position, 7, (0, 0, 255), -1)
                    cv2.circle(annotated_frame, ball.last_known_position, 8, (255, 255, 255), 1)
                elif ball.state == 'STATIONARY':
                    # Draw a green box and ID for stationary balls
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display and write the frame
    cv2.imshow('Flickit CV - State-Based Logic', annotated_frame)
    writer.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- CLEANUP ---
cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Processing complete. Annotated video saved to {output_path}")