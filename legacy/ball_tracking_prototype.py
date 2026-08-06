from ultralytics import YOLO
import cv2
from collections import defaultdict, deque
import numpy as np

# --- MAXIMUM ACCURACY CONFIGURATION ---
MODEL_NAME = 'yolov8x.pt'          # Upgraded to the most accurate model
VIDEO_PATH = 'test.mp4'           # Or 'test.mp4'
CONFIDENCE_THRESHOLD = 0.3         # Stricter confidence for a more accurate model
MOVEMENT_THRESHOLD = 10            # Pixels
HISTORY_LENGTH = 900               # Longer history for more stable movement calculation

# The class ID for "sports ball" in the COCO dataset is 32
SPORTS_BALL_CLASS_ID = 32

# --- INITIALIZATION ---
model = YOLO(MODEL_NAME)

# Use a deque for efficient, fixed-length history
ball_positions = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))

# Open video file and prepare for writing output
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = "final_output_max_accuracy1000.mp4"
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))


# --- MAIN PROCESSING LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. DETECTION & TRACKING (Upgraded)
    # Use the BoT-SORT tracker for higher accuracy
    results = model.track(
        frame,
        persist=True,
        conf=CONFIDENCE_THRESHOLD,
        classes=[SPORTS_BALL_CLASS_ID],
        tracker='botsort.yaml'  # Specify the tracker
    )

    annotated_frame = results[0].plot() # Use the built-in plotting for boxes and IDs!

    # Check if any tracks exist in the current frame
    if results[0].boxes.id is not None:
        
        # Extract track IDs and bounding boxes
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu()

        action_ball_id = -1

        # 2. MOVEMENT ANALYSIS
        for i, track_id in enumerate(track_ids):
            # Calculate centroid
            x1, y1, x2, y2 = boxes[i]
            centroid_x = (x1 + x2) / 2
            centroid_y = (y1 + y2) / 2
            
            # Add current centroid to history using the deque
            ball_positions[track_id].append((centroid_x, centroid_y))
            
            # Check for movement if we have enough history
            if len(ball_positions[track_id]) == HISTORY_LENGTH:
                start_point = np.array(ball_positions[track_id][0])
                end_point = np.array(ball_positions[track_id][-1])
                movement = np.linalg.norm(end_point - start_point)

                if movement > MOVEMENT_THRESHOLD:
                    action_ball_id = track_id
                    break # Assume only one action ball
        
        # 3. ANNOTATE ACTION BALL
        # Draw a special marker on the action ball
        if action_ball_id != -1:
            try:
                # Find the latest position of the action ball
                action_ball_index = track_ids.index(action_ball_id)
                x1, y1, x2, y2 = boxes[action_ball_index]
                center_x, center_y = int((x1+x2)/2), int((y1+y2)/2)
                
                # Draw a red circle with a white outline for better visibility
                cv2.circle(annotated_frame, (center_x, center_y), 7, (0, 0, 255), -1)
                cv2.circle(annotated_frame, (center_x, center_y), 8, (255, 255, 255), 1)
            except ValueError:
                # This handles a rare case where the action ball might be lost for a frame
                pass


    # Display and write the frame
    cv2.imshow('Flickit CV - Maximum Accuracy', annotated_frame)
    writer.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- CLEANUP ---
cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Processing complete. Annotated video saved to {output_path}")