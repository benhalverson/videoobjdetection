from roboflow import Roboflow
from dotenv import load_dotenv
import os
import cv2

# Load environment variables
load_dotenv()
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace().project("rc-cars-xlkrl")
model = project.version('1').model

# Paths for input and output
video_path = "./bootcamp.mov"
output_path = "./bootcamp_result.mov"

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

MINUTES = 4
# Calculate the maximum frames to process for 2 minutes
max_frames = fps * MINUTES * 60

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f"Processing the first {MINUTES} minutes of video '{video_path}'...")

frame_idx = 0
while frame_idx < max_frames:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Perform inference on the frame
    predictions = model.predict(frame, confidence=10, overlap=30).json()

    # Draw predictions only if detections are present
    if predictions['predictions']:
        for prediction in predictions['predictions']:
            x, y, w, h = prediction['x'], prediction['y'], prediction['width'], prediction['height']
            label = prediction['class']
            confidence = prediction['confidence']

            # Bounding box coordinates
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Draw the bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write processed frame to the output video
    out.write(frame)

    frame_idx += 1
    if frame_idx % 10 == 0:  # Log every 10 frames
        print(f"Processed frame {frame_idx}/{max_frames}")

# Release resources
cap.release()
out.release()

print(f"Processing completed. Video saved to '{output_path}'")
