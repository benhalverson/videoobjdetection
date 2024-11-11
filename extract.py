import cv2
import os

def extract_frames(video_path: str, output_folder: str, frame_interval=30):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Initialize variables
    frame_count = 0
    frame_number = 0
    
    while cap.isOpened():
        # Read a frame
        ret, frame = cap.read()
        
        # Break the loop if the video has ended
        if not ret:
            break
        
        # Increment the frame count
        frame_count += 1
        
        # Skip frames if needed
        if frame_count % frame_interval != 0:
            continue
        
        # Save the frame
        frame_path = os.path.join(output_folder, f"image_{frame_number:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        # Increment the frame number
        frame_number += 1
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Extract frames from the video
extract_frames('./shortVideo.mov', './images' )