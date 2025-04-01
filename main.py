import cv2
from ultralytics import YOLO
from roboflow import Roboflow
from dotenv import load_dotenv
import os
def main():
    # Load the YOLOv8 model
    # model = YOLO('yolo11x.pt')  # Use yolov8n.pt, yolov8s.pt, etc., based on your needs
    
    load_dotenv()
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace().project("rc-cars-xlkrl")
    model = project.version('1').model
    cap = cv2.VideoCapture('./3lapseries.mp4')
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform object detection
        # results = model(frame)
        
        # Plot the results on the frame
        # annotated_frame = results[0].plot()
        
        # Display the frame

        predictions = model.predict(frame, confidence=10, overlap=30).json()
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

        cv2.imshow('custom Object Detection', frame)
         
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
