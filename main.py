import cv2
from ultralytics import YOLO

def main():
    # Load the YOLOv8 model
    model = YOLO('yolo11x.pt')  # Use yolov8n.pt, yolov8s.pt, etc., based on your needs
    
    cap = cv2.VideoCapture('./F1.mov')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform object detection
        results = model(frame)
        
        # Plot the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the frame
        cv2.imshow('YOLOv11 Object Detection', annotated_frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
