import cv2
from ultralytics import YOLO
import os

def main():
    # Load the YOLOv8 model
    try:
        model = YOLO('yolov8n.pt')  # Choose the appropriate model, e.g., yolov8n.pt, yolov8s.pt
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return


    # Specify the image path
    image_path = './rc_car_dataset/images/image_0000.jpg'
    if not os.path.isfile(image_path):
        print(f"Error: Image path '{image_path}' does not exist.")
        return

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    try:
        # Perform object detection
        results = model(image)

        # Extract and plot the annotated image
        annotated_image = results[0].plot()
        
        # Display the annotated image
        cv2.imshow('YOLOv8 Object Detection', annotated_image)
        cv2.waitKey(0)  # Wait until a key is pressed

        # Save the annotated image
        output_path = 'annotated_image.jpg'
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved as '{output_path}'")

    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting...")

    finally:
        # Ensure all windows are closed, even if interrupted
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
