import cv2
import os
from ultralytics import YOLO

# Load YOLO models
model = YOLO('Files/yolov8n.pt')
model = YOLO('Files/best.pt')

# Define the path to the images folder
images_folder = r"Files/Images"

# Get a list of all image files in the folder
image_files = [
    os.path.join(images_folder, f)
    for f in os.listdir(images_folder)
    if os.path.isfile(os.path.join(images_folder, f)) and f.lower().endswith((".png", ".jpg", ".jpeg"))
]

# Loop through each image file
for image_path in image_files:
    # Read the image
    frame = cv2.imread(image_path)

    if frame is not None:

        # Run YOLOv8 tracking on the image
        results = model.track(frame, persist=True, show=True, tracker="botsort.yaml")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated image
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Wait for user input to proceed to the next image or quit
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

# Close the display window
cv2.destroyAllWindows()
