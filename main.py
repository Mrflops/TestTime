import cv2
import os
import time
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
        # Get image dimensions
        height, width, _ = frame.shape
        num_pixels = height * width

        # Start timing
        start_time = time.time()

        # Run YOLOv8 tracking on the image
        results = model.track(frame, persist=True, show=False, tracker="botsort.yaml")

        # Stop timing
        end_time = time.time()

        # Calculate processing time and time per pixel
        processing_time = end_time - start_time
        time_per_pixel = processing_time / num_pixels

        # Print results (rounded to 3 decimal places)
        print(f"Processed {image_path}:")
        print(f"  Image size: {width}x{height} ({num_pixels} pixels)")
        print(f"  Processing time: {processing_time:.3f} seconds")
        print(f"  Time per pixel: {time_per_pixel:.3e} seconds/pixel")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated image
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Wait a small amount of time for display (adjust or remove if needed)
        cv2.waitKey(1)

# Close the display window
cv2.destroyAllWindows()
