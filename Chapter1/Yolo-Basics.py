from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('../Yolo-Weights/yolov8m.pt')

# Load and resize the image
image_path = "Images/5.png"
original_image = cv2.imread(image_path)
resized_image = cv2.resize(original_image, (800, 600))  # Adjust the dimensions as needed

# Perform inference and display the resized image
results = model(resized_image, show=True)

# Wait for a key event and close the OpenCV window
cv2.waitKey(0)
cv2.destroyAllWindows()
