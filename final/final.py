import cv2
import cvzone
import math
import pyttsx3
from ultralytics import YOLO

# Initialize the text-to-speech engine
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)

# Set the desired width and height for display
display_width = 1280  # Full  width
display_height = 720  # Full  height

# Set the desired width and height for video capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

model = YOLO("../Yolo-Weights/yolov8m.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()

    # Resize the image for display
    img_display = cv2.resize(img, (display_width, display_height))

    results = model(img_display, stream=True)
    recognized_objects = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img_display, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])

            object_name = classNames[cls]
            recognized_objects.append(object_name)

            cvzone.putTextRect(img_display, f'{object_name} {conf}', (max(0, x1), max(35, y1)))

    cv2.imshow("Image", img_display)
    cv2.waitKey(1)

    # Convert recognized objects into audio
    if recognized_objects:
        text = "I see " + ", ".join(recognized_objects)
        engine.say(text)
        engine.runAndWait()
