# Blind Aid System

A project designed to assist visually impaired individuals by detecting objects and people in their surroundings. The system converts visual information into audio cues, enabling a better understanding of their environment.

---

## Features

- **Image Classification:** Detects and classifies objects in static images.
- **Video Processing:** Processes video files to identify objects in motion.
- **Real-Time Detection:** Utilizes live video feed for object recognition and provides audio feedback.

---

## Folder Structure

1. **Chapter 1: Image Classification**  
   Contains datasets, trained models, and scripts for performing object classification on static images.

2. **Chapter 2: Video Processing**  
   Includes tools and models for object detection and tracking in video files.

3. **Final: Live Video Capture and Detection**  
   Implements real-time object detection, converting visual data into audio for immediate feedback.

---

## Technology Stack

- **YOLOv8**: Used for object detection and recognition.
- **Roboflow**: For custom dataset creation and annotation.
- **Python**: Core programming language for the system.
- **PyTorch**: Framework used for training and deploying the deep learning model.
- **Google Colab**: Platform for model training with GPU acceleration.

---

## How to Run

1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/blind-aid-system.git
