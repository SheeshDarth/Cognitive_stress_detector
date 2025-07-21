# Real-Time Cognitive Fatigue Detector

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A real-time, multi-model system built with Python and PyTorch to detect cognitive fatigue by analyzing a user's facial expressions, eye state, and behavior via a webcam feed.

The application uses a two-stage pipeline: it first detects the user's face and then runs a suite of specialized deep learning models on the cropped facial region to calculate a real-time fatigue score. The system is optimized with multithreading and frame throttling to ensure smooth performance.

## Features

-   **Multi-Model Analysis:** Aggregates predictions from five different models for a robust assessment.
-   **Real-Time Face Detection:** Uses a Haar Cascade classifier to locate the face, ensuring classifiers focus only on relevant data.
-   **Fatigue Scoring:** Implements a custom scoring algorithm to convert model outputs into an actionable fatigue level.
-   **Live Alerts:** Provides visual on-screen alerts and an audible beep when the fatigue score crosses a set threshold.
-   **Performance Optimized:** Employs multithreading to prevent camera lag and frame throttling to manage computational load.

## Technology Stack

-   **Language:** Python
-   **Core Libraries:** PyTorch, OpenCV, Ultralytics (YOLOv8)
-   **Models:** ResNet-18, VGG-16, Custom CNN, YOLOv8n

---

## Installation & Setup

Follow these steps to get the project running on your local machine.

### 1. Prerequisites

-   Git
-   Python 3.9 or higher
-   An NVIDIA GPU with CUDA installed is highly recommended for real-time performance.

### 2. Clone the Repository

```bash
git clone [https://github.com/YourUsername/cognitive-fatigue-detector.git](https://github.com/YourUsername/cognitive-fatigue-detector.git)
cd cognitive-fatigue-detector
```

### 3. Set Up a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
```

### 4. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Download Model Files

The model weights and classifier files are not included in this repository due to their size. You must download them and place them in the `models` folder.

| Model File                          | Purpose                           | Download Link                                                                               |
| ----------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------- |
| `yolov8n.pt`                        | YOLOv8 Nano Object Detection      | https://drive.google.com/drive/folders/1jqGzJG9pUxzR75q5A0-FTrgRnN55Nxmm?usp=drive_link     |                                                              |
| `resnet18_daisee.pt`                | DAISEE Dataset Classifier         |  all the files are present in the above folder.                                             |
| `yawdd_model.pth`                   | Yawning Detection Classifier      |                                                                                             |
| `emotion_model_cnn.pth`             | Emotion Recognition Classifier    |                                                                                             |
| `nthu_drowsy_cnn.pth`               | NTHU-DDD Drowsiness Classifier    |                                                                                             |
| `haarcascade_frontalface_default.xml`| OpenCV Face Detector             | [Download from OpenCV GitHub](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml) |

Your final folder structure should look like this:

```
cognitive-fatigue-detector/
├── models/
│   ├── emotion_model_cnn.pth
│   ├── haarcascade_frontalface_default.xml
│   ├── nthu_drowsy_cnn.pth
│   ├── resnet18_daisee.pt
│   ├── yawdd_model.pth
│   └── yolov8n.pt
│
├── src/
│   └── main.py
│
├── README.md
└── requirements.txt
```

---

## How to Use

Once the setup is complete, run the main application from the root directory of the project:

```bash
python src/main.py
```

-   A window will appear showing your webcam feed.
-   A green rectangle will be drawn around your detected face.
-   Bounding boxes from YOLO may appear over certain objects or behaviors.
-   The fatigue score will be displayed in the top-left corner.
-   If the score exceeds the threshold, an "ALERT" message will appear, and a beep will sound.
-   Press the **'q'** key on your keyboard to close the application.

---

## How It Works

The application follows a two-stage, throttled pipeline for efficiency:

1.  **Face Detection:** A lightweight Haar Cascade classifier runs on every frame to find the user's face.
2.  **Throttled Inference:** Periodically (e.g., every 15 frames), the following occurs:
    * The detected face region is cropped from the main camera frame.
    * This small `face_crop` is sent to the four classifier models (DAISEE, YawDD, FER, NTHU) for analysis.
    * Simultaneously, the full camera frame is sent to the YOLOv8 model to detect behaviors like yawning.
3.  **Scoring:** The outputs from all models are collected and passed to a `compute_fatigue_score` function, which calculates the final score.
4.  **Display:** The results are overlaid on the video feed for the user to see.

This approach ensures the video remains smooth while the computationally expensive analysis happens in the background at a reasonable interval.
