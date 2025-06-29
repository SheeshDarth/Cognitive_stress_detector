import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import tensorflow as tf
import datetime
from playsound import playsound
import time

# ========== Load Models ==========
print("[INFO] Loading models...")
emotion_model = load_model(r"C:\Users\Siddharth\cognitive-fatigue-detector\models\emotion_model_cnn.keras")
fatigue_model = load_model(r"C:\Users\Siddharth\cognitive-fatigue-detector\models\daisee_fatigue_cnn.keras")
drowsy_model = load_model(r"C:\Users\Siddharth\cognitive-fatigue-detector\models\nthu_drowsy_cnn.keras")
yawn_model = load_model(r"C:\Users\Siddharth\cognitive-fatigue-detector\models\yawdd_model.keras")
yolo_model = YOLO(r"C:\Users\Siddharth\cognitive-fatigue-detector\runs\detect\train5\weights\best.pt")
print("[INFO] Models loaded.")

emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
fatigue_labels = ['Low', 'Medium', 'High']
drowsy_labels = ['Alert', 'Drowsy']  # 0: eyes open, 1: eyes closed
yawn_labels = ['Normal', 'Talking', 'Yawning']

# ========== Webcam Selection ==========
# Set this to the index found with the camera test code above
CAMERA_INDEX = 0  # Change to 1, 2, etc. if needed for your phone webcam

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open webcam at index {CAMERA_INDEX}.")
print(f"[INFO] Webcam {CAMERA_INDEX} opened.")

log_file = open("stress_events.csv", "a")
if log_file.tell() == 0:
    log_file.write("Timestamp,Emotion,Fatigue,Drowsy,Yawn\n")

cv2.namedWindow("Cognitive Fatigue Detector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Cognitive Fatigue Detector", 900, 700)

frame_skip = 2  # Only run inference every N frames
frame_count = 0
last_results = []
eyes_closed_start_time = None
alarm_played = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            continue

        # Mirror the frame for natural webcam feel
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        frame_count += 1

        # Only run inference every N frames
        if frame_count % frame_skip == 0:
            last_results = []
            infer_frame = cv2.resize(frame, (416, 416))
            yolo_results = yolo_model.predict(infer_frame, verbose=False)[0]
            boxes = yolo_results.boxes.xyxy.cpu().numpy()
            scale_x = frame.shape[1] / 416
            scale_y = frame.shape[0] / 416
            boxes = np.array([
                [int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)]
                for x1, y1, x2, y2 in boxes
            ])
            classes = yolo_results.boxes.cls.cpu().numpy()

            eyes_closed_detected = False

            for box, cls_id in zip(boxes, classes):
                x1, y1, x2, y2 = box
                # Ensure box is within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0 or (x2-x1) < 10 or (y2-y1) < 10:
                    continue

                try:
                    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    face_gray_resized = cv2.resize(face_gray, (48,48))/255.0
                    face_gray_resized = np.expand_dims(face_gray_resized, (-1,0))
                    face_rgb_resized = cv2.resize(face_crop, (48,48))/255.0
                    face_rgb_resized = np.expand_dims(face_rgb_resized,0)
                except Exception as e:
                    print("[ERROR] Face preprocessing error:", e)
                    continue

                emotion_pred = fatigue_pred = drowsy_pred = yawn_pred = -1
                try:
                    emotion_pred = emotion_model.predict(face_gray_resized,verbose=0).argmax()
                    fatigue_pred = fatigue_model.predict(face_gray_resized,verbose=0).argmax()
                    drowsy_pred = drowsy_model.predict(face_gray_resized,verbose=0).argmax()
                    yawn_pred = yawn_model.predict(face_rgb_resized,verbose=0).argmax()
                except Exception as e:
                    print("[ERROR] Prediction error:", e)

                # Only set eyes_closed_detected if drowsy_pred is valid
                if drowsy_pred == 1:
                    eyes_closed_detected = True

                is_stress = (
                    fatigue_pred == 2 or
                    drowsy_pred == 1 or
                    yawn_pred == 2 or
                    emotion_pred in [0,3,6]
                )

                if is_stress:
                    timestamp = datetime.datetime.now().isoformat()
                    log_file.write(
                        f"{timestamp},{emotion_labels[emotion_pred]},{fatigue_labels[fatigue_pred]},{drowsy_labels[drowsy_pred]},{yawn_labels[yawn_pred]}\n"
                    )
                    log_file.flush()

                last_results.append({
                    "box": (x1, y1, x2, y2),
                    "is_stress": is_stress,
                    "emotion": emotion_pred,
                    "fatigue": fatigue_pred,
                    "drowsy": drowsy_pred,
                    "yawn": yawn_pred
                })

            # Timestamp-based alarm
            if eyes_closed_detected:
                if eyes_closed_start_time is None:
                    eyes_closed_start_time = time.time()
            else:
                eyes_closed_start_time = None
                alarm_played = False

            if eyes_closed_start_time:
                closed_duration = time.time() - eyes_closed_start_time
                if closed_duration >= 2.0 and not alarm_played:
                    try:
                        playsound(r"C:\Users\Siddharth\cognitive-fatigue-detector\japan-eas-alarm-j-alert-262887.wav")
                        alarm_played = True
                    except Exception as e:
                        print("[WARN] Could not play sound:", e)

        # Draw
        for res in last_results:
            x1, y1, x2, y2 = res["box"]
            color = (0,0,255) if res["is_stress"] else (0,255,0)
            cv2.rectangle(display_frame,(x1,y1),(x2,y2),color,2)
            label_text = (
                f"E:{emotion_labels[res['emotion']]} "
                f"F:{fatigue_labels[res['fatigue']]} "
                f"D:{drowsy_labels[res['drowsy']]} "
                f"Y:{yawn_labels[res['yawn']]}"
            )
            (tw,th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
            label_y = max(y1 - th - 8,0)
            cv2.rectangle(display_frame,(x1,label_y),(x1+tw+8,label_y+th+12),color,-1)
            cv2.putText(display_frame,label_text,(x1+4,label_y+th+4),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        cv2.imshow("Cognitive Fatigue Detector", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    log_file.close()
    cv2.destroyAllWindows()
    print("[INFO] Webcam and windows closed.")