import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import winsound
from ultralytics import YOLO
from torchvision import models
from PIL import Image
import threading
from queue import Queue

# --- DEFINE THE CUSTOM CNN FOR EMOTION RECOGNITION ---
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        # Input: 48x48 -> Pool -> 24x24 -> Pool -> 12x12 -> Pool -> 6x6
        # 128 channels * 6 * 6 = 4608
        self.fc1 = nn.Linear(128 * 6 * 6, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- THREAD-SAFE VIDEO CAPTURE CLASS ---
class VideoCapture:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.q = Queue()
        self.stopped = False
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                return
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # Discard old frame to prevent lag
                except Queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.stopped = True
        self.cap.release()

# --- SETUP AND PATHS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

DAISEE_MODEL_PATH = "C:/Users/Siddharth/cognitive-fatigue-detector/models/resnet18_daisee.pt"
YOLO_MODEL_PATH = "C:/Users/Siddharth/cognitive-fatigue-detector/models/yolov8_fatigue.pt"
YAWDD_MODEL_PATH = r"C:\Users\Siddharth\cognitive-fatigue-detector\models\yawdd_model.pth"
FERPLUS_MODEL_PATH = r"C:\Users\Siddharth\cognitive-fatigue-detector\models\emotion_model_cnn.pth"
NTHU_MODEL_PATH = r"C:\Users\Siddharth\cognitive-fatigue-detector\models\nthu_drowsy_cnn.pth"

# --- MODEL LOADING ---
# Load DAISEE Model
daisee_model = models.resnet18(weights=None)
daisee_model.fc = nn.Linear(daisee_model.fc.in_features, 4)
daisee_model.load_state_dict(torch.load(DAISEE_MODEL_PATH, map_location=device, weights_only=True))
daisee_model.to(device).eval()

# Load YawDD Model
yawdd_model = models.resnet18(weights=None)
yawdd_model.fc = nn.Linear(yawdd_model.fc.in_features, 3)
yawdd_model.load_state_dict(torch.load(YAWDD_MODEL_PATH, map_location=device, weights_only=True))
yawdd_model.to(device).eval()

# Load FERPlus Model
FERPLUS_LOADED = False
try:
    ferplus_model = EmotionCNN().to(device)
    ferplus_model.load_state_dict(torch.load(FERPLUS_MODEL_PATH, map_location=device, weights_only=True))
    ferplus_model.eval()
    FERPLUS_LOADED = True
    print("[INFO] FERPlus model loaded successfully.")
except Exception as e:
    print(f"[WARNING] FERPlus (EmotionCNN) model could not be loaded. Skipping. Error: {e}")

# Load NTHU-DDD Model
NTHU_LOADED = False
try:
    nthu_model = models.resnet18(weights=None).to(device)
    nthu_model.fc = nn.Linear(nthu_model.fc.in_features, 2)
    nthu_model.load_state_dict(torch.load(NTHU_MODEL_PATH, map_location=device, weights_only=True))
    nthu_model.eval()
    NTHU_LOADED = True
    print("[INFO] NTHU-DDD model loaded successfully.")
except Exception as e:
    print(f"[WARNING] NTHU-DDD model not found or architecture mismatch. Skipping. Error: {e}")

# Load YOLO Model
yolo_model = YOLO(YOLO_MODEL_PATH)

# --- IMAGE TRANSFORMS ---
transform_rgb = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_gray = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# --- SCORING LOGIC AND MAPPINGS ---
FERPLUS_EMOTIONS = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disgust', 6: 'fear', 7: 'contempt'}
FATIGUE_EMOTIONS = {'sadness', 'anger', 'contempt', 'neutral'}

def compute_fatigue_score(outputs):
    score, total_sources = 0, 0
    if "daisee" in outputs:
        total_sources += 1
        if outputs["daisee"] in [2, 3]: score += 1
    if "yawdd" in outputs:
        total_sources += 1
        if outputs["yawdd"] in [1, 2]: score += 1
    if "yolo" in outputs:
        total_sources += 1
        if outputs["yolo"]: score += 1
    if FERPLUS_LOADED and "ferplus" in outputs:
        total_sources += 1
        if FERPLUS_EMOTIONS.get(outputs["ferplus"]) in FATIGUE_EMOTIONS: score += 1
    if NTHU_LOADED and "nthu" in outputs:
        total_sources += 1
        if outputs["nthu"] == 1: score += 1
    return score, total_sources

# --- MAIN APPLICATION LOOP ---
cap = VideoCapture(0)
print("[INFO] Live Fatigue Monitor started. Press 'q' to quit.")

# Variables for throttling and persistent display
frame_counter = 0
inference_interval = 5  # Run inference every 5 frames (~6 FPS)
outputs = {}
yolo_results = None

while True:
    frame = cap.read()
    frame_counter += 1

    # --- THROTTLED INFERENCE BLOCK ---
    # Only run the heavy models periodically
    if frame_counter % inference_interval == 0:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor_rgb = transform_rgb(frame_rgb).unsqueeze(0).to(device)
            if FERPLUS_LOADED:
                input_tensor_gray = transform_gray(frame_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs["daisee"] = torch.argmax(daisee_model(input_tensor_rgb), dim=1).item()
                outputs["yawdd"] = torch.argmax(yawdd_model(input_tensor_rgb), dim=1).item()
                if NTHU_LOADED:
                    outputs["nthu"] = torch.argmax(nthu_model(input_tensor_rgb), dim=1).item()
                if FERPLUS_LOADED:
                    outputs["ferplus"] = torch.argmax(ferplus_model(input_tensor_gray), dim=1).item()

            yolo_results_new = yolo_model(frame, verbose=False)[0]
            outputs["yolo"] = any(
                yolo_model.names[int(box.cls.item())] in ['yawn', 'drowsy', 'distraction', 'phone_usage']
                for box in yolo_results_new.boxes
            )
            yolo_results = yolo_results_new # Update display results
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")

    # --- DISPLAY LOGIC (runs every frame) ---
    score, total_sources = compute_fatigue_score(outputs)
    frame_display = yolo_results.plot() if yolo_results else frame.copy()
    
    is_fatigued = score >= (total_sources / 2) and total_sources > 0
    color = (0, 0, 255) if is_fatigued else (0, 255, 0)
    
    cv2.putText(frame_display, f"Fatigue Score: {score}/{total_sources}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if is_fatigued:
        cv2.putText(frame_display, "ALERT: Fatigue Detected!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if frame_counter % inference_interval == 0: # Beep only on inference frames
            winsound.Beep(2000, 500)

    cv2.imshow("Live Fatigue Monitor", frame_display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- CLEANUP ---
cap.stop()
cv2.destroyAllWindows()
print("[INFO] Application terminated.")
