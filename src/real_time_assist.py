import cv2
import torch
import pyttsx3
import time
from collections import deque
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# -------------------------------
# Configuration
# -------------------------------
CAMERA_INDEX = 0
YOLO_MODEL_PATH = "../models/yolov8n.pt"   # Use nano/small model for CPU
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-large"
CAPTION_FREQUENCY = 50     # Generate caption every N frames
CONF_THRESHOLD = 0.4       # Minimum object confidence
CAPTION_SMOOTHING = 5      # Smooth last N captions
SPEECH_INTERVAL = 5        # Seconds between speaking
BEAM_WIDTH = 5             # Beam search width for BLIP captions
MAX_OBJECTS = 2            # Max objects to caption per frame
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480

# -------------------------------
# Device setup
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"🧠 Using device: {device.upper()} ({dtype})")

# -------------------------------
# Load YOLOv8
# -------------------------------
print("🔄 Loading YOLOv8 model...")
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model.to(device)
print("✅ YOLOv8 loaded!")

# -------------------------------
# Load BLIP
# -------------------------------
print("🔄 Loading BLIP captioning model...")
processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
blip_model = BlipForConditionalGeneration.from_pretrained(
    BLIP_MODEL_NAME, torch_dtype=dtype
).to(device)
blip_model.eval()
print(f"✅ BLIP loaded on {device.upper()}!")

# -------------------------------
# Text-to-Speech setup
# -------------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 165)
engine.setProperty("volume", 1.0)

def speak(text: str):
    """Speak text aloud."""
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass

# -------------------------------
# Helper functions
# -------------------------------
def preprocess_image(image_np):
    """Enhance image for better captioning."""
    image_np = cv2.convertScaleAbs(image_np, alpha=1.2, beta=10)
    image_np = cv2.GaussianBlur(image_np, (3, 3), 0)
    return image_np

def generate_caption(image_np):
    """Generate caption for a cropped object."""
    try:
        image_np = preprocess_image(image_np)
        image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        inputs = processor(image, return_tensors="pt").to(device, dtype=dtype)
        with torch.no_grad():
            out = blip_model.generate(
                **inputs,
                max_new_tokens=60,
                num_beams=BEAM_WIDTH,
                repetition_penalty=1.2
            )
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        return f"Error: {e}"

# -------------------------------
# Camera initialization
# -------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"❌ Could not open camera at index {CAMERA_INDEX}")
    exit()
print("✅ Camera opened. Press 'q' to quit.\n")

# -------------------------------
# Main loop
# -------------------------------
frame_count = 0
last_spoken_time = 0
recent_captions = deque(maxlen=CAPTION_SMOOTHING)
last_caption = ""

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    # Resize for faster CPU processing
    frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
    frame_count += 1

    # Run YOLO detection
    results = yolo_model(frame, verbose=False)
    boxes = results[0].boxes
    annotated = results[0].plot()

    # Caption every few frames
    if frame_count % CAPTION_FREQUENCY == 0 and boxes is not None and len(boxes) > 0:
        # Sort boxes by confidence
        detections = sorted(boxes, key=lambda b: b.conf[0], reverse=True)[:MAX_OBJECTS]
        captions = []
        for box in detections:
            if box.conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            caption = generate_caption(crop)
            obj_name = results[0].names[int(box.cls)]
            captions.append(f"{caption} (object: {obj_name})")

        if captions:
            final_caption = "; ".join(captions)
            recent_captions.append(final_caption)
            # Smooth captions
            caption_counts = {c: recent_captions.count(c) for c in recent_captions}
            smooth_caption = max(caption_counts, key=caption_counts.get)

            if smooth_caption != last_caption and (time.time() - last_spoken_time) > SPEECH_INTERVAL:
                last_caption = smooth_caption
                last_spoken_time = time.time()
                print("🗣️ Caption:", last_caption)
                speak(last_caption)

    # FPS & overlay
    fps = 1 / (time.time() - start_time + 1e-6)
    display_caption = last_caption[:70] + "..." if len(last_caption) > 70 else last_caption
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated, f"Caption: {display_caption}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("🔍 Real-Time Object Captioning (CPU-Friendly)", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exiting...")
