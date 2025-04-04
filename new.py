import cv2
import numpy as np
import pyttsx3  # Offline TTS
import sys
import pandas as pd
from ultralytics import YOLO
import tensorflow as tf

# Set file paths
yolo_model_path = "best.pt"
recognition_model_path = "modelnet2.keras"
video_path = "Traffic2.mp4"
output_video_path = "output_video.mp4"
csv_path = "labels.csv"


# Load YOLO detection model
yolo_model = YOLO(yolo_model_path)

# Load Keras recognition model
recognition_model = tf.keras.models.load_model(recognition_model_path)

# Load sign names from CSV file
try:
    sign_data = pd.read_csv(csv_path)
    sign_names = {row["ClassId"]: row["Name"] for _, row in sign_data.iterrows()}
except Exception as e:
    print(f"Error loading CSV file: {e}")
    sign_names = {}

# Initialize Text-to-Speech engine
tts_engine = pyttsx3.init()

# Initialize video processing
cap = cv2.VideoCapture(video_path)

# ✅ **Check if video is loaded correctly**
if not cap.isOpened():
    print(f"❌ Error: Could not open video file {video_path}")
    sys.exit(1)  # Exit if video file isn't found

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Voice-over toggle
voice_enabled = True

# Load sound icons (handle missing files)
try:
    sound_on_icon = cv2.imread("sound on.jpg")
    sound_off_icon = cv2.imread("sound off.jpg")
    icon_size = (50, 50)  # Resize icons
    sound_on_icon = cv2.resize(sound_on_icon, icon_size) if sound_on_icon is not None else None
    sound_off_icon = cv2.resize(sound_off_icon, icon_size) if sound_off_icon is not None else None
except:
    sound_on_icon = sound_off_icon = None

# Toggle sound function
def toggle_sound(event, x, y, flags, param):
    global voice_enabled
    if event == cv2.EVENT_LBUTTONDOWN:
        if 20 < x < 70 and 20 < y < 70:
            voice_enabled = not voice_enabled

# ✅ **Preprocess Image for Recognition**
def preprocess_for_recognition(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Create window and set callback
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", toggle_sound)

# Track previous sign to avoid repeating the same announcement
previous_sign_class = None
announcement_cooldown = 0
cooldown_frames = 30

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame (Video may have ended)")
            break

        if announcement_cooldown > 0:
            announcement_cooldown -= 1

        results = yolo_model(frame, conf=0.7, iou=0.5)

        for result in results:
            if not result.boxes:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()

                if conf > 0.5:
                    sign_img = frame[y1:y2, x1:x2]
                    if sign_img.size > 0 and sign_img.shape[0] > 10 and sign_img.shape[1] > 10:
                        try:
                            preprocessed_img = preprocess_for_recognition(sign_img)
                            prediction = recognition_model.predict(preprocessed_img, verbose=0)
                            max_confidence = np.max(prediction[0])
                            class_id = np.argmax(prediction[0])

                            if max_confidence > 0.7:
                                sign_class = sign_names.get(class_id, f"Unknown Sign ({class_id})")
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"{sign_class}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                                if voice_enabled and (previous_sign_class != sign_class or announcement_cooldown == 0):
                                    tts_engine.say(sign_class)
                                    tts_engine.runAndWait()
                                    previous_sign_class = sign_class
                                    announcement_cooldown = cooldown_frames

                        except Exception as e:
                            print(f"❌ Error processing sign: {e}")
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(frame, "Traffic Sign", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if sound_on_icon is not None and sound_off_icon is not None:
            icon = sound_on_icon if voice_enabled else sound_off_icon
            x_offset, y_offset = 20, 20
            if y_offset + icon_size[1] <= frame.shape[0] and x_offset + icon_size[0] <= frame.shape[1]:
                frame[y_offset:y_offset+icon_size[1], x_offset:x_offset+icon_size[0]] = icon

        cv2.imshow("Video", frame)
        out.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
            break

except Exception as e:
    print(f"❌ Error occurred: {e}")
finally:
    print("🧹 Cleaning up resources...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    try:
        tts_engine.stop()
    except:
        pass
    print("🚀 Program terminated")
