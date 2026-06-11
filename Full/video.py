import torch
import numpy as np
import cv2
import model
from ultralytics import YOLO
import time

# ================= CONFIG =================
VIDEO_PATH = "Full/test.mp4"
ZERO_DCE_MODEL = "ZeroDCE/zerodce.pth"
YOLO_MODEL = "Anomaly/anomaly.pt"

CONF_THRESHOLD = 0.4
SCALE_FACTOR = 12
INFER_SIZE = 320
FRAME_SKIP = 2
TARGET_HEIGHT = 250   # Final display height
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------- Load Zero-DCE --------
DCE_net = model.enhance_net_nopool(SCALE_FACTOR).to(device)
DCE_net.load_state_dict(torch.load(ZERO_DCE_MODEL, map_location=device))
DCE_net.eval()

# -------- Load YOLO --------
yolo_model = YOLO(YOLO_MODEL)
yolo_model.to(device)

# -------- Open Video --------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

frame_count = 0
last_boxes = None
fps_start = time.time()

# -------- Helper Functions --------
def enhance_frame(frame):
    h, w = frame.shape[:2]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    frame_tensor = torch.from_numpy(frame_rgb).float()

    h2 = (frame_tensor.shape[0] // SCALE_FACTOR) * SCALE_FACTOR
    w2 = (frame_tensor.shape[1] // SCALE_FACTOR) * SCALE_FACTOR
    frame_tensor = frame_tensor[0:h2, 0:w2, :]
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced, _ = DCE_net(frame_tensor)

    enhanced = enhanced.squeeze().permute(1, 2, 0).cpu().numpy()
    enhanced = np.clip(enhanced, 0, 1)
    enhanced = (enhanced * 255).astype(np.uint8)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
    enhanced = cv2.resize(enhanced, (w, h))

    return enhanced


def add_title(img, text):
    cv2.putText(img, text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2)
    return img


def resize_keep_aspect(img, target_h):
    h, w = img.shape[:2]
    scale = target_h / h
    return cv2.resize(img, (int(w * scale), target_h))


# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    original = frame.copy()

    # ---- Enhance ----
    enhanced = enhance_frame(frame)

    # ---- Detection (Frame Skip) ----
    if frame_count % FRAME_SKIP == 0:
        results = yolo_model(
            enhanced,
            conf=CONF_THRESHOLD,
            imgsz=INFER_SIZE,
            half=True if device.type == "cuda" else False,
            verbose=False
        )
        last_boxes = results[0].boxes

    detected = enhanced.copy()

    if last_boxes is not None:
        for box in last_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{yolo_model.names[cls]} {conf:.2f}"

            cv2.rectangle(detected, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(detected, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

    # ---- Add Titles ----
    original_labeled = add_title(original, "Original")
    enhanced_labeled = add_title(enhanced, "Zero-DCE Enhanced")
    detected_labeled = add_title(detected, "Enhanced + Detection")

    # ---- Resize Compact ----
    original_small = resize_keep_aspect(original_labeled, TARGET_HEIGHT)
    enhanced_small = resize_keep_aspect(enhanced_labeled, TARGET_HEIGHT)
    detected_small = resize_keep_aspect(detected_labeled, TARGET_HEIGHT)

    combined = np.hstack((original_small,
                          enhanced_small,
                          detected_small))

    # ---- FPS Display ----
    elapsed = time.time() - fps_start
    fps = frame_count / elapsed if elapsed > 0 else 0
    cv2.putText(combined, f"FPS: {fps:.2f}",
                (20, TARGET_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2)

    cv2.imshow("Low-Light Enhancement + Weapon Detection", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()