import torch
import numpy as np
import cv2
import model
from ultralytics import YOLO
import time

# ================= CONFIG =================
VIDEO_PATH = "ZeroDCE/test.mp4"  # or 0 for webcam
ZERO_DCE_MODEL = "ZeroDCE/zerodce.pth"
YOLO_MODEL = "Anomaly/anomaly.pt"

CONF_THRESHOLD = 0.4
SCALE_FACTOR = 12

INFER_SIZE = 320        # small for 4GB GPU
FRAME_SKIP = 2
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------- Load Zero-DCE --------
print("[INFO] Loading Zero-DCE...")
DCE_net = model.enhance_net_nopool(SCALE_FACTOR).to(device)
DCE_net.load_state_dict(torch.load(ZERO_DCE_MODEL, map_location=device))
DCE_net.eval()

# -------- Load YOLO --------
print("[INFO] Loading YOLO...")
yolo_model = YOLO(YOLO_MODEL)
yolo_model.to(device)

# -------- Open Video --------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

frame_count = 0
last_boxes = None

print("[INFO] Starting pipeline...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    original_h, original_w = frame.shape[:2]

    # ===============================
    # STEP 1: Enhance Frame (Zero-DCE)
    # ===============================
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    frame_tensor = torch.from_numpy(frame_rgb).float()

    h = (frame_tensor.shape[0] // SCALE_FACTOR) * SCALE_FACTOR
    w = (frame_tensor.shape[1] // SCALE_FACTOR) * SCALE_FACTOR
    frame_tensor = frame_tensor[0:h, 0:w, :]
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced_frame, _ = DCE_net(frame_tensor)

    enhanced_frame = enhanced_frame.squeeze().permute(1, 2, 0).cpu().numpy()
    enhanced_frame = np.clip(enhanced_frame, 0, 1)
    enhanced_frame = (enhanced_frame * 255).astype(np.uint8)
    enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)

    enhanced_frame = cv2.resize(enhanced_frame, (original_w, original_h))

    # ===============================
    # STEP 2: Run Detection (Skip Frames)
    # ===============================
    if frame_count % FRAME_SKIP == 0:
        results = yolo_model(
            enhanced_frame,
            conf=CONF_THRESHOLD,
            imgsz=INFER_SIZE,
            half=True if device.type == "cuda" else False,
            verbose=False
        )
        last_boxes = results[0].boxes

    # ===============================
    # STEP 3: Draw Detection
    # ===============================
    if last_boxes is not None:
        for box in last_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{yolo_model.names[cls_id]} {conf:.2f}"

            cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(enhanced_frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

    cv2.imshow("Zero-DCE + Knife Detection", enhanced_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Finished.")