import torch
import numpy as np
import cv2
import model
from ultralytics import YOLO

# ================= CONFIG =================
IMAGE_PATH = "Full\image.png"
ZERO_DCE_MODEL = "ZeroDCE/zerodce.pth"
YOLO_MODEL = "Anomaly/anomaly.pt"

CONF_THRESHOLD = 0.4
SCALE_FACTOR = 12
INFER_SIZE = 320
TARGET_HEIGHT = 300   # Final display height (change 300–500)
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

# -------- Read Image --------
original = cv2.imread(IMAGE_PATH)
if original is None:
    print("❌ Cannot read image")
    exit()

h, w = original.shape[:2]

# ===============================
# STEP 1: Zero-DCE Enhancement
# ===============================
frame_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB) / 255.0
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

# ===============================
# STEP 2: Detection on Enhanced
# ===============================
detected = enhanced.copy()

results = yolo_model(
    detected,
    conf=CONF_THRESHOLD,
    imgsz=INFER_SIZE,
    half=True if device.type == "cuda" else False,
    verbose=False
)

boxes = results[0].boxes
if boxes is not None:
    for box in boxes:
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

# ===============================
# STEP 3: Add Titles
# ===============================
def add_title(img, text):
    cv2.putText(img, text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2)
    return img

original_labeled = add_title(original.copy(), "Original")
enhanced_labeled = add_title(enhanced.copy(), "Zero-DCE Enhanced")
detected_labeled = add_title(detected.copy(), "Enhanced + Detection")

# ===============================
# STEP 4: Resize for Compact View
# ===============================
def resize_keep_aspect(img, target_h):
    h, w = img.shape[:2]
    scale = target_h / h
    return cv2.resize(img, (int(w * scale), target_h))

original_small = resize_keep_aspect(original_labeled, TARGET_HEIGHT)
enhanced_small = resize_keep_aspect(enhanced_labeled, TARGET_HEIGHT)
detected_small = resize_keep_aspect(detected_labeled, TARGET_HEIGHT)

combined = np.hstack((original_small,
                      enhanced_small,
                      detected_small))

# ===============================
# STEP 5: Show Result
# ===============================
cv2.imshow("Low-Light Enhancement + Anomaly Detection", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()