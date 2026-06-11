import torch
import numpy as np
import cv2
import model
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db

# ================= CONFIG =================
VIDEO_PATH = "rtsp://admin:L26D33A2@10.253.225.25:554/cam/realmonitor?channel=1&subtype=1"
ZERO_DCE_MODEL = "ZeroDCE/zerodce.pth"
YOLO_MODEL = "Anomaly/anomaly.pt"

SERVICE_ACCOUNT_PATH = "ServiceAccountKey.json"
DATABASE_URL = "https://darksightai-35def-default-rtdb.firebaseio.com/"

CONF_THRESHOLD = 0.4
SCALE_FACTOR = 12
INFER_SIZE = 320
FRAME_SKIP = 2
# ==========================================


# ================= FIREBASE INIT =================
cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred, {
    'databaseURL': DATABASE_URL
})

detect_ref = db.reference("/detect")
detect_ref.set(0)

print("[Firebase] Connected")


# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ================= LOAD MODELS =================
print("[INFO] Loading Zero-DCE...")
DCE_net = model.enhance_net_nopool(SCALE_FACTOR).to(device)
DCE_net.load_state_dict(torch.load(ZERO_DCE_MODEL, map_location=device))
DCE_net.eval()

print("[INFO] Loading YOLO...")
yolo_model = YOLO(YOLO_MODEL)
yolo_model.to(device)


# ================= RTSP CAPTURE =================
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

if not cap.isOpened():
    print("❌ Cannot connect to RTSP stream")
    exit()

frame_count = 0
last_boxes = None
previous_state = 0

print("[SYSTEM] RTSP Monitoring Started... Press Q to exit")


# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠ Stream lost... reconnecting")
        cap.release()
        cap = cv2.VideoCapture(VIDEO_PATH)
        continue

    frame_count += 1
    h, w = frame.shape[:2]

    # -------- Zero-DCE Enhancement --------
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

    # -------- YOLO Detection --------
    if frame_count % FRAME_SKIP == 0:
        results = yolo_model(
            enhanced,
            conf=CONF_THRESHOLD,
            imgsz=INFER_SIZE,
            half=True if device.type == "cuda" else False,
            verbose=False
        )
        last_boxes = results[0].boxes

    detected_frame = enhanced.copy()
    knife_detected = 0

    if last_boxes is not None:
        for box in last_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{yolo_model.names[cls_id]} {conf:.2f}"

            cv2.rectangle(detected_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(detected_frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)

            knife_detected = 1

    # -------- Firebase Update (Only if State Changes) --------
    if knife_detected != previous_state:
        detect_ref.set(knife_detected)
        previous_state = knife_detected

    # -------- Alert Overlay --------
    if knife_detected == 1:
        cv2.putText(detected_frame, "ALERT: KNIFE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

    # -------- Final Preview Only --------
    cv2.imshow("DarkSightAI - RTSP Monitoring", detected_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
print("System Stopped.")