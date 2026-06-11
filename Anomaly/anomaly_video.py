import cv2
import torch
from ultralytics import YOLO

MODEL_PATH = "Anomaly/anomaly.pt"
VIDEO_PATH = "Anomaly/test.mp4"

CONF_THRESHOLD = 0.4
INFER_WIDTH = 416
DISPLAY_WIDTH = 960
INFER_SIZE = 320
FRAME_SKIP = 3   # run detection every 3 frames

device = 0 if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = YOLO(MODEL_PATH)
model.to(device)

cap = cv2.VideoCapture(VIDEO_PATH)

frame_count = 0
last_boxes = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    h, w = frame.shape[:2]

    # Resize for inference
    scale_infer = INFER_WIDTH / w
    infer_frame = cv2.resize(frame, (INFER_WIDTH, int(h * scale_infer)))

    # Run detection every N frames
    if frame_count % FRAME_SKIP == 0:
        results = model(
            infer_frame,
            conf=CONF_THRESHOLD,
            imgsz=INFER_SIZE,
            half=True if device == 0 else False,
            verbose=False
        )
        last_boxes = results[0].boxes

    # Resize original frame for display
    scale_display = DISPLAY_WIDTH / w
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, int(h * scale_display)))

    # Draw last detected boxes
    if last_boxes is not None:
        scale_ratio = DISPLAY_WIDTH / INFER_WIDTH

        for box in last_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 = int(x1 * scale_ratio)
            y1 = int(y1 * scale_ratio)
            x2 = int(x2 * scale_ratio)
            y2 = int(y2 * scale_ratio)

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{model.names[cls_id]} {conf:.2f}"

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(display_frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Live Knife Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()