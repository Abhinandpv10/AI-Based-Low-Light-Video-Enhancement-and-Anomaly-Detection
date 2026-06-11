import cv2
from ultralytics import YOLO

# ================= CONFIG =================
MODEL_PATH = "Anomaly\\anomaly.pt"     # path to your trained model
IMAGE_PATH = "Anomaly\\test.jpg"    # image to test
CONF_THRESHOLD = 0.4
# ==========================================

# Load model
model = YOLO(MODEL_PATH)

# Run inference
results = model(IMAGE_PATH, conf=CONF_THRESHOLD)

# Get annotated frame
annotated_frame = results[0].plot()

# Show image
cv2.imshow("Anomaly Detection - Image", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: print detections
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    class_name = model.names[cls_id]
    print(f"Detected: {class_name} | Confidence: {conf:.2f}")