import torch
import numpy as np
import cv2
import model
import time

# ================= SETTINGS =================
VIDEO_PATH = "ZeroDCE\\test.mp4"
MODEL_PATH = "ZeroDCE\\zerodce.pth"
OUTPUT_PATH = "ZeroDCE\\enhanced.mp4"
SCALE_FACTOR = 12
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] Loading Zero-DCE++ model...")
DCE_net = model.enhance_net_nopool(SCALE_FACTOR).to(device)
DCE_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
DCE_net.eval()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Error opening video file")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25  # fallback FPS

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[INFO] FPS: {fps}")
print(f"[INFO] Resolution: {width}x{height}")

# MP4 Codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

if not out.isOpened():
    print("❌ VideoWriter failed to open")
    exit()

print("[INFO] Processing video...")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original_frame = frame.copy()

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = frame_rgb / 255.0

    frame_tensor = torch.from_numpy(frame_rgb).float()

    # Make dimensions divisible by SCALE_FACTOR
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

    # Resize back to original size
    enhanced_frame = cv2.resize(enhanced_frame, (width, height))

    out.write(enhanced_frame)

    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames")

cap.release()
out.release()

print("===================================")
print("Total frames:", frame_count)
print("Total time:", time.time() - start_time)
print("Saved as:", OUTPUT_PATH)
print("===================================")