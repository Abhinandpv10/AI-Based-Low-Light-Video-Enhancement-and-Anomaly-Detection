import torch
import torchvision
import numpy as np
from PIL import Image
import model
import time

# ================= SETTINGS =================
IMAGE_PATH = "ZeroDCE\\test.jpg"
MODEL_PATH = "ZeroDCE\\zerodce.pth"
OUTPUT_PATH = "ZeroDCE\\enhanced_image.jpg"
SCALE_FACTOR = 12
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] Loading image...")
data_lowlight = Image.open(IMAGE_PATH)

data_lowlight = (np.asarray(data_lowlight) / 255.0)
data_lowlight = torch.from_numpy(data_lowlight).float()

h = (data_lowlight.shape[0] // SCALE_FACTOR) * SCALE_FACTOR
w = (data_lowlight.shape[1] // SCALE_FACTOR) * SCALE_FACTOR
data_lowlight = data_lowlight[0:h, 0:w, :]
data_lowlight = data_lowlight.permute(2, 0, 1)
data_lowlight = data_lowlight.unsqueeze(0).to(device)

print("[INFO] Loading Zero-DCE++ model...")
DCE_net = model.enhance_net_nopool(SCALE_FACTOR).to(device)
DCE_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
DCE_net.eval()

print("[INFO] Enhancing...")
start = time.time()

with torch.no_grad():
    enhanced_image, _ = DCE_net(data_lowlight)

print("Processing time:", time.time() - start)

torchvision.utils.save_image(enhanced_image, OUTPUT_PATH)

print("[INFO] Done! Saved as:", OUTPUT_PATH)