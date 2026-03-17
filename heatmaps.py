import torch
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from model import MILModel

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "model.pth"
SLIDE_NAME = "slide1"   # change this

PATCH_DIR = f"normalized_dataset/{SLIDE_NAME}"
RAW_IMAGE = f"labelled_png/{SLIDE_NAME}.png"

PATCH_SIZE = 224

# Load model
model = MILModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Load patches
patch_paths = glob.glob(f"{PATCH_DIR}/*.png")

patches = []
coords = []

for p in patch_paths:
    img = cv2.imread(p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor = torch.tensor(img).permute(2,0,1).float()/255.0
    patches.append(tensor)

    # extract coordinates from filename
    name = os.path.basename(p)
    x = int(name.split("_")[0][1:])
    y = int(name.split("_")[1].split(".")[0][1:])
    coords.append((x, y))

patches = torch.stack(patches).to(device)

# Get attention
with torch.no_grad():
    outputs, attention = model(patches, return_attention=True)

attention = attention.squeeze().cpu().numpy()

# Normalize attention
attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

# Load original image
orig = cv2.imread(RAW_IMAGE)
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

heatmap = np.zeros((orig.shape[0], orig.shape[1]))

# Place attention values back
for (x, y), att in zip(coords, attention):
    heatmap[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = att

# Resize heatmap if needed
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))

# Convert to color
heatmap_color = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

# Overlay
overlay = (0.6 * orig + 0.4 * heatmap_color).astype(np.uint8)

# Show
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(orig)

plt.subplot(1,2,2)
plt.title("Attention Heatmap (ROI)")
plt.imshow(overlay)

plt.show()