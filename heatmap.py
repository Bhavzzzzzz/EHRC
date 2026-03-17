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

    # Extract coords
    name = os.path.basename(p)
    x = int(name.split("_")[0][1:])
    y = int(name.split("_")[1].split(".")[0][1:])
    coords.append((x, y))

patches = torch.stack(patches).to(device)

# Get attention maps
with torch.no_grad():
    outputs, attentions = model(patches, return_attention=True)

# Load original image
orig = cv2.imread(RAW_IMAGE)
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

biomarkers = ["IDH1R132H", "ATRX", "P53"]

plt.figure(figsize=(15,5))

for i, (att, name) in enumerate(zip(attentions, biomarkers)):

    att = att.squeeze().cpu().numpy()

    # Normalize
    att = (att - np.min(att)) / (np.max(att) - np.min(att) + 1e-8)

    heatmap = np.zeros((orig.shape[0], orig.shape[1]))

    for (x, y), val in zip(coords, att):
        heatmap[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = val

    # Smooth heatmap
    heatmap = cv2.GaussianBlur(heatmap, (51,51), 0)

    # Resize
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))

    # Color map
    heatmap_color = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = (0.7 * orig + 0.6 * heatmap_color)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Plot
    plt.subplot(1,3,i+1)
    plt.title(name)
    plt.imshow(overlay)
    plt.axis('off')

plt.suptitle("Biomarker-wise ROI Heatmaps")
plt.show()