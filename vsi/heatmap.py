import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import os
import glob
import re

# --- CONFIGURATION ---
SLIDE_NAME = "0002"
PATCH_DIR = f"./normalized_patches/{SLIDE_NAME}"
MODEL_WEIGHTS = "./best_mil_model.pth"
PATCH_SIZE = 224
BATCH_SIZE = 256 # Massive speed boost for 61k patches
SCALE = 0.1 # Scales the final heatmap canvas down to 10% to save RAM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. THE MIL MODEL
# ==========================================
class MILModel_ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_IDH1 = self._make_attention()
        self.attention_ATRX = self._make_attention()
        self.attention_P53 = self._make_attention()
        self.classifier = nn.Linear(2048, 3)

    def _make_attention(self):
        return nn.Sequential(
            nn.Linear(2048, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, feats, return_attention=False):
        A_idh1 = torch.softmax(self.attention_IDH1(feats), dim=0)
        A_atrx = torch.softmax(self.attention_ATRX(feats), dim=0)
        A_p53  = torch.softmax(self.attention_P53(feats), dim=0)

        M_idh1 = torch.sum(A_idh1 * feats, dim=0)
        M_atrx = torch.sum(A_atrx * feats, dim=0)
        M_p53  = torch.sum(A_p53 * feats, dim=0)

        M = torch.stack([M_idh1, M_atrx, M_p53], dim=0)
        out = self.classifier(M.mean(dim=0))

        if return_attention:
            return out, [A_idh1, A_atrx, A_p53]
        return out

# ==========================================
# 2. THE BATCHING DATASET
# ==========================================
class HeatmapDataset(Dataset):
    def __init__(self, patch_files, transform):
        self.transform = transform
        self.valid_data = []

        # Parse valid files and coordinates instantly
        for f in patch_files:
            # FIX: Matches your format like "0001_1568_45024.png"
            match = re.search(r'\d+_(\d+)_(\d+)\.png', os.path.basename(f))
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                self.valid_data.append((f, x, y))

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        file_path, x, y = self.valid_data[idx]
        img = Image.open(file_path).convert('RGB')
        tensor = self.transform(img)
        return tensor, x, y

# ==========================================
# 3. THE GENERATOR
# ==========================================
def generate_heatmap():
    print(f"Generating Heatmaps for {SLIDE_NAME}...")
    
    # Setup Models
    model = MILModel_ResNet50().to(DEVICE)
    if os.path.exists(MODEL_WEIGHTS):
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
        model.eval()
        print("Successfully loaded trained weights.")
    else:
        print("[WARNING] No weights found! Visualizing untrained random attention.")

    encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    encoder.fc = nn.Identity()
    encoder = encoder.to(DEVICE)
    encoder.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    patch_files = glob.glob(os.path.join(PATCH_DIR, "*.png"))
    if not patch_files:
        print(f"Error: No patches found in {PATCH_DIR}")
        return

    dataset = HeatmapDataset(patch_files, transform)
    # Using multiple workers to feed the GPU as fast as possible
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    all_features = []
    all_x = []
    all_y = []

    print(f"Extracting features for {len(dataset)} patches using batch size {BATCH_SIZE}...")
    with torch.no_grad():
        for i, (tensors, xs, ys) in enumerate(dataloader):
            tensors = tensors.to(DEVICE)
            feats = encoder(tensors)
            
            all_features.append(feats.cpu())
            all_x.extend(xs.numpy())
            all_y.extend(ys.numpy())
            
            if i % 5 == 0:
                print(f"  Processed {min((i+1)*BATCH_SIZE, len(dataset))}/{len(dataset)}...", end='\r')

    # Stack all features
    print("\nCalculating Attention Weights...")
    slide_features = torch.cat(all_features, dim=0).to(DEVICE)

    with torch.no_grad():
        preds, attentions = model(slide_features, return_attention=True)
        
    # Example: Generating IDH1 Heatmap (Index 0)
    attn_idh1 = attentions[0].cpu().numpy().flatten()
    
    # Normalize attention scores to 0-1 range
    attn_idh1 = (attn_idh1 - attn_idh1.min()) / (attn_idh1.max() - attn_idh1.min())

    # Build the Canvas
    max_x = max(all_x) + PATCH_SIZE
    max_y = max(all_y) + PATCH_SIZE
    
    canvas_h, canvas_w = int(max_y * SCALE), int(max_x * SCALE)
    heatmap_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    print("Painting the thermal canvas...")
    for i in range(len(all_x)):
        sx, sy = int(all_x[i] * SCALE), int(all_y[i] * SCALE)
        spatch = max(1, int(PATCH_SIZE * SCALE))
        
        heatmap_canvas[sy:sy+spatch, sx:sx+spatch] = attn_idh1[i]

    # Apply JET Colormap
    heatmap_uint8 = np.uint8(255 * heatmap_canvas)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    save_file = f"heatmap_IDH1_{SLIDE_NAME}.png"
    cv2.imwrite(save_file, colored_heatmap)
    print(f"🎉 SUCCESS! Saved full-slide attention map to {save_file}")

if __name__ == "__main__":
    generate_heatmap()
