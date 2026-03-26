import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import glob

# --- CONFIGURATION ---
INPUT_DIR = "./normalized_patches"
OUTPUT_DIR = "./extracted_features"
BATCH_SIZE = 128 # Adjust based on your GPU VRAM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PatchDataset(Dataset):
    def __init__(self, patch_paths, transform):
        self.patch_paths = patch_paths
        self.transform = transform

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        img = Image.open(self.patch_paths[idx]).convert('RGB')
        return self.transform(img)

def extract_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading ResNet50...")
    # Load ResNet50 and remove the final classification head
    encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    encoder.fc = nn.Identity()
    encoder = encoder.to(DEVICE)
    encoder.eval()

    # Standard ImageNet transforms required by ResNet
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    slide_folders = [f.path for f in os.scandir(INPUT_DIR) if f.is_dir()]
    print(f"Found {len(slide_folders)} slides to extract.")

    with torch.no_grad(): # No gradients needed for extraction!
        for slide_dir in slide_folders:
            slide_name = os.path.basename(slide_dir)
            save_path = os.path.join(OUTPUT_DIR, f"{slide_name}.pt")
            
            if os.path.exists(save_path):
                print(f"Skipping {slide_name}, already extracted.")
                continue
                
            patch_paths = glob.glob(os.path.join(slide_dir, "*.png"))
            if not patch_paths: continue
            
            print(f"Extracting {len(patch_paths)} patches for {slide_name}...")
            
            dataset = PatchDataset(patch_paths, transform)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            
            slide_features = []
            
            for batch in loader:
                batch = batch.to(DEVICE)
                feats = encoder(batch) # Outputs [Batch, 2048]
                slide_features.append(feats.cpu())
                
            # Stack all batches into one massive tensor [N, 2048]
            final_tensor = torch.cat(slide_features, dim=0)
            torch.save(final_tensor, save_path)
            
    print("\nFeature extraction complete!")

if __name__ == "__main__":
    extract_all()
