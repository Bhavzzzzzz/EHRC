import torch
import cv2
import numpy as np
import os
import glob
import torchstain
from torchvision import transforms

INPUT_DIR = os.path.expanduser("~/local_workspace/unnormalized_patches")
OUTPUT_DIR = os.path.expanduser("~/local_workspace/normalized_patches") 
REFERENCE_IMAGE = "./master_reference.png" # Must be a manually selected, perfect tissue patch

def normalize_dataset():
    # 1. SETUP COMPUTE DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Initialization ---")
    print(f"Using Compute Device: {DEVICE}")

    # 2. FIND ALL PATCHES
    all_patches = glob.glob(os.path.join(INPUT_DIR, "**", "*.png"), recursive=True)
    if not all_patches:
        print(f"ERROR: No patches found in: {INPUT_DIR}!")
        return
    print(f"Found {len(all_patches)} patches total to process.")

    # 3. LOAD THE GOLD STANDARD REFERENCE
    if not os.path.exists(REFERENCE_IMAGE):
        print(f"ERROR: Could not find {REFERENCE_IMAGE}. Please upload a good tissue patch.")
        return
        
    print(f"Loading Reference Image: {REFERENCE_IMAGE}")
    target = cv2.imread(REFERENCE_IMAGE)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    # 4. INITIALIZE GPU MACENKO NORMALIZER
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])
    
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    
    # Push the reference image to the GPU and calculate the color matrix
    normalizer.fit(T(target).to(DEVICE))
    
    count = 0
    errors = 0
    
    for i, patch_path in enumerate(all_patches):
        try:
            # Read image from hard drive
            img = cv2.imread(patch_path)
            if img is None: continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Push the raw patch to the GPU
            img_tensor = T(img_rgb).to(DEVICE)
            
            # Normalize (stains=False saves massive compute time)
            try:
                # We use torch.no_grad() so PyTorch doesn't waste RAM tracking gradients
                with torch.no_grad():
                    norm_tensor, _, _ = normalizer.normalize(I=img_tensor, stains=False)
            except Exception:
                errors += 1
                continue

            # Pull the normalized image back to the CPU and convert to OpenCV format
            norm_numpy = norm_tensor.cpu().numpy().astype(np.uint8)
            norm_bgr = cv2.cvtColor(norm_numpy, cv2.COLOR_RGB2BGR)
          
            # Calculate save path and save
            relative_path = os.path.relpath(patch_path, INPUT_DIR)
            save_path = os.path.join(OUTPUT_DIR, relative_path)
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, norm_bgr)
            
            count += 1
            if count % 100 == 0: 
                print(f"    Normalized {count}/{len(all_patches)} patches...", end='\r')
                
        except Exception as e:
            print(f"\n[Warning] Skipped {patch_path}: {e}")

    print(f"\n\nDone! {count} patches successfully normalized.")
    print(f"Skipped {errors} patches due to math errors (likely mostly white/empty).")

if __name__ == "__main__":
    normalize_dataset()
