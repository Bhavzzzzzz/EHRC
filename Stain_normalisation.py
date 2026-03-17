import torch
import cv2
import numpy as np
import os
import glob
import torchstain
from torchvision import transforms

INPUT_DIR = "./processed_dataset"    
OUTPUT_DIR = "./normalized_dataset"  
BATCH_SIZE = 100

def normalize_dataset():
    # 1. FIND A REFERENCE IMAGE
    # We grab the very first patch we can find to serve as the "Standard"
    all_patches = glob.glob(os.path.join(INPUT_DIR, "**", "*.png"), recursive=True)
    
    if not all_patches:
        print("ERROR: No patches found in processed_dataset!")
        return

    print(f"Found {len(all_patches)} patches total.")
    
    # Pick the first one as reference
    ref_path = all_patches[0]
    print(f"Using Reference Image: {ref_path}")
    
    target = cv2.imread(ref_path)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    # 2. SETUP THE NORMALIZER (Macenko Method)
    # We use backend='torch' for speed. It works on CPU too.
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])
    
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    
    # "Fit" the normalizer to our reference image
    # (This calculates the specific Purple/Pink vectors of the reference)
    normalizer.fit(T(target))

    # 3. PROCESS EVERYTHING
    print("Starting normalization...")
    
    count = 0
    errors = 0
    
    for i, patch_path in enumerate(all_patches):
        try:
            # Read image
            img = cv2.imread(patch_path)
            if img is None: continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize
            try:
                norm, H, E = normalizer.normalize(I=T(img_rgb), stains=True)
            except Exception:
                
                errors += 1
                continue

            
            norm_numpy = norm.numpy().astype(np.uint8)
            norm_bgr = cv2.cvtColor(norm_numpy, cv2.COLOR_RGB2BGR)
          
            relative_path = os.path.relpath(patch_path, INPUT_DIR)
            save_path = os.path.join(OUTPUT_DIR, relative_path)
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, norm_bgr)
            
            count += 1
            if count % 100 == 0: 
                print(f"Normalized {count}/{len(all_patches)} patches...", end='\r')
                
        except Exception as e:
            print(f"\nError on {patch_path}: {e}")

    print(f"\n\nDone! {count} patches normalized.")
    print(f"Skipped {errors} patches (likely empty/white).")

if __name__ == "__main__":
    normalize_dataset()
