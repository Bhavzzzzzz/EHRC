import slideio
import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

def get_high_res_scene(slide):
    """Finds the scene with the highest resolution (the actual tissue scan)."""
    best_scene = None
    max_pixels = 0
    for i in range(slide.num_scenes):
        scene = slide.get_scene(i)
        pixels = scene.size[0] * scene.size[1]
        if pixels > max_pixels:
            max_pixels = pixels
            best_scene = scene
    return best_scene

def create_tissue_mask(scene, downsample_factor=32):
    """
    Creates a binary mask of the tissue using a downsampled version of the scene.
    """
    w_0, h_0 = scene.size
    
    # Calculate thumbnail size
    thumb_w = w_0 // downsample_factor
    thumb_h = h_0 // downsample_factor
    
    print(f"Generating tissue mask (Thumbnail size: {thumb_w}x{thumb_h})...")
    
    # Read the downsampled image
    # SlideIO reads a block: (x, y, width, height) and scales it to `size`
    thumb_np = scene.read_block((0, 0, w_0, h_0), size=(thumb_w, thumb_h))
    
    # Convert RGB to HSV
    hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
    _, saturation, _ = cv2.split(hsv)
    
    # Apply Otsu's thresholding
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up dust
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask, downsample_factor

def tile_vsi_image(vsi_path, output_dir, patch_size=224, tissue_threshold=0.4):
    slide_name = os.path.splitext(os.path.basename(vsi_path))[0]
    slide_output_dir = os.path.join(output_dir, slide_name)
    os.makedirs(slide_output_dir, exist_ok=True)
    
    try:
        # slideio explicitly uses the VSI driver
        slide = slideio.open_slide(vsi_path, "VSI")
    except Exception as e:
        print(f"Error opening {vsi_path}: {e}")
        return

    # 1. Get the actual tissue scan scene
    scene = get_high_res_scene(slide)
    if scene is None:
        print(f"No valid scenes found in {vsi_path}")
        return
        
    w_0, h_0 = scene.size
    print(f"Slide dimensions: {w_0} x {h_0}")

    # 2. Get the tissue mask
    mask, downsample_factor = create_tissue_mask(scene)
    
    print(f"Extracting patches to {slide_output_dir}...")

    # Calculate total patches (avoiding out-of-bounds at the edges)
    total_x = w_0 // patch_size
    total_y = h_0 // patch_size
    total_patches = total_x * total_y
    extracted_count = 0

    with tqdm(total=total_patches, desc=f"Processing {slide_name}") as pbar:
        # Stop before the edge to ensure full 224x224 patches
        for y in range(0, h_0 - patch_size + 1, patch_size):
            for x in range(0, w_0 - patch_size + 1, patch_size):
                pbar.update(1)
                
                # Map high-res coordinates to mask coordinates
                mask_x = int(x / downsample_factor)
                mask_y = int(y / downsample_factor)
                mask_w = max(1, int(patch_size / downsample_factor))
                mask_h = max(1, int(patch_size / downsample_factor))
                
                # Boundary check for the mask
                if mask_y >= mask.shape[0] or mask_x >= mask.shape[1]:
                    continue
                
                # Check tissue percentage
                mask_region = mask[mask_y : mask_y + mask_h, mask_x : mask_x + mask_w]
                tissue_ratio = np.count_nonzero(mask_region) / (mask_region.size + 1e-6)
                
                if tissue_ratio >= tissue_threshold:
                    # Extract high-res patch directly as a numpy array
                    patch_np = scene.read_block((x, y, patch_size, patch_size), size=(patch_size, patch_size))
                    
                    # SlideIO returns numpy arrays; convert to PIL to save
                    patch = Image.fromarray(patch_np)
                    patch_filename = f"{slide_name}_{x}_{y}.png"
                    patch.save(os.path.join(slide_output_dir, patch_filename))
                    extracted_count += 1

    print(f"Finished {slide_name}. Extracted {extracted_count} tissue patches.\n")

if __name__ == "__main__":
    INPUT_DIR = "raw_data"              
    OUTPUT_DIR = "unnormalized_patches" 
    PATCH_SIZE = 224                    
    TISSUE_THRESHOLD = 0.4              
    
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    vsi_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith('.vsi'):
                full_path = os.path.join(root, file)
                vsi_files.append(full_path)
    
    if not vsi_files:
        print(f"No .vsi files found in '{INPUT_DIR}' or its subfolders.")
    else:
        print(f"Found {len(vsi_files)} .vsi files. Starting extraction...")
        for vsi_path in vsi_files:
            tile_vsi_image(vsi_path, OUTPUT_DIR, patch_size=PATCH_SIZE, tissue_threshold=TISSUE_THRESHOLD)
