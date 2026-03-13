import os
import subprocess
import cv2
import numpy as np
import tifffile
import glob
import re
# --- CONFIGURATION ---
INPUT_ROOT = "/mnt/c/Users/Bhavya Jain/Downloads/H&E"
OUTPUT_DIR = "./processed_dataset"
BF_TOOLS_DIR = os.path.expanduser("~/EHRC/tools/bftools") 
BFCONVERT_PATH = os.path.join(BF_TOOLS_DIR, "bfconvert")
SHOWINF_PATH = os.path.join(BF_TOOLS_DIR, "showinf")
PATCH_SIZE = 224

def get_large_series_list(file_path):
    """
    Returns a LIST of all series indices that are 'High Res' (Width > 10,000).
    Example Output: ['9', '10', '11']
    """
    cmd = [SHOWINF_PATH, "-nopix", file_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    valid_series = []
    current_series = "0"
    
    for line in result.stdout.splitlines():
        # Detect Series Header
        series_match = re.search(r"Series #(\d+)", line)
        if series_match:
            current_series = series_match.group(1)
            
        # Detect Dimensions
        if "Width =" in line:
            try:
                width = int(line.split("=")[1].strip())
                # THRESHOLD: Any image wider than 5,000 pixels is likely tissue
                # (Thumbnails are usually < 2,000)
                if width > 5000:
                    print(f"    -> Found High-Res Series #{current_series} (Width: {width}px)")
                    valid_series.append(current_series)
            except:
                pass
                
    return list(set(valid_series)) # Remove duplicates


def estimate_magnification(image, patch_size=1024):
    """
    Estimates if image is 40x or 20x based on nuclei size.
    """
    h, w, _ = image.shape
    
    # Crop center for speed
    cy, cx = h // 2, w // 2
    crop = image[cy-patch_size//2 : cy+patch_size//2, cx-patch_size//2 : cx+patch_size//2]
    
    if crop.size == 0: return 20 

    # Image Processing (Grayscale -> Blur -> Otsu)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find Nuclei
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nuclei_sizes = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 5000: # Filter noise
            _, _, w_box, h_box = cv2.boundingRect(cnt)
            nuclei_sizes.append(max(w_box, h_box))
            
    if not nuclei_sizes:
        print("    [Auto-Detect] No nuclei found. Defaulting to 20x.")
        return 20 # Safe default
        
    avg_diameter = np.median(nuclei_sizes)
    print(f"    [Auto-Detect] Avg Nucleus Size: {avg_diameter:.1f} pixels")
    
    # The thresholds you asked about!
    if avg_diameter > 28:
        print("    -> Detected 40x. Will downsample.")
        return 40
    elif avg_diameter > 14:
        print("    -> Detected 20x. Keeping original.")
        return 20
    else:
        print("    -> Detected 10x (Low Res).")
        return 10

# --- THE HEAVY LIFTING (From the new pipeline) ---
def process_dataset():
    vsi_files = glob.glob(os.path.join(INPUT_ROOT, "**", "*.vsi"), recursive=True)
    print(f"Found {len(vsi_files)} slides.")

    for vsi_path in vsi_files:
        slide_base_name = os.path.basename(vsi_path).split('.')[0]
        
        # 1. FIND ALL VALID SERIES
        print(f"\nScanning {slide_base_name} for high-res images...")
        target_series_list = get_large_series_list(vsi_path)
        
        if not target_series_list:
            print("    [Warning] No large images found in this VSI.")
            continue

        # 2. PROCESS EACH SERIES SEPARATELY
        for series_idx in target_series_list:
            # We add the series number to the name so they don't overwrite each other
            # e.g., "0001_S9", "0001_S10"
            slide_name = f"{slide_base_name}_S{series_idx}"
            temp_tif = f"./temp_{slide_name}.ome.tif"
            
            # Skip if output folder already exists
            slide_out_dir = os.path.join(OUTPUT_DIR, slide_name)
            if os.path.exists(slide_out_dir) and len(os.listdir(slide_out_dir)) > 50:
                print(f"    Skipping {slide_name} (Already done).")
                continue

            print(f"  === Processing Series {series_idx} as {slide_name} ===")
            
            # CONVERT
            if not os.path.exists(temp_tif):
                print("    Converting...")
                cmd = [BFCONVERT_PATH, "-overwrite", "-compression", "LZW", 
                       "-series", series_idx, vsi_path, temp_tif]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            # TILE
            if os.path.exists(temp_tif):
                try:
                    tile_slide(temp_tif, slide_name)
                    os.remove(temp_tif)
                except Exception as e:
                    print(f"    [Error] Tiling Failed: {e}")


def tile_slide(tif_path, slide_name):
    save_dir = os.path.join(OUTPUT_DIR, slide_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with tifffile.TiffFile(tif_path) as tif:
        image = tif.asarray() 
        
        # 1. Fix Dimensions & Colors
        if image.ndim == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[-1] == 4: image = image[:,:,:3]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # OpenCV likes BGR

        # 2. Setup Tiling
        h, w, _ = image.shape
        source_mag = estimate_magnification(image)
        target_mag = 20
        scale_factor = source_mag / target_mag
        read_size = int(PATCH_SIZE * scale_factor)
        
        print(f"    Tiling {source_mag}x -> {target_mag}x (Step: {read_size})")

        count = 0
        for y in range(0, h, read_size):
            for x in range(0, w, read_size):
                if y + read_size > h or x + read_size > w: continue
                
                patch = image[y:y+read_size, x:x+read_size]
                
                # --- STRONGER FILTERING LOGIC ---
                
                # Filter 1: Background Percentage
                # Convert to grayscale
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                
                # Threshold: Anything brighter than 210 is "Background"
                # (Standard glass is usually > 220)
                _, binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
                
                # Count white pixels
                white_pixels = cv2.countNonZero(binary)
                total_pixels = patch.shape[0] * patch.shape[1]
                white_ratio = white_pixels / total_pixels
                
                # Rule: If more than 50% of the patch is white background -> DROP IT
                if white_ratio > 0.50:
                    continue

                # Filter 2: Edge/Texture Check (Canny)
                # Tissue has texture (edges). Glass is smooth.
                edges = cv2.Canny(gray, 50, 150)
                if np.count_nonzero(edges) < 100: # Less than 100 edge pixels = Empty
                    continue
                
                # -------------------------------

                # Resize and Save
                if read_size != PATCH_SIZE:
                    patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
                
                cv2.imwrite(f"{save_dir}/x{x}_y{y}.png", patch)
                count += 1
                if count % 100 == 0: print(f"    Saved {count} patches...", end='\r')

        print(f"\n    Finished: {count} clean patches saved.")

if __name__ == "__main__":
    process_dataset()
