import os
import cv2
import numpy as np
import glob

#need to download some python libraries before runnnig this code 
#pip install openslide-python opencv-python numpy torch torchvision pandas h5py matplotlib scikit-learn (copy paste into terminal to download required libraries)
#sudo apt-get install openslide-tools
def estimate_magnification(image, patch_size=512):
    """
    Estimates magnification (20x or 40x) by measuring average nuclei size.
    
    Logic:
    - At 40x (0.25 MPP), a nucleus (approx 8-10um) is ~30-40 pixels wide.
    - At 20x (0.50 MPP), a nucleus is ~15-20 pixels wide.
    """
    h, w, _ = image.shape
    
    # Take a center crop to analyze (faster than checking whole image)
    cy, cx = h // 2, w // 2
    crop = image[cy-patch_size//2 : cy+patch_size//2, cx-patch_size//2 : cx+patch_size//2]
    
    if crop.size == 0: return 20 # Fallback
    
    # 1. Convert to Grayscale & Blur
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # 2. Thresholding (Otsu) to find dark nuclei
    # In H&E, nuclei are dark purple. We want to separate them from pink/white.
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Find Contours (Nuclei candidates)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    nuclei_sizes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter noise: Nuclei are usually between 100 and 2000 pixels depending on mag
        if 50 < area < 5000:
            # Get bounding box width
            _, _, w_box, h_box = cv2.boundingRect(cnt)
            nuclei_sizes.append(max(w_box, h_box)) # Use largest dimension
            
    if not nuclei_sizes:
        print("    [Warning] No nuclei found in check region. Defaulting to 20x.")
        return 20
        
    avg_diameter = np.median(nuclei_sizes)
    
    print(f"    [Auto-Detect] Avg Nucleus Size: {avg_diameter:.1f} pixels")
    
    # 4. Heuristic Decision
    if avg_diameter > 28:
        print("    -> Estimated Mag: 40x")
        return 40
    elif avg_diameter > 14:
        print("    -> Estimated Mag: 20x")
        return 20
    else:
        print("    -> Estimated Mag: 10x")
        return 10

def tile_standard_image(image_path, output_folder, target_mag=20, patch_size=224):
    filename = os.path.basename(image_path).split('.')[0]
    save_dir = os.path.join(output_folder, filename)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Processing: {filename}...")

    img = cv2.imread(image_path)
    if img is None:
        return

    # --- AUTO DETECT MAGNIFICATION ---
    source_mag = estimate_magnification(img)
    # ---------------------------------

    img_h, img_w, _ = img.shape
    scale_factor = source_mag / target_mag
    read_size = int(patch_size * scale_factor)

    print(f"  - Input: {source_mag}x | Target: {target_mag}x | Step: {read_size}px")

    count = 0
    for y in range(0, img_h, read_size):
        for x in range(0, img_w, read_size):
            if y + read_size > img_h or x + read_size > img_w:
                continue

            patch = img[y:y+read_size, x:x+read_size]
            
            # Tissue Check (Saturation)
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            if np.mean(hsv[:, :, 1]) > 20: 
                if read_size != patch_size:
                    patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
                
                cv2.imwrite(f"{save_dir}/x{x}_y{y}.png", patch)
                count += 1

    print(f"  - Saved {count} patches.")

if __name__ == "__main__":
    input_folder = "./images"
    output_folder = "./patches"
    
    if not os.path.exists(input_folder):
        print("Please create an 'images' folder and put your PNGs inside.")
    else:
        png_files = glob.glob(os.path.join(input_folder, "*.png"))
        for f in png_files:
            tile_standard_image(f, output_folder, target_mag=20)
