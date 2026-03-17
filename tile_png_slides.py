import os
import cv2
import numpy as np

INPUT_DIR = "./labelled_png"          # your raw slides
OUTPUT_DIR = "./normalized_dataset"  # final patch dataset
PATCH_SIZE = 224

def is_tissue(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # Relaxed background filter
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    white_ratio = cv2.countNonZero(binary) / (PATCH_SIZE * PATCH_SIZE)

    if white_ratio > 0.7:   # was 0.5 → now relaxed
        return False

    # Relaxed edge check
    edges = cv2.Canny(gray, 30, 100)
    if np.count_nonzero(edges) < 30:   # was 100 → now relaxed
        return False

    return True


def tile_slide(slide_path, slide_name):
    img = cv2.imread(slide_path)
    if img is None:
        print(f"Error reading {slide_name}")
        return

    h, w, _ = img.shape

    save_dir = os.path.join(OUTPUT_DIR, slide_name)
    os.makedirs(save_dir, exist_ok=True)

    count = 0

    for y in range(0, h, PATCH_SIZE):
        for x in range(0, w, PATCH_SIZE):

            if y + PATCH_SIZE > h or x + PATCH_SIZE > w:
                continue

            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            if not is_tissue(patch):
                continue

            save_path = os.path.join(save_dir, f"x{x}_y{y}.png")
            cv2.imwrite(save_path, patch)
            count += 1

    print(f"{slide_name}: {count} patches saved")


def process_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    slides = [f for f in os.listdir(INPUT_DIR) if f.endswith(".png")]

    print(f"Found {len(slides)} slides")

    for slide in slides:
        slide_name = os.path.splitext(slide)[0]
        slide_path = os.path.join(INPUT_DIR, slide)

        tile_slide(slide_path, slide_name)


if __name__ == "__main__":
    process_all()