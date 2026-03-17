import os
import subprocess

# Paths
RAW_DATA = "/mnt/c/Users/Bhavya Jain/Downloads/H&E"
PROCESSED_DIR = "./processed_dataset"
NORMALIZED_DIR = "./normalized_dataset"

# Step 1: Run tiling
def run_tiling():
    print("\n=== STEP 1: TILING ===")
    subprocess.run(["python", "tiling.py"], check=True)

# Step 2: Run normalization
def run_normalization():
    print("\n=== STEP 2: STAIN NORMALIZATION ===")
    subprocess.run(["python", "Stain_normalization.py"], check=True)

# Step 3: Verify structure
def verify_dataset():
    print("\n=== STEP 3: VERIFYING DATASET ===")

    slides = os.listdir(NORMALIZED_DIR)
    slides = [s for s in slides if os.path.isdir(os.path.join(NORMALIZED_DIR, s))]

    print(f"Total slides: {len(slides)}")

    for slide in slides[:5]:
        patch_count = len(os.listdir(os.path.join(NORMALIZED_DIR, slide)))
        print(f"{slide}: {patch_count} patches")

    print("\nDataset ready ✅")

if __name__ == "__main__":
    run_tiling()
    run_normalization()
    verify_dataset()