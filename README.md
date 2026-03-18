# 🧬 Biomarker Classification from H&E Slides using Attention-based MIL

## 📌 Overview

This project implements a **deep learning pipeline** for predicting molecular biomarkers from **Hematoxylin & Eosin (H&E) stained histopathology images**.

The system:

* Processes whole-slide images (WSI) or large PNG slides
* Breaks them into smaller patches
* Uses an **Attention-based Multiple Instance Learning (MIL)** model
* Predicts **three biomarkers**:

  * IDH1R132H
  * ATRX
  * P53
* Generates **biomarker-specific heatmaps** for **Region of Interest (ROI) detection**

---

## 🚀 Pipeline

```plaintext
Raw Slides (PNG / WSI)
        ↓
Tiling (224×224 patches)
        ↓
Tissue Filtering
        ↓
Stain Normalization (Macenko)
        ↓
Dataset Loader
        ↓
MIL Model (ResNet + Attention)
        ↓
Training (BCE Loss)
        ↓
Prediction (3 biomarkers)
        ↓
Attention Heatmaps (ROI Visualization)
```

---

## 🧠 Key Concepts

### 🔹 Multiple Instance Learning (MIL)

* Each slide is treated as a **bag of patches**
* Only **slide-level labels** are available
* Model learns which patches are important

### 🔹 Attention Mechanism

* Assigns **importance weights** to each patch
* Enables:

  * Better prediction
  * Interpretability

### 🔹 ROI Detection

* Attention scores are mapped back to the slide
* Produces **heatmaps highlighting important regions**

---

## 🏗️ Model Architecture

* **Backbone:** ResNet18 (pretrained)
* **Feature Size:** 512-d per patch
* **Attention:** Separate attention networks per biomarker
* **Classifier:** Fully connected layer → 3 outputs

---

## 📂 Project Structure

```plaintext
EHRC/
│
├── labelled_png/              # Raw slides
├── normalized_dataset/        # Tiled patches
├── processed_dataset/         # Intermediate data
│
├── dataset.py                 # Data loader
├── model.py                   # MIL model + attention
├── train.py                   # Training + evaluation
├── heatmaps.py                # ROI visualization
├── tile_png_slides.py         # Tiling pipeline
├── Stain_normalisation.py     # Stain normalization
├── prepare_dataset.py         # Full pipeline automation
│
├── labels.csv                 # Slide-level labels
├── README.md
```

---

## ⚙️ Installation

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install torch torchvision opencv-python numpy pandas matplotlib scikit-learn torchstain
```

---

## ▶️ Usage

### 1️⃣ Prepare Dataset

```bash
python tile_png_slides.py
python Stain_normalisation.py
```

OR

```bash
python prepare_dataset.py
```

---

### 2️⃣ Train Model

```bash
python train.py
```

---

### 3️⃣ Generate Heatmaps (ROI)

```bash
python heatmaps.py
```

---

## 📊 Output

### ✔ Predictions

* IDH1R132H
* ATRX
* P53

### ✔ Heatmaps

* Biomarker-wise ROI visualization
* Highlights regions contributing to predictions

---

## ⚠️ Notes

* Dataset size is currently **small**, so:

  * AUROC may be unstable
  * Some biomarkers may show `NaN` (single-class issue)

* For larger datasets:

  * Use proper **train/validation/test splits**
  * Avoid overfitting

---

## 🔬 Future Improvements

* Transformer-based models (ViT)
* Better patch sampling strategies
* Whole-slide (.svs) integration
* Streamlit UI for deployment
* Larger dataset for robust evaluation

---

