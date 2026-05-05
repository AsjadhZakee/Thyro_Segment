# Thyro_Segment
A GUI to detect thyroid nodules from unclear Thyroid Ultra Sound images using image enhancement and segmentation techniques

## 📋 Overview

This project segments (detects) thyroid nodules in ultrasound images using:
- **Enhancement**: Anisotropic diffusion + CLAHE + Gamma correction + Frangi fusion
- **Segmentation**: Ensemble of 4 methods (Multi-Otsu, Adaptive, Watershed, Canny-Fill)
- **GUI**: Beautiful web interface for single-image analysis

## 📁 Project Structure

```
segmentation_only/
├── enhancement.py       # Image enhancement pipeline (library)
├── segmentation.py      # Nodule segmentation (library)
├── inference.py         # Single-image prediction (library)
├── train.py             # Training script 
├── app.py               # Web GUI server 
├── Datasets/
│   ├── p_image/         # ultrasound images
│   └── p_mask/          # Ground-truth nodule masks
└── results/             # Auto-created outputs
    ├── model.pkl        # Trained model info
    ├── sample_results.png
    └── metric_distributions.png
```

## 📊 Understanding the Output

### Training Output

```
[1/3] Loading dataset...
      Found 637 images

[2/3] Processing images (enhancement + segmentation)...
Processing: 100%|██████| 637/637 [08:24<00:00,  1.26 img/s]

[3/3] Results

=======================================================
  SEGMENTATION METRICS  (N=637 images)
=======================================================
  Dice  :  0.3081  ±  0.2089 
  IoU   : 0.2020  ±  0.1626 
=======================================================

   Dice > 0.60 (excellent) :   72 (11.3%)
  Dice 0.40-0.60 (good)   :  145 (22.8%)
  Dice 0.20-0.40 (fair)   :  166 (26.1%)
  Dice < 0.20 (poor)      :  254 (39.9%)
```

**What these metrics mean:**
- **Dice 0.30** = average 30% overlap between predicted and ground-truth mask
- **Good (>0.60)** = excellent segmentation on those 11.3% of images
- **Poor (<0.20)** = 39.9% of images are still challenging

## 🌐 Using the GUI

### What the GUI does:

1. **Upload** a thyroid ultrasound image
2. **Analysis** button runs:
   - Enhancement (7 steps)
   - Segmentation (ensemble of 4 methods)
   - Computes nodule area
3. **Results tab** shows:
   - Original image
   - Enhanced image
   - Prediction overlay (red nodule region)
   - Saliency map
4. **Methods tab** shows output of each of the 4 segmentation methods
5. **Steps tab** shows intermediate enhancement steps
