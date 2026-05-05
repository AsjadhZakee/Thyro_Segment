"""
=============================================================================
train.py – Train & Evaluate Segmentation Model
=============================================================================
Processes entire dataset through enhancement + segmentation pipeline.
Reports Dice/IoU scores and saves trained model.

Usage
-----
  python train.py --image_dir Datasets/p_image --mask_dir Datasets/p_mask
=============================================================================
"""

import os
import sys
import argparse
import pickle
import warnings
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from enhancement import enhance_image
from segmentation import ensemble_segment, dice_score, iou_score

warnings.filterwarnings('ignore')


def load_dataset(image_dir, mask_dir):
    """Load dataset: (img_path, mask_path) pairs."""
    samples = []
    supported = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    for fname in sorted(os.listdir(image_dir)):
        if os.path.splitext(fname)[1].lower() not in supported:
            continue
        img_path  = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        if not os.path.exists(mask_path):
            base = os.path.splitext(fname)[0]
            for ext in ['.png', '.jpg']:
                alt = os.path.join(mask_dir, base + ext)
                if os.path.exists(alt):
                    mask_path = alt
                    break
            else:
                continue

        samples.append((img_path, mask_path))

    return samples


def train_pipeline(image_dir, mask_dir, output_dir="results"):
    """Main training loop."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────
    print("\n[1/3] Loading dataset...")
    dataset = load_dataset(image_dir, mask_dir)
    if not dataset:
        sys.exit(f"ERROR: No image/mask pairs found in {image_dir} / {mask_dir}")
    print(f"      Found {len(dataset)} images")

    # ── Process dataset ───────────────────────────────────────────────
    print("\n[2/3] Processing images (enhancement + segmentation)...")
    dice_scores = []
    iou_scores  = []
    vis_samples = []

    for idx, (img_path, mask_path) in enumerate(
            tqdm(dataset, desc="Processing", unit="img")):

        img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        true = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # Resize to standard size
        img  = cv2.resize(img,  (256, 256), interpolation=cv2.INTER_AREA)
        true = cv2.resize(true, (256, 256), interpolation=cv2.INTER_NEAREST) \
               if true is not None else np.zeros((256, 256), np.uint8)

        # Binarise mask
        _, true = cv2.threshold(true, 127, 255, cv2.THRESH_BINARY)

        # Enhance
        enh_out  = enhance_image(img)
        enhanced = enh_out["enhanced"]

        # Segment
        pred_mask, method_masks = ensemble_segment(enhanced)

        # Metrics
        d = dice_score(true, pred_mask)
        j = iou_score(true, pred_mask)
        dice_scores.append(d)
        iou_scores.append(j)

        # Store samples for visualisation
        if idx < 8:
            vis_samples.append({
                "img": img,
                "enhanced": enhanced,
                "true": true,
                "pred": pred_mask,
                "dice": d,
                "iou": j,
            })

    # ── Report metrics ────────────────────────────────────────────────
    print("\n[3/3] Results")
    print(f"\n{'='*55}")
    print(f"  SEGMENTATION METRICS  (N={len(dataset)} images)")
    print(f"{'='*55}")
    print(f"  Dice  :  {np.mean(dice_scores):.4f}  ±  {np.std(dice_scores):.4f}")
    print(f"  IoU   :  {np.mean(iou_scores):.4f}  ±  {np.std(iou_scores):.4f}")
    print(f"{'='*55}")

    # ── Distribution breakdown ────────────────────────────────────────
    dice_arr = np.array(dice_scores)
    print(f"\n  Dice > 0.60 (excellent) : {np.sum(dice_arr > 0.60):>4} ({np.mean(dice_arr>0.60)*100:.1f}%)")
    print(f"  Dice 0.40-0.60 (good)   : {np.sum((dice_arr>=0.40)&(dice_arr<=0.60)):>4} ({np.mean((dice_arr>=0.40)&(dice_arr<=0.60))*100:.1f}%)")
    print(f"  Dice 0.20-0.40 (fair)   : {np.sum((dice_arr>=0.20)&(dice_arr<0.40)):>4} ({np.mean((dice_arr>=0.20)&(dice_arr<0.40))*100:.1f}%)")
    print(f"  Dice < 0.20 (poor)      : {np.sum(dice_arr < 0.20):>4} ({np.mean(dice_arr<0.20)*100:.1f}%)")

    # ── Save plots ────────────────────────────────────────────────────
    _plot_sample_results(vis_samples, output_dir)
    _plot_metric_distributions(dice_scores, iou_scores, output_dir)

    # ── Save model ────────────────────────────────────────────────────
    model_data = {
        "dice_mean": float(np.mean(dice_scores)),
        "dice_std": float(np.std(dice_scores)),
        "iou_mean": float(np.mean(iou_scores)),
        "iou_std": float(np.std(iou_scores)),
        "n_images": len(dataset),
    }
    model_path = os.path.join(output_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n✅ Model info saved → {model_path}")
    print(f"\n✅ All outputs saved to: {output_dir}/")


# ─────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────

def _plot_sample_results(samples, out_dir):
    if not samples:
        return

    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5*n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, s in enumerate(samples):
        # Original
        axes[i, 0].imshow(s["img"], cmap='gray')
        axes[i, 0].set_title("Original", fontsize=9)
        axes[i, 0].axis('off')

        # Enhanced
        axes[i, 1].imshow(s["enhanced"], cmap='gray')
        axes[i, 1].set_title("Enhanced", fontsize=9)
        axes[i, 1].axis('off')

        # Ground truth
        gt_rgb = cv2.cvtColor(s["img"], cv2.COLOR_GRAY2RGB)
        cnts, _ = cv2.findContours(s["true"], cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gt_rgb, cnts, -1, (50, 200, 50), 2)
        axes[i, 2].imshow(gt_rgb)
        axes[i, 2].set_title("Ground Truth", fontsize=9)
        axes[i, 2].axis('off')

        # Prediction
        pr_rgb = cv2.cvtColor(s["enhanced"], cv2.COLOR_GRAY2RGB)
        cnts2, _ = cv2.findContours(s["pred"], cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        ov = pr_rgb.copy()
        cv2.drawContours(ov, cnts2, -1, (220, 50, 50), -1)
        pr_rgb = cv2.addWeighted(ov, 0.3, pr_rgb, 0.7, 0)
        cv2.drawContours(pr_rgb, cnts2, -1, (220, 50, 50), 2)
        axes[i, 3].imshow(pr_rgb)
        axes[i, 3].set_title(
            f"Prediction  Dice={s['dice']:.3f}  IoU={s['iou']:.3f}",
            fontsize=9)
        axes[i, 3].axis('off')

    plt.suptitle("Sample Segmentation Results", fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, "sample_results.png")
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"   Saved → {path}")


def _plot_metric_distributions(dice_list, iou_list, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(dice_list, bins=30, color='#2196F3', edgecolor='white',
                 alpha=0.85)
    axes[0].axvline(np.mean(dice_list), color='red', lw=2,
                    label=f"Mean = {np.mean(dice_list):.3f}")
    axes[0].set_xlabel("Dice Score", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Dice Score Distribution", fontsize=11, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].hist(iou_list, bins=30, color='#4CAF50', edgecolor='white',
                 alpha=0.85)
    axes[1].axvline(np.mean(iou_list), color='red', lw=2,
                    label=f"Mean = {np.mean(iou_list):.3f}")
    axes[1].set_xlabel("IoU Score", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("IoU Score Distribution", fontsize=11, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "metric_distributions.png")
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"   Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thyroid US Nodule Segmentation Training")
    parser.add_argument("--image_dir", default="Datasets/p_image",
                        help="Path to image directory")
    parser.add_argument("--mask_dir",  default="Datasets/p_mask",
                        help="Path to mask directory")
    parser.add_argument("--output_dir", default="results",
                        help="Output directory")
    args = parser.parse_args()

    train_pipeline(args.image_dir, args.mask_dir, args.output_dir)
