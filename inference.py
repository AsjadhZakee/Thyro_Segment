"""
=============================================================================
inference.py – Single-Image Nodule Detection
=============================================================================
Loads an image and runs the full enhancement + segmentation pipeline.
Returns visualization-ready outputs.
=============================================================================
"""

import numpy as np
import cv2
from enhancement import enhance_image
from segmentation import ensemble_segment, dice_score, iou_score


def predict_single(img_bgr_or_gray, true_mask=None):
    """
    Run the full pipeline on a single image.
    
    """
    
    # ── Preprocessing ─────────────────────────────────────────────────
    if img_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr_or_gray.copy()

    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

    # ── Enhancement ───────────────────────────────────────────────────
    enh_out  = enhance_image(gray, return_steps=True)  
    enhanced = enh_out["enhanced"]   # uint8 enhanced image
    saliency = enh_out["saliency"]   # float32 blob detection map
    steps    = enh_out["steps"]      

    # ── Segmentation ──────────────────────────────────────────────────
    pred_mask, method_masks = ensemble_segment(enhanced)

    # ── Metrics ───────────────────────────────────────────────────────
    metrics = None
    if true_mask is not None:
        tm = cv2.resize(true_mask, (256, 256),
                        interpolation=cv2.INTER_NEAREST)
        _, tm = cv2.threshold(tm, 127, 255, cv2.THRESH_BINARY)
        metrics = {
            "dice": dice_score(tm, pred_mask),
            "iou": iou_score(tm, pred_mask),
        }

    # ── Nodule detection ──────────────────────────────────────────────
    nodule_area = np.sum(pred_mask > 0)
    nodule_area_pct = (nodule_area / pred_mask.size) * 100
    nodule_detected = nodule_area > 100  # at least 100 pixels

    # ── Overlay ───────────────────────────────────────────────────────
    overlay = _draw_overlay(enhanced, pred_mask)

    return {
    "original": gray,
    "enhanced": enhanced,
    "pred_mask": pred_mask,
    "overlay": overlay,
    "saliency": saliency,
    "method_masks": method_masks,
    "nodule_detected": nodule_detected,
    "nodule_area_pct": nodule_area_pct,
    "metrics": metrics,
    "steps": steps,  
    }


def _draw_overlay(enhanced, pred_mask):
    """Draw nodule contour on enhanced image (RGB)."""
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Semi-transparent fill
        overlay = rgb.copy()
        cv2.drawContours(overlay, contours, -1, (220, 50, 50), -1)
        rgb = cv2.addWeighted(overlay, 0.30, rgb, 0.70, 0)
        # Bright boundary
        cv2.drawContours(rgb, contours, -1, (255, 60, 60), 2)
    return rgb
