"""
=============================================================================
segmentation.py – Thyroid Nodule Segmentation (Multi-Method Ensemble)
=============================================================================
4 complementary segmentation methods fused via weighted voting.

Methods:
  1. Multi-Otsu  – threshold-based, fast
  2. Adaptive    – local threshold, handles varying contrast
  3. Watershed   – marker-based, good for compact shapes
  4. Canny-Fill  – edge-based, captures irregular boundaries

Fusion:
  Confidence-weighted pixel voting. A pixel is nodule if methods
  contributing ≥35% of weight agree.
=============================================================================
"""

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, remove_small_holes


# ─────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────

def _clean_mask(mask, min_area=200, close_ksize=7, open_ksize=3):
    """Morphological cleanup: close gaps, open noise, remove tiny blobs."""
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (close_ksize, close_ksize))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (open_ksize, open_ksize))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    m = cv2.morphologyEx(m,    cv2.MORPH_OPEN,  k_open)
    m_bool = remove_small_objects(m.astype(bool), min_size=min_area)
    m_bool = remove_small_holes(m_bool, area_threshold=min_area)
    return (m_bool * 255).astype(np.uint8)


def _keep_largest(mask):
    """Keep only the single largest connected component."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if contours:
        best = max(contours, key=cv2.contourArea)
        # Don't discard if it covers most of image (nodules CAN be large)
        if cv2.contourArea(best) < 0.95 * mask.size:
            cv2.drawContours(out, [best], -1, 255, -1)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Segmentation Methods
# ─────────────────────────────────────────────────────────────────────────

def segment_multi_otsu(enhanced):
    """3-level Otsu – picks class with best fit to nodule patterns."""
    h, w = enhanced.shape
    cx, cy = w // 2, h // 2

    try:
        thresholds = threshold_multiotsu(enhanced, classes=3)
    except Exception:
        _, t = cv2.threshold(enhanced, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return _clean_mask(_keep_largest((t > 0).astype(np.uint8) * 255))

    regions = np.digitize(enhanced, bins=thresholds)
    best_mask  = np.zeros((h, w), dtype=np.uint8)
    best_score = -1

    # Try all 3 classes, score by centrality + reasonable area
    for cls in range(3):
        candidate = ((regions == cls).astype(np.uint8) * 255)
        cleaned   = _clean_mask(candidate)
        if np.sum(cleaned) == 0:
            continue

        ys, xs = np.where(cleaned > 0)
        area_frac = np.sum(cleaned > 0) / cleaned.size
        
        # Skip if too small or covers entire image
        if area_frac < 0.005 or area_frac > 0.88:
            continue

        # Score: prefer central, moderate-sized regions
        dist_to_centre = np.sqrt((xs.mean()-cx)**2 + (ys.mean()-cy)**2)
        score = 1.0 / (dist_to_centre + 1) * area_frac
        
        if score > best_score:
            best_score = score
            best_mask  = cleaned

    if best_score < 0:
        centre_cls = regions[cy, cx]
        best_mask = _clean_mask(
            ((regions == centre_cls).astype(np.uint8) * 255))

    return _keep_largest(best_mask)


def segment_adaptive_thresh(enhanced):
    """Adaptive threshold – works when nodule brightness varies."""
    h, w = enhanced.shape
    block = max(11, (min(h, w) // 8) | 1)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Try both BINARY and BINARY_INV (hypo vs hyperechoic)
    thresh1 = cv2.adaptiveThreshold(blurred, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,
                                    block, -5)
    thresh2 = cv2.adaptiveThreshold(blurred, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV,
                                    block, -5)

    m1 = _clean_mask(_keep_largest(thresh1))
    m2 = _clean_mask(_keep_largest(thresh2))

    # Pick the more central one
    def centrality(m):
        ys, xs = np.where(m > 0)
        if len(ys) == 0:
            return 0
        cx, cy = w // 2, h // 2
        return 1.0 / (np.sqrt((xs.mean()-cx)**2 +
                              (ys.mean()-cy)**2) + 1)

    return m1 if centrality(m1) >= centrality(m2) else m2


def segment_watershed(enhanced):
    """Watershed – good for compact, well-separated nodules."""
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dist = ndi.distance_transform_edt(binary)
    coords = peak_local_max(dist, min_distance=15,
                            labels=binary.astype(bool))
    markers = np.zeros(dist.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    if markers.max() == 0:
        return np.zeros_like(enhanced, dtype=np.uint8)

    labels = watershed(-dist, markers, mask=binary.astype(bool))
    h, w   = enhanced.shape
    cx, cy = w // 2, h // 2

    best_label, best_d = None, np.inf
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        ys, xs = np.where(labels == lbl)
        if len(ys) < 50:
            continue
        d = np.hypot(xs.mean()-cx, ys.mean()-cy)
        if d < best_d:
            best_d = d
            best_label = lbl

    mask = np.zeros((h, w), dtype=np.uint8)
    if best_label is not None:
        mask[labels == best_label] = 255
    return _clean_mask(mask)


def segment_canny_fill(enhanced):
    """Canny edges + flood fill – captures irregular boundaries."""
    h, w    = enhanced.shape
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    med     = np.median(blurred)
    lo      = int(max(0,   0.33 * med))
    hi      = int(min(255, 1.00 * med))
    edges   = cv2.Canny(blurred, lo, hi)
    k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_d = cv2.dilate(edges, k, iterations=2)
    inv     = cv2.bitwise_not(edges_d)
    fill    = inv.copy()
    cv2.floodFill(fill, np.zeros((h+2, w+2), np.uint8),
                  (w//2, h//2), 255)
    return _clean_mask(_keep_largest(fill))


# ─────────────────────────────────────────────────────────────────────────
# Ensemble Segmentation
# ─────────────────────────────────────────────────────────────────────────

def ensemble_segment(enhanced):
    """
    Run 4 methods, fuse via confidence-weighted voting.
    
    Returns
    -------
    pred_mask : uint8 binary mask
    method_masks : dict of individual method results
    """
    h, w = enhanced.shape
    cx, cy = w // 2, h // 2

    masks = {
        "Multi-Otsu"  : segment_multi_otsu(enhanced),
        "Adaptive"    : segment_adaptive_thresh(enhanced),
        "Watershed"   : segment_watershed(enhanced),
        "Canny-Fill"  : segment_canny_fill(enhanced),
    }

    def mask_weight(m):
        """Weight = how plausible is this mask as a real nodule."""
        area_frac = np.sum(m > 0) / m.size
        
        # Penalise very small (<0.5%) or very large (>88%) masks
        if area_frac < 0.005 or area_frac > 0.88:
            return 0.1
        
        # Bonus for central location
        ys, xs = np.where(m > 0)
        if len(ys) == 0:
            return 0.1
        dist = np.hypot(xs.mean()-cx, ys.mean()-cy) / (min(h,w)/2)
        centrality = max(0, 1 - dist)
        
        return max(0.1, centrality * (1 - abs(area_frac - 0.15)))

    weights = {name: mask_weight(m) for name, m in masks.items()}
    total_w = sum(weights.values())

    # Weighted pixel voting
    vote = np.zeros((h, w), dtype=np.float32)
    if total_w > 0:
        for name, m in masks.items():
            vote += (weights[name] / total_w) * (m / 255.0)

    fused = ((vote >= 0.35) * 255).astype(np.uint8)
    fused = _clean_mask(fused)
    fused = _keep_largest(fused)

    # Fallback: if ensemble is empty, use best individual method
    if np.sum(fused) == 0:
        best_name = max(weights, key=weights.get)
        fused = masks[best_name]

    return fused, masks


# ─────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────

def dice_score(y_true, y_pred, smooth=1e-6):
    """Sørensen-Dice coefficient."""
    t = y_true.flatten().astype(float) / 255.0
    p = y_pred.flatten().astype(float) / 255.0
    return (2*np.sum(t*p) + smooth) / (np.sum(t) + np.sum(p) + smooth)


def iou_score(y_true, y_pred, smooth=1e-6):
    """Jaccard / IoU."""
    t = y_true.flatten().astype(float) / 255.0
    p = y_pred.flatten().astype(float) / 255.0
    inter = np.sum(t * p)
    union = np.sum(t) + np.sum(p) - inter
    return (inter + smooth) / (union + smooth)
