"""
=============================================================================
enhancement.py – Thyroid US Image Enhancement for Nodule Detection
=============================================================================
Focuses purely on making nodules more visible (not classification).

Pipeline:
  1. Anisotropic diffusion     – edge-preserving speckle removal
  2. Bilateral/Wavelet denoise – residual noise removal
  3. Adaptive CLAHE            – local contrast enhancement
  4. Gamma correction          – global brightness tuning
  5. Unsharp mask              – edge sharpening
  6. Frangi saliency fusion    – selective nodule region brightening
=============================================================================
"""

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import img_as_float, img_as_ubyte
from skimage.filters import frangi


def anisotropic_diffusion(img, num_iter=12, kappa=35, gamma=0.25):
    """
    Perona-Malik anisotropic diffusion.
    Smooths speckle while preserving sharp nodule edges.
    """
    img_f = img.astype(np.float64) / 255.0
    for _ in range(num_iter):
        dN = np.roll(img_f, -1, axis=0) - img_f
        dS = np.roll(img_f,  1, axis=0) - img_f
        dE = np.roll(img_f, -1, axis=1) - img_f
        dW = np.roll(img_f,  1, axis=1) - img_f
        cN = np.exp(-(dN / kappa) ** 2)
        cS = np.exp(-(dS / kappa) ** 2)
        cE = np.exp(-(dE / kappa) ** 2)
        cW = np.exp(-(dW / kappa) ** 2)
        img_f += gamma * (cN*dN + cS*dS + cE*dE + cW*dW)
    return np.clip(img_f * 255, 0, 255).astype(np.uint8)


def bilateral_denoise(img, d=9, sigma_color=55, sigma_space=55):
    """Bilateral filter – edge-preserving smoothing."""
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color,
                               sigmaSpace=sigma_space)


def adaptive_clahe(img):
    """CLAHE with tile size adapted to image resolution."""
    h, w = img.shape
    tile = max(8, min(h, w) // 16)
    tile = tile if tile % 2 == 0 else tile + 1
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(tile, tile))
    return clahe.apply(img)


def gamma_correction(img, gamma=1.3):
    """Brighten dark regions (hypoechoic nodules)."""
    lut = np.array([((i/255.0)**(1.0/gamma))*255
                    for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def unsharp_mask(img, radius=1.5, amount=1.2):
    """Sharpen nodule boundaries."""
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    sharpened = cv2.addWeighted(img, 1+amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def frangi_saliency(img):
    """
    Frangi vesselness filter highlights blob structures (nodules).
    Returns float32 saliency map [0, 1].
    """
    img_f = img_as_float(img)
    response = frangi(img_f, sigmas=range(2, 8, 1),
                      black_ridges=True, alpha=0.5, beta=0.5)
    rmin, rmax = response.min(), response.max()
    if rmax > rmin:
        response = (response - rmin) / (rmax - rmin)
    return response.astype(np.float32)


def enhance_image(img, return_steps=False):
    """
    Complete enhancement pipeline.
    
    Parameters
    ----------
    img : uint8 grayscale image
    return_steps : if True, return intermediate results
    
    Returns
    -------
    dict with 'enhanced', 'saliency', and optionally 'steps'
    """
    assert img.ndim == 2, "Expected grayscale image"
    
    steps = []
    
    # Step 1 – Anisotropic diffusion
    s1 = anisotropic_diffusion(img, num_iter=12, kappa=35)
    steps.append(("1. Anisotropic Diffusion", s1.copy()))
    
    # Step 2 – Bilateral denoise
    s2 = bilateral_denoise(s1)
    steps.append(("2. Bilateral Denoise", s2.copy()))
    
    # Step 3 – Adaptive CLAHE
    s3 = adaptive_clahe(s2)
    steps.append(("3. Adaptive CLAHE", s3.copy()))
    
    # Step 4 – Gamma correction
    s4 = gamma_correction(s3, gamma=1.3)
    steps.append(("4. Gamma Correction", s4.copy()))
    
    # Step 5 – Unsharp mask
    s5 = unsharp_mask(s4, radius=1.5, amount=1.2)
    steps.append(("5. Unsharp Mask", s5.copy()))
    
    # Step 6 – Frangi saliency
    saliency = frangi_saliency(s2)
    steps.append(("6. Frangi Saliency", (saliency*255).astype(np.uint8)))
    
    # Step 7 – Fusion: boost saliency regions
    alpha = 0.35
    fused = np.clip(s5.astype(np.float32) * (1 + alpha * saliency), 0, 255)
    fused = fused.astype(np.uint8)
    steps.append(("7. Fused Output", fused.copy()))
    
    result = {"enhanced": fused, "saliency": saliency}
    if return_steps:
        result["steps"] = steps
    return result
