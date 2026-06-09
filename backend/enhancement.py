import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Enhance a low light or low contrast BGR frame using CLAHE on the L channel
    of the LAB color space, followed by bilateral filtering.

    Why LAB + L channel only:
        Applying histogram equalisation directly to BGR channels shifts the hue
        (colour fidelity distortion). Operating on the L (luminance) channel
        alone preserves colour while improving contrast.

    Why bilateral filter (not Gaussian):
        CCTV sensor noise is approximately Gaussian. Bilateral filtering removes
        Gaussian noise while preserving sharp edges, which is critical for
        accurate downstream bounding box detection. A Gaussian blur would soften
        luggage/person edges and reduce YOLOv8 accuracy.

    Args:
        frame: BGR image as a NumPy array (H, W, 3), dtype uint8.

    Returns:
        Enhanced BGR image, same shape and dtype as input.
    """
    if frame is None or frame.size == 0:
        raise ValueError("enhance_frame received an empty or None frame.")

    # Step 1: BGR to LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Step 2: CLAHE on L channel only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Step 3: Reconstruct BGR from enhanced LAB
    enhanced_bgr = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)

    # Step 4: Bilateral filter — edge-preserving noise reduction
    result = cv2.bilateralFilter(enhanced_bgr, d=9, sigmaColor=75, sigmaSpace=75)

    return result


def evaluate_enhancement(original: np.ndarray, enhanced: np.ndarray) -> dict:
    """
    Compute PSNR and SSIM between original and enhanced frames.
    This will be Used to fill the quantitative results table in the report.

    Args:
        original:  Original (unenhanced) BGR frame.
        enhanced:  Enhanced BGR frame.

    Returns:
        dict with keys 'psnr' (float, dB) and 'ssim' (float, 0–1).
    """
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    enh_gray  = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    psnr_val = psnr(orig_gray, enh_gray, data_range=255)
    ssim_val = ssim(orig_gray, enh_gray, data_range=255)

    return {"psnr": round(psnr_val, 2), "ssim": round(float(ssim_val), 4)}


def simulate_dark_frame(frame: np.ndarray, gamma: float = 2.5) -> np.ndarray:
    """
    Artificially darken a frame to simulate low light CCTV conditions.
    Used to generate the before/after test set.

    Args:
        frame: BGR image, dtype uint8.
        gamma: Values > 1 darken the image (gamma correction).

    Returns:
        Darkened BGR image, same shape and dtype.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, table)

# Standalone test, run: python enhance.py

if __name__ == "__main__":
    import sys
    import os

    test_path = "test_frame.jpg"
    if not os.path.exists(test_path):
        print(f"Place a test image at '{test_path}' and re-run.")
        sys.exit(1)

    original = cv2.imread(test_path)
    dark     = simulate_dark_frame(original, gamma=2.5)
    enhanced = enhance_frame(dark)

    cv2.imwrite("output_dark.jpg",     dark)
    cv2.imwrite("output_enhanced.jpg", enhanced)

    metrics = evaluate_enhancement(dark, enhanced)
    print("=== Enhancement evaluation ===")
    print(f"  PSNR:  {metrics['psnr']} dB   (higher = less noise introduced)")
    print(f"  SSIM:  {metrics['ssim']}       (closer to 1 = more structural similarity)")
    print("Saved: output_dark.jpg  →  output_enhanced.jpg")