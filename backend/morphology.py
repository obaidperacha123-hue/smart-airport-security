import cv2
import numpy as np

# Why morphological operations here:
#   YOLOv8 returns bounding boxes, but when we extract a binary mask from the
#   detected region (like via thresholding the ROI), the mask usually contains:
#       - Small noise dots in the background,  removed by OPENING
#       - Small holes/gaps in the object body, filled by CLOSING
#
#   This cleans the mask before it is passed to the tracker, reducing jitter
#   in the bounding box coordinates and improving DeepSORT's IoU matching.

def threshold_roi(frame: np.ndarray, bbox: list[float]) -> np.ndarray:
    """
    Extract a binary mask for the region inside a detection bounding box.
    Uses adaptive Gaussian thresholding to handle varying CCTV lighting.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Clamp coordinates to frame bounds
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return np.zeros((h, w), dtype=np.uint8)

    roi = frame[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Adaptive Gaussian threshold handles uneven illumination within the ROI.
    binary_roi = cv2.adaptiveThreshold(
        gray_roi, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=2
    )

    # Place the ROI mask back into a full frame mask
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = binary_roi
    return full_mask


def refine_mask(binary_mask: np.ndarray,
                kernel_size: int = 3) -> np.ndarray:
    """
    Apply opening then closing to a binary mask.

    Args:
        binary_mask: uint8 binary image (0 or 255).
        kernel_size: Side length of the rectangular structuring element.

    Returns:
        Refined binary mask, same shape as input.
    """
    # Rectangular structuring element 
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )

    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  kernel)
    closed = cv2.morphologyEx(opened,      cv2.MORPH_CLOSE, kernel)

    return closed


def refine_detection_bbox(frame: np.ndarray,
                           bbox: list[float]) -> list[float]:
    """
    Full pipeline: threshold the detection ROI, apply morphological refinement,
    and return a tighter bounding box fitted to the cleaned mask contour.

    This gives the tracker a more stable, noise free bbox on each frame.

    Args:
        frame: Enhanced BGR frame.
        bbox:  [x1, y1, x2, y2] from YOLOv8 detect_luggage().

    Returns:
        Refined [x1, y1, x2, y2].  Falls back to original bbox if
        no contour is found after refinement.
    """
    mask    = threshold_roi(frame, bbox)
    refined = refine_mask(mask)

    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bbox  # fallback: return original

    # Use the largest contour as the object boundary
    largest = max(contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(largest)
    return [float(rx), float(ry), float(rx + rw), float(ry + rh)]


def process_detections(frame: np.ndarray,
                        detections: list[dict]) -> list[dict]:
    """
    Apply morphological refinement to every detection in a frame.
    Drop in replacement that takes and returns the same list format as detect.py.

    Args:
        frame:      Enhanced BGR frame.
        detections: Output of detect_luggage() from detect.py.

    Returns:
        Detections with refined bboxes (original conf and class unchanged).
    """
    refined = []
    for d in detections:
        new_bbox = refine_detection_bbox(frame, d["bbox"])
        refined.append({**d, "bbox": new_bbox})
    return refined


def visualise_morphology(frame: np.ndarray,
                          bbox: list[float]) -> dict[str, np.ndarray]:
    """
    Return intermediate images that we can use for the report (before/after each step).
    Call this on one example detection and save the results as screenshots.

    Returns dict with keys: 'original_roi', 'thresholded', 'after_open', 'after_close'
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(w,x2), min(h,y2)

    roi      = frame[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh   = cv2.adaptiveThreshold(gray_roi, 255,
                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                   cv2.THRESH_BINARY, 11, 2)

    kernel      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    after_open  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)
    after_close = cv2.morphologyEx(after_open, cv2.MORPH_CLOSE, kernel)

    return {
        "original_roi":  roi,
        "thresholded":   thresh,
        "after_open":    after_open,
        "after_close":   after_close,
    }

# Standalone test — run: python morphology.py
# Requires output_detections.jpg and a detection from detect.py.
# Saves the four morphology stages as separate images 
if __name__ == "__main__":
    import sys
    from enhancement import enhance_frame
    from detection   import detect_luggage

    test_path = "test_frame.jpg"
    if not __import__("os").path.exists(test_path):
        print(f"Place a test image at '{test_path}' and re-run.")
        sys.exit(1)

    original   = cv2.imread(test_path)
    enhanced   = enhance_frame(original)
    detections = detect_luggage(enhanced)

    if not detections:
        print("No luggage detected in test image. Try a different image.")
        sys.exit(0)

    # Use the first detection for visualisation
    first_bbox = detections[0]["bbox"]
    stages     = visualise_morphology(enhanced, first_bbox)

    for name, img in stages.items():
        out_path = f"output_morph_{name}.jpg"
        cv2.imwrite(out_path, img)
        print(f"Saved: {out_path}")

    # Also run full pipeline and print refined vs original bbox
    refined_detections = process_detections(enhanced, detections)
    print("\n=== Bbox refinement ===")
    for orig, ref in zip(detections, refined_detections):
        o = [round(v) for v in orig["bbox"]]
        r = [round(v) for v in ref["bbox"]]
        print(f"  original: {o}  →  refined: {r}")