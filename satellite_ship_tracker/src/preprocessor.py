"""
Image loading and preprocessing pipeline for satellite imagery.

Handles loading, resizing, contrast enhancement (CLAHE), and optional
frame-to-frame alignment using ORB feature matching + homography.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

CONFIG = {
    "target_size": (800, 800),
    "clahe_clip_limit": 3.0,
    "clahe_tile_grid": (8, 8),
    "align_frames": True,
    "max_features": 500,
    "min_match_count": 4,
}


def load_images(
    folder: str,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tiff", ".tif"),
) -> Tuple[List[np.ndarray], List[str]]:
    """Load and sort images from a directory by filename.

    Args:
        folder: Path to directory containing satellite images.
        extensions: Accepted file extensions.

    Returns:
        Tuple of (images list, filenames list).
    """
    folder = Path(folder)
    files = sorted(
        f for f in folder.iterdir() if f.suffix.lower() in extensions
    )
    images, names = [], []
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            images.append(img)
            names.append(f.name)
        else:
            print(f"  Warning: could not read {f.name}, skipping.")
    return images, names


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = CONFIG["target_size"],
) -> np.ndarray:
    """Resize image to target (width, height) using bilinear interpolation."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = CONFIG["clahe_clip_limit"],
    tile_grid: Tuple[int, int] = CONFIG["clahe_tile_grid"],
) -> np.ndarray:
    """Apply CLAHE contrast enhancement to make ships stand out against water.

    Works on both grayscale and BGR images (operates on L channel in LAB space).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return clahe.apply(image)


def align_frame_to_reference(
    frame: np.ndarray,
    reference: np.ndarray,
    max_features: int = CONFIG["max_features"],
    min_matches: int = CONFIG["min_match_count"],
) -> np.ndarray:
    """Align a frame to the reference frame using ORB + RANSAC homography.

    Corrects for satellite camera drift between frames. Returns the original
    frame unchanged if alignment cannot be computed.
    """
    to_gray = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray_ref = to_gray(reference)
    gray_frm = to_gray(frame)

    orb = cv2.ORB_create(max_features)
    kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
    kp_frm, des_frm = orb.detectAndCompute(gray_frm, None)

    if des_ref is None or des_frm is None or len(kp_ref) < min_matches or len(kp_frm) < min_matches:
        return frame

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des_ref, des_frm), key=lambda m: m.distance)[:50]

    if len(matches) < min_matches:
        return frame

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frm[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        return frame

    h, w = reference.shape[:2]
    return cv2.warpPerspective(frame, H, (w, h))


def preprocess_frames(
    images: List[np.ndarray],
    align: bool = CONFIG["align_frames"],
) -> List[np.ndarray]:
    """Full preprocessing pipeline: resize → CLAHE → optional alignment.

    Args:
        images: Raw loaded frames (BGR or grayscale).
        align: Whether to align subsequent frames to the first frame.

    Returns:
        List of preprocessed frames ready for detection.
    """
    processed = []
    reference: Optional[np.ndarray] = None

    for i, img in enumerate(images):
        img = resize_image(img)
        img = apply_clahe(img)

        if align:
            if i == 0:
                reference = img
            elif reference is not None:
                img = align_frame_to_reference(img, reference)

        processed.append(img)

    return processed
