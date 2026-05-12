"""
Ship detection using classical computer vision or optional YOLOv8.

Classical pipeline: adaptive thresholding → morphological cleanup → contour
filtering by area and aspect ratio.  YOLOv8 is used when ultralytics is
installed and mode="yolo" is requested; falls back to classical automatically.
"""

import cv2
import numpy as np
from typing import Any, Dict, List, Tuple

# Type alias for a single detection result
Detection = Dict[str, Any]  # {"bbox": [x, y, w, h], "confidence": float, "class": str}

CONFIG = {
    # Contour filtering
    "min_area": 80,
    "max_area": 40000,
    "min_aspect_ratio": 1.5,
    "max_aspect_ratio": 12.0,
    # Adaptive threshold params
    "threshold_block_size": 21,
    "threshold_C": -5,
    # Morphological structuring element size
    "morph_kernel_size": 3,
    # YOLO settings
    "yolo_model": "yolov8n.pt",
    "confidence_threshold": 0.25,
}


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale if needed."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def compute_iou(box_a: List[int], box_b: List[int]) -> float:
    """Compute Intersection-over-Union for two bboxes in [x, y, w, h] format."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / (union + 1e-6)


def classical_detect(image: np.ndarray, config: dict = CONFIG) -> List[Detection]:
    """Detect ships via adaptive thresholding + morphological ops + contour filtering.

    Highlights bright, elongated blobs characteristic of ships on dark water.

    Args:
        image: Preprocessed frame (BGR or grayscale).
        config: Detection hyperparameters.

    Returns:
        List of detection dicts with "bbox", "confidence", "class" keys.
    """
    gray = _to_gray(image)

    # Adaptive threshold — handles uneven illumination across the frame
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        config["threshold_block_size"],
        config["threshold_C"],
    )

    # Morphological close to merge ship pixels, then open to remove noise
    k = config["morph_kernel_size"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: List[Detection] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config["min_area"] or area > config["max_area"]:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        long_side  = max(w, h)
        short_side = min(w, h) + 1e-6
        aspect = long_side / short_side

        if aspect < config["min_aspect_ratio"] or aspect > config["max_aspect_ratio"]:
            continue

        # Heuristic confidence: reward ship-like size and 4:1 aspect ratio
        area_score   = min(area / 4000.0, 1.0)
        aspect_score = max(0.0, 1.0 - abs(aspect - 4.0) / 8.0)
        confidence   = 0.5 * area_score + 0.5 * aspect_score

        detections.append({
            "bbox":       [x, y, w, h],
            "confidence": float(confidence),
            "class":      "ship",
        })

    return detections


def merge_tile_detections(
    tile_detections: List[List[Detection]],
    tile_bboxes: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.3,
) -> List[Detection]:
    """Map tile-local detections back to full-image coordinates and apply NMS."""
    full = []
    for dets, (tx, ty, tw, th) in zip(tile_detections, tile_bboxes):
        for det in dets:
            x, y, w, h = det["bbox"]
            x = min(x, tw - 1)
            y = min(y, th - 1)
            w = min(w, tw - x)
            h = min(h, th - y)
            full.append({
                "bbox": [tx + x, ty + y, w, h],
                "confidence": det["confidence"],
                "class": det["class"],
            })

    full.sort(key=lambda d: d["confidence"], reverse=True)
    kept = []
    suppressed = set()
    for i, det in enumerate(full):
        if i in suppressed:
            continue
        kept.append(det)
        for j in range(i + 1, len(full)):
            if j not in suppressed and compute_iou(det["bbox"], full[j]["bbox"]) > iou_threshold:
                suppressed.add(j)
    return kept


def yolo_detect(image: np.ndarray, model, config: dict = CONFIG) -> List[Detection]:
    """Run YOLOv8 inference and return all detections above confidence threshold.

    Args:
        image: BGR frame.
        model: Loaded YOLO model instance.
        config: Detection hyperparameters.

    Returns:
        List of detection dicts.
    """
    results = model(image, verbose=False)[0]
    detections: List[Detection] = []
    for box in results.boxes:
        conf = float(box.conf)
        if conf < config["confidence_threshold"]:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "bbox":       [x1, y1, x2 - x1, y2 - y1],
            "confidence": conf,
            "class":      results.names[int(box.cls)],
        })
    return detections


class ShipDetector:
    """Unified ship detector supporting classical CV and optional YOLOv8.

    Automatically falls back to classical detection if YOLO is unavailable.
    """

    def __init__(self, mode: str = "classical", config: dict = CONFIG):
        """
        Args:
            mode: "classical" or "yolo".
            config: Detection hyperparameters dict.
        """
        self.mode   = mode
        self.config = config
        self.model  = None

        if mode == "yolo":
            try:
                from ultralytics import YOLO  # type: ignore
                self.model = YOLO(config["yolo_model"])
                print("YOLOv8 model loaded successfully.")
            except ImportError:
                print("ultralytics not installed — falling back to classical detection.")
                self.mode = "classical"
            except Exception as exc:
                print(f"Failed to load YOLO model ({exc}) — falling back to classical detection.")
                self.mode = "classical"

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect ships in a single frame.

        Returns:
            List of Detection dicts: [{"bbox": [x,y,w,h], "confidence": float, "class": str}, ...]
        """
        if self.mode == "yolo" and self.model is not None:
            return yolo_detect(image, self.model, self.config)
        return classical_detect(image, self.config)

    def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        """Detect ships in a list of frames.

        Returns:
            List of detection lists, one per frame.
        """
        return [self.detect(img) for img in images]
