"""
Ship detection using classical computer vision, a drone-tuned detector,
or optional YOLOv8.

Classical pipeline: adaptive thresholding → morphological cleanup → contour
filtering by area and aspect ratio.  YOLOv8 is used when ultralytics is
installed and mode="yolo" is requested; falls back to classical automatically.
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
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
    # Drone detector settings
    "drone_min_area": 80,
    "drone_max_area": 20000,
    "drone_min_aspect_ratio": 1.2,
    "drone_max_aspect_ratio": 6.0,
    "drone_min_center_x_ratio": 0.45,
    "drone_max_center_y_ratio": 0.80,
    "drone_border_margin": 5,
    "drone_sat_min": 40,
    "drone_val_min": 80,
    "drone_distance_penalty": 0.02,
    "drone_multi_min_area": 140,
    "drone_multi_max_area": 5000,
    "drone_multi_min_aspect_ratio": 1.0,
    "drone_multi_max_aspect_ratio": 4.0,
    "drone_multi_min_center_x_ratio": 0.60,
    "drone_max_detections": 3,
    "drone_single_dominance_ratio": 5.0,
    "drone_multi_match_distance": 120.0,
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


def _drone_hull_mask(hsv: np.ndarray, broad: bool = False) -> np.ndarray:
    """Return a color mask for drone-boat hulls."""
    if broad:
        return cv2.inRange(hsv, (0, 40, 45), (179, 255, 255))

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    ranges = [
        ((35, 50, 40), (95, 255, 255)),   # green / cyan hulls
        ((0, 40, 50), (30, 255, 255)),    # orange / pink bows
        ((95, 40, 40), (140, 255, 255)),  # blue cabins / canopies
    ]
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo), np.array(hi)))
    return mask


def _refine_drone_bbox(
    image: np.ndarray,
    coarse_bbox: List[int],
    config: dict = CONFIG,
) -> List[int]:
    """Snap a coarse drone tracker box back onto the colorful boat hull."""
    x, y, w, h = coarse_bbox
    h_img, w_img = image.shape[:2]
    if y < 28 or x < 12 or x + w > w_img - 12 or y + h > h_img - 12:
        return coarse_bbox
    pad_x = max(24, int(round(w * 1.4)))
    pad_y = max(24, int(round(h * 1.4)))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)
    roi = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = _drone_hull_mask(hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coarse_cx = x + w / 2.0
    coarse_cy = y + h / 2.0
    coarse_area = max(1.0, float(w * h))

    best_score = None
    best_bbox = coarse_bbox
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < max(35.0, coarse_area * 0.03) or area > coarse_area * 2.5:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect < 1.0 or aspect > 4.5:
            continue

        cx = x1 + bx + bw / 2.0
        cy = y1 + by + bh / 2.0
        dist = float(np.hypot(cx - coarse_cx, cy - coarse_cy))
        score = area - 4.0 * dist
        if best_score is None or score > best_score:
            best_score = score
            pad_bx = max(3, int(round(bw * 0.12)))
            pad_by = max(3, int(round(bh * 0.12)))
            rx1 = max(0, x1 + bx - pad_bx)
            ry1 = max(0, y1 + by - pad_by)
            rx2 = min(w_img, x1 + bx + bw + pad_bx)
            ry2 = min(h_img, y1 + by + bh + pad_by)
            best_bbox = [rx1, ry1, rx2 - rx1, ry2 - ry1]

    return best_bbox


def drone_detect(
    image: np.ndarray,
    config: dict = CONFIG,
    prev_center: Tuple[float, float] | None = None,
    prev_centers: List[Tuple[float, float]] | None = None,
) -> List[Detection]:
    """Detect a single colorful boat in stable drone footage.

    This detector is tuned for the real_data* drone sequences where the hull is
    more saturated than the surrounding water and the camera is stationary.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_img, w_img = image.shape[:2]

    cyan = cv2.inRange(
        hsv,
        (70, config["drone_sat_min"], config["drone_val_min"]),
        (110, 255, 255),
    )
    orange = cv2.inRange(
        hsv,
        (5, config["drone_sat_min"], config["drone_val_min"]),
        (30, 255, 255),
    )
    # Focus on the colorful hull and cabin instead of the bright wake.
    mask = cv2.bitwise_or(cyan, orange)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[float, List[int]]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config["drone_min_area"] or area > config["drone_max_area"]:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        margin = config["drone_border_margin"]
        if x <= margin or y <= margin or x + w >= w_img - margin or y + h >= h_img - margin:
            continue

        long_side = max(w, h)
        short_side = min(w, h) + 1e-6
        aspect = long_side / short_side
        if aspect < config["drone_min_aspect_ratio"] or aspect > config["drone_max_aspect_ratio"]:
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0
        if cx < config["drone_min_center_x_ratio"] * w_img:
            continue

        score = area * 0.02 + min(aspect, 5.0)
        if prev_center is not None:
            dist = float(np.hypot(cx - prev_center[0], cy - prev_center[1]))
            score -= config["drone_distance_penalty"] * dist

        candidates.append((score, [x, y, w, h]))

    # When several boats are visible, use a broader saturation mask and keep the
    # strongest distinct candidates. This works well on the fixed-camera
    # real_data_1 sequence where boats are small but colorful.
    broad_mask = _drone_hull_mask(hsv)
    broad_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    broad_mask = cv2.morphologyEx(broad_mask, cv2.MORPH_OPEN, broad_kernel)
    broad_mask = cv2.morphologyEx(broad_mask, cv2.MORPH_CLOSE, broad_kernel, iterations=1)
    broad_contours, _ = cv2.findContours(broad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    multi_candidates: List[Tuple[float, Detection]] = []
    for cnt in broad_contours:
        area = cv2.contourArea(cnt)
        if area < config["drone_multi_min_area"] or area > config["drone_multi_max_area"]:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w / 2.0
        cy = y + h / 2.0
        if cx < config["drone_multi_min_center_x_ratio"] * w_img:
            continue
        if cy > config["drone_max_center_y_ratio"] * h_img:
            continue

        long_side = max(w, h)
        short_side = min(w, h) + 1e-6
        aspect = long_side / short_side
        if aspect < config["drone_multi_min_aspect_ratio"] or aspect > config["drone_multi_max_aspect_ratio"]:
            continue

        # Penalize low-frame clutter and give each boat box a little breathing room.
        score = area * (1.2 - (cy / h_img))
        pad_x = max(4, int(round(w * 0.12)))
        pad_y = max(4, int(round(h * 0.12)))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)
        bbox = [x1, y1, x2 - x1, y2 - y1]
        confidence = max(0.1, min(0.99, score / 1200.0))
        multi_candidates.append((score, {
            "bbox": bbox,
            "confidence": float(confidence),
            "class": "ship",
        }))

    multi_candidates.sort(key=lambda item: item[0], reverse=True)

    if prev_centers and len(prev_centers) > 1 and multi_candidates:
        candidate_dets = [det for _, det in multi_candidates]
        cost = np.full((len(prev_centers), len(candidate_dets)), 1e6, dtype=np.float32)
        for i, (px, py) in enumerate(prev_centers):
            for j, det in enumerate(candidate_dets):
                x, y, w, h = det["bbox"]
                cx = x + w / 2.0
                cy = y + h / 2.0
                dist = float(np.hypot(cx - px, cy - py))
                if dist <= config["drone_multi_match_distance"]:
                    cost[i, j] = dist
        row_ind, col_ind = linear_sum_assignment(cost)
        matched: List[Detection] = []
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < config["drone_multi_match_distance"]:
                matched.append(candidate_dets[j])
        if matched:
            return matched

    allow_multi_mode = prev_centers is None or len(prev_centers) != 1

    if allow_multi_mode and len(multi_candidates) >= 2:
        dominance = multi_candidates[0][0] / max(multi_candidates[1][0], 1e-6)
        if dominance < config["drone_single_dominance_ratio"]:
            picked: List[Detection] = []
            for _, det in multi_candidates:
                if all(compute_iou(det["bbox"], kept["bbox"]) < 0.2 for kept in picked):
                    picked.append(det)
                if len(picked) >= config["drone_max_detections"]:
                    break
            if picked:
                return picked

    if not candidates:
        if multi_candidates:
            return [multi_candidates[0][1]]
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_bbox = candidates[0]
    confidence = max(0.1, min(0.99, best_score / 100.0))
    return [{
        "bbox": best_bbox,
        "confidence": float(confidence),
        "class": "ship",
    }]


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
        self._prev_center: Tuple[float, float] | None = None
        self._prev_centers: List[Tuple[float, float]] = []
        self._mil_trackers: List[Any] = []
        self._mil_boxes: List[List[int]] = []

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
        if self.mode == "drone":
            if self._mil_trackers:
                tracked: List[Detection] = []
                centers: List[Tuple[float, float]] = []
                next_boxes: List[List[int]] = []
                for idx, tracker in enumerate(self._mil_trackers):
                    ok, box = tracker.update(image)
                    prev_bbox = self._mil_boxes[idx] if idx < len(self._mil_boxes) else None
                    if ok:
                        x, y, w, h = [int(round(v)) for v in box]
                        coarse_bbox = [x, y, max(1, w), max(1, h)]
                        if prev_bbox is not None:
                            prev_area = max(1, prev_bbox[2] * prev_bbox[3])
                            new_area = coarse_bbox[2] * coarse_bbox[3]
                            if (
                                new_area > prev_area * 1.8
                                or coarse_bbox[2] > prev_bbox[2] * 1.6
                                or coarse_bbox[3] > prev_bbox[3] * 1.6
                            ):
                                coarse_bbox = list(prev_bbox)
                    elif idx < len(self._mil_boxes):
                        coarse_bbox = list(self._mil_boxes[idx])
                    else:
                        continue

                    refined_bbox = _refine_drone_bbox(image, coarse_bbox, self.config)
                    x, y, w, h = refined_bbox
                    next_boxes.append(refined_bbox)
                    tracked.append({
                        "bbox": [x, y, max(1, w), max(1, h)],
                        "confidence": 0.9,
                        "class": "ship",
                    })
                    centers.append((x + w / 2.0, y + h / 2.0))
                if tracked:
                    self._mil_boxes = next_boxes
                    self._prev_centers = centers
                    self._prev_center = centers[0]
                    return tracked
                self._mil_trackers = []
                self._mil_boxes = []

            detections = drone_detect(image, self.config, self._prev_center, self._prev_centers)
            if detections:
                self._prev_centers = []
                for det in detections:
                    x, y, w, h = det["bbox"]
                    self._prev_centers.append((x + w / 2.0, y + h / 2.0))
                self._prev_center = self._prev_centers[0]
                if len(detections) > 1:
                    self._mil_trackers = []
                    self._mil_boxes = []
                    for det in detections[: self.config["drone_max_detections"]]:
                        tracker = cv2.TrackerMIL_create()
                        tracker.init(image, tuple(det["bbox"]))
                        self._mil_trackers.append(tracker)
                        self._mil_boxes.append(list(det["bbox"]))
            return detections
        return classical_detect(image, self.config)

    def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        """Detect ships in a list of frames.

        Returns:
            List of detection lists, one per frame.
        """
        return [self.detect(img) for img in images]
