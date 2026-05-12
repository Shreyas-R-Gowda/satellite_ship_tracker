"""
Multi-object tracker using IoU-based greedy frame-to-frame assignment.

Each ship is assigned a unique Track with a persistent ID, color-coded
trajectory, velocity estimate, and heading angle.  Tracks survive up to
MAX_LOST_FRAMES consecutive missed detections before being retired.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import cv2
from scipy.optimize import linear_sum_assignment

Detection = Dict[str, Any]  # {"bbox": [x, y, w, h], "confidence": float, "class": str}

CONFIG = {
    "iou_threshold":    0.25,   # minimum IoU to associate a detection with a track
    "max_lost_frames":  5,      # frames a track survives without a matching detection
    "min_track_length": 2,      # minimum detections to include a track in reports
    "velocity_window":  5,      # number of recent positions used for velocity estimate
    "trail_length":     20,     # max positions stored for rendering the trail
}

# Visually distinct BGR colours for up to 12 simultaneous tracks.
# OpenCV drawing and video export both use BGR, while visualizer.py converts
# these values to RGB only when plotting with matplotlib.
_TRACK_COLORS = [
    (100, 100, 255), (100, 255, 100), (255, 100, 100),
    (255, 255,  80), (255,  80, 255), ( 80, 255, 255),
    (200, 130,  50), ( 50, 200, 130), (130,  50, 200),
    (220, 180,  60), ( 60, 220, 180), (180,  60, 220),
]


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


def bbox_center(bbox: List[int]) -> Tuple[float, float]:
    """Return the (cx, cy) center of a bbox [x, y, w, h]."""
    return float(bbox[0] + bbox[2] / 2), float(bbox[1] + bbox[3] / 2)


class Track:
    """Single ship track storing bounding box history, velocity, and trail.

    Attributes:
        id:          Unique integer track identifier.
        color:       BGR tuple for rendering this track with OpenCV.
        bboxes:      All bounding boxes [x, y, w, h] seen so far.
        centers:     Corresponding center coordinates.
        frame_ids:   Frame indices where this track was updated.
        lost_frames: Consecutive frames without a matching detection.
        active:      False once the track has been retired.
        start_frame: Frame index when this track was created.
    """

    _id_counter: int = 0

    def __init__(self, detection: Detection, frame_id: int) -> None:
        cx, cy = bbox_center(detection["bbox"])

        Track._id_counter += 1
        self.id          = Track._id_counter
        self.color       = _TRACK_COLORS[(self.id - 1) % len(_TRACK_COLORS)]
        self.bboxes:      List[List[int]]            = [detection["bbox"]]
        self.centers:     List[Tuple[float, float]]  = [(cx, cy)]
        self.confidences: List[float]                = [detection["confidence"]]
        self.frame_ids:   List[int]                  = [frame_id]
        self.lost_frames: int                        = 0
        self.active:      bool                       = True
        self.start_frame: int                        = frame_id

        self._kf = cv2.KalmanFilter(4, 2)
        self._kf.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self._kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self._kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        self._predicted_center: Tuple[float, float] = (cx, cy)

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------

    def update(self, detection: Detection, frame_id: int) -> None:
        """Attach a new matched detection to this track."""
        cx, cy = bbox_center(detection["bbox"])
        self.kalman_correct(cx, cy)
        self.bboxes.append(detection["bbox"])
        self.centers.append((cx, cy))
        self.confidences.append(detection["confidence"])
        self.frame_ids.append(frame_id)
        self.lost_frames = 0

    def mark_lost(self) -> None:
        """Increment the lost-frame counter (called when no detection matched)."""
        self.lost_frames += 1

    def kalman_predict(self) -> Tuple[float, float]:
        """Predict the next center position using the Kalman motion model."""
        pred = self._kf.predict()
        cx = float(pred[0, 0])
        cy = float(pred[1, 0])
        self._predicted_center = (cx, cy)
        return self._predicted_center

    def kalman_correct(self, cx: float, cy: float) -> None:
        """Correct the Kalman state with a measured center position."""
        self._kf.correct(np.array([[cx], [cy]], dtype=np.float32))

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def last_bbox(self) -> List[int]:
        return self.bboxes[-1]

    @property
    def last_center(self) -> Tuple[float, float]:
        return self.centers[-1]

    @property
    def predicted_center(self) -> Tuple[float, float]:
        return self._predicted_center

    def predicted_bbox(self) -> List[int]:
        """Return the last bbox size centered on the Kalman-predicted center."""
        px, py = self._predicted_center
        _, _, w, h = self.last_bbox
        return [int(px - w / 2), int(py - h / 2), w, h]

    def velocity(self, window: int = CONFIG["velocity_window"]) -> Tuple[float, float]:
        """Estimate velocity (pixels/frame) from the last `window` center positions.

        Returns (vx, vy).  Returns (0, 0) if fewer than 2 positions exist.
        """
        if len(self.centers) < 2:
            return 0.0, 0.0
        recent = self.centers[-min(window, len(self.centers)):]
        n = len(recent) - 1
        if n == 0:
            return 0.0, 0.0
        vx = (recent[-1][0] - recent[0][0]) / n
        vy = (recent[-1][1] - recent[0][1]) / n
        return vx, vy

    def speed(self) -> float:
        """Scalar speed in pixels per frame."""
        vx, vy = self.velocity()
        return float(np.hypot(vx, vy))

    def speed_knots(self, gsd_m_per_px: float, fps: int) -> float:
        """Convert pixel/frame speed to knots.

        gsd_m_per_px: ground sample distance in metres per pixel
        fps: frames per second
        """
        speed_m_per_s = self.speed() * gsd_m_per_px * fps
        return speed_m_per_s / 0.51444   # 1 knot = 0.51444 m/s

    def heading(self) -> float:
        """Movement heading in degrees (0° = right, 90° = down, per screen coords)."""
        vx, vy = self.velocity()
        return float(np.degrees(np.arctan2(vy, vx)))

    def trail(self, length: int = CONFIG["trail_length"]) -> List[Tuple[float, float]]:
        """Return the most recent `length` center positions for trail rendering."""
        return self.centers[-length:]


class MultiObjectTracker:
    """IoU-based greedy multi-object tracker.

    At each frame:
      1. Build the IoU matrix between existing tracks and new detections.
      2. Greedily assign the highest-IoU pairs until the threshold is exhausted.
      3. Unmatched tracks accrue lost-frame counts; retired after MAX_LOST_FRAMES.
      4. Unmatched detections spawn new tracks.
    """

    def __init__(self, config: dict = CONFIG) -> None:
        self.config           = config
        self.tracks:          List[Track] = []
        self.finished_tracks: List[Track] = []
        Track._id_counter = 0   # reset IDs for each new tracker instance

    def update(self, detections: List[Detection], frame_id: int) -> List[Track]:
        """Process one frame of detections and return currently visible tracks.

        Args:
            detections: List of Detection dicts from the ship detector.
            frame_id:   Zero-based index of the current frame.

        Returns:
            List of Track objects that were matched in this frame (lost_frames == 0).
        """
        for track in self.tracks:
            track.kalman_predict()

        matched_track_idx: set = set()
        matched_det_idx:   set = set()

        # ---- Hungarian IoU assignment -------------------------------------
        if self.tracks and detections:
            iou_mat = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    track_bbox = track.predicted_bbox() if track.lost_frames > 0 else track.last_bbox
                    iou_mat[i, j] = compute_iou(track_bbox, det["bbox"])

            cost_mat = 1.0 - iou_mat
            cost_mat[iou_mat < self.config["iou_threshold"]] = 1.0
            row_ind, col_ind = linear_sum_assignment(cost_mat)

            for i, j in zip(row_ind, col_ind):
                if cost_mat[i, j] < (1.0 - self.config["iou_threshold"]):
                    self.tracks[i].update(detections[j], frame_id)
                    matched_track_idx.add(i)
                    matched_det_idx.add(j)

        # ---- Handle unmatched tracks --------------------------------------
        for i, track in enumerate(self.tracks):
            if i not in matched_track_idx:
                track.mark_lost()

        # ---- Spawn new tracks for unmatched detections --------------------
        for j, det in enumerate(detections):
            if j not in matched_det_idx:
                self.tracks.append(Track(det, frame_id))

        # ---- Retire dead tracks ------------------------------------------
        alive: List[Track] = []
        for track in self.tracks:
            if track.lost_frames > self.config["max_lost_frames"]:
                track.active = False
                self.finished_tracks.append(track)
            else:
                alive.append(track)
        self.tracks = alive

        return [t for t in self.tracks if t.lost_frames == 0]

    def all_tracks(self) -> List[Track]:
        """Return every track (active + finished) that meets the minimum length."""
        combined = self.tracks + self.finished_tracks
        return [t for t in combined if len(t.centers) >= self.config["min_track_length"]]
