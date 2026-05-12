"""
Multi-object tracker using IoU-based greedy frame-to-frame assignment.

Each ship is assigned a unique Track with a persistent ID, color-coded
trajectory, velocity estimate, and heading angle.  Tracks survive up to
MAX_LOST_FRAMES consecutive missed detections before being retired.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

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
        Track._id_counter += 1
        self.id          = Track._id_counter
        self.color       = _TRACK_COLORS[(self.id - 1) % len(_TRACK_COLORS)]
        self.bboxes:      List[List[int]]            = [detection["bbox"]]
        self.centers:     List[Tuple[float, float]]  = [bbox_center(detection["bbox"])]
        self.confidences: List[float]                = [detection["confidence"]]
        self.frame_ids:   List[int]                  = [frame_id]
        self.lost_frames: int                        = 0
        self.active:      bool                       = True
        self.start_frame: int                        = frame_id

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------

    def update(self, detection: Detection, frame_id: int) -> None:
        """Attach a new matched detection to this track."""
        self.bboxes.append(detection["bbox"])
        self.centers.append(bbox_center(detection["bbox"]))
        self.confidences.append(detection["confidence"])
        self.frame_ids.append(frame_id)
        self.lost_frames = 0

    def mark_lost(self) -> None:
        """Increment the lost-frame counter (called when no detection matched)."""
        self.lost_frames += 1

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def last_bbox(self) -> List[int]:
        return self.bboxes[-1]

    @property
    def last_center(self) -> Tuple[float, float]:
        return self.centers[-1]

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
        matched_track_idx: set = set()
        matched_det_idx:   set = set()

        # ---- Greedy IoU assignment ----------------------------------------
        if self.tracks and detections:
            iou_mat = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    iou_mat[i, j] = compute_iou(track.last_bbox, det["bbox"])

            while iou_mat.max() >= self.config["iou_threshold"]:
                i, j = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
                self.tracks[i].update(detections[j], frame_id)
                matched_track_idx.add(i)
                matched_det_idx.add(j)
                iou_mat[i, :] = 0.0
                iou_mat[:, j] = 0.0

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
