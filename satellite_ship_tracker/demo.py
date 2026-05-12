"""
Demo: synthetic satellite data generator + full end-to-end tracking pipeline.

Run with:
    python demo.py

Produces in output/:
  demo_annotated.mp4   — annotated video with boxes, IDs, trails
  demo_raw.mp4         — clean video (no annotations)
  tracking_report.txt  — per-ship stats
  trajectory_map.png   — top-down trajectory visualisation
"""

import sys
from pathlib import Path

# Allow imports from the project root regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from typing import List, Tuple

from src.preprocessor  import preprocess_frames
from src.ship_detector import ShipDetector
from src.tracker       import MultiObjectTracker
from src.video_builder import write_video, export_frame_pngs
from src.visualizer    import annotate_frame, save_trajectory_map

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("output")

# ── Synthetic generation config ───────────────────────────────────────────────
SYNTH_CONFIG = {
    "num_frames":  25,
    "frame_size":  (800, 800),   # height × width
    "num_ships":   3,
    "fps":         8,
    "noise_std":   0.15,         # ship motion noise per frame (pixels)
}


# ── Synthetic ocean texture ───────────────────────────────────────────────────

def _layered_noise(h: int, w: int, rng: np.random.Generator, seed_offset: int = 0) -> np.ndarray:
    """Generate a single layer of spatially-correlated noise."""
    layer = rng.random((h, w)).astype(np.float32)
    return cv2.GaussianBlur(layer, (0, 0), 25 + seed_offset * 3)


def generate_water_texture(h: int, w: int, frame_idx: int) -> np.ndarray:
    """Generate a realistic ocean surface texture for one frame.

    Uses layered Gaussian-blurred noise at different scales to mimic
    satellite imagery of open water.  frame_idx shifts the seed slightly
    to introduce subtle animation between frames.

    Returns:
        Grayscale uint8 image with pixel values in [15, 75].
    """
    rng = np.random.default_rng(seed=frame_idx * 7 + 42)
    base  = _layered_noise(h, w, rng, 0)
    mid   = _layered_noise(h, w, rng, 3) * 0.35
    fine  = _layered_noise(h, w, rng, 6) * 0.12
    tex   = base + mid + fine
    # Normalise then map to a dark-ocean luminance range
    tex   = (tex - tex.min()) / (tex.max() - tex.min() + 1e-8)
    return (tex * 55 + 15).astype(np.uint8)


# ── Synthetic ship simulation ─────────────────────────────────────────────────

class SyntheticShip:
    """A single simulated ship with constant velocity and optional spawn edge.

    Ships start from one of the four frame edges so they appear to sail in
    from outside the field of view, cross the scene, and exit the other side.
    """

    def __init__(
        self,
        ship_id: int,
        frame_hw: Tuple[int, int],
        rng: np.random.Generator,
    ) -> None:
        h, w = frame_hw
        edge = rng.integers(0, 4)          # 0=top 1=bottom 2=left 3=right

        if edge == 0:   # enters from the top, moves downward
            self.cx, self.cy = float(rng.integers(80, w - 80)), float(rng.integers(-30, 10))
            self.vx = float(rng.uniform(-1.2, 1.2))
            self.vy = float(rng.uniform(2.0, 5.0))
        elif edge == 1: # enters from the bottom, moves upward
            self.cx, self.cy = float(rng.integers(80, w - 80)), float(rng.integers(h - 10, h + 30))
            self.vx = float(rng.uniform(-1.2, 1.2))
            self.vy = float(rng.uniform(-5.0, -2.0))
        elif edge == 2: # enters from the left, moves rightward
            self.cx, self.cy = float(rng.integers(-30, 10)), float(rng.integers(80, h - 80))
            self.vx = float(rng.uniform(2.0, 5.0))
            self.vy = float(rng.uniform(-1.2, 1.2))
        else:           # enters from the right, moves leftward
            self.cx, self.cy = float(rng.integers(w - 10, w + 30)), float(rng.integers(80, h - 80))
            self.vx = float(rng.uniform(-5.0, -2.0))
            self.vy = float(rng.uniform(-1.2, 1.2))

        # Physical dimensions vary per ship
        self.length     = int(rng.integers(38, 58))
        self.width      = int(rng.integers(8,  15))
        self.brightness = int(rng.integers(195, 245))

        # Heading matches velocity direction
        self.angle_deg  = float(np.degrees(np.arctan2(self.vy, self.vx)))
        self.rng        = rng
        self.noise_std  = SYNTH_CONFIG["noise_std"]

    def step(self) -> None:
        """Advance the ship by one frame with small Gaussian motion noise."""
        self.cx += self.vx + self.rng.normal(0, self.noise_std)
        self.cy += self.vy + self.rng.normal(0, self.noise_std)

    def is_visible(self, frame_hw: Tuple[int, int], margin: int = 80) -> bool:
        """Return True while the ship center is within the extended frame boundary."""
        h, w = frame_hw
        return -margin < self.cx < w + margin and -margin < self.cy < h + margin


def _draw_ship(
    frame: np.ndarray,
    ship: SyntheticShip,
) -> np.ndarray:
    """Render one ship as a bright rotated rectangle with a faint wake."""
    cx, cy = ship.cx, ship.cy

    # Main hull
    rect = ((cx, cy), (ship.length, ship.width), ship.angle_deg)
    hull = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(frame, [hull], int(ship.brightness))

    # Wake — pale elongated region trailing behind the ship
    wake_len = int(ship.length * 0.6)
    angle_rad = np.radians(ship.angle_deg + 180)
    wx = cx + np.cos(angle_rad) * wake_len
    wy = cy + np.sin(angle_rad) * wake_len
    wake_rect = ((wx, wy), (wake_len, max(2, ship.width // 3)), ship.angle_deg)
    wake_pts  = cv2.boxPoints(wake_rect).astype(np.int32)
    cv2.fillPoly(frame, [wake_pts], int(ship.brightness * 0.38))

    return frame


def generate_synthetic_frames(config: dict = SYNTH_CONFIG) -> List[np.ndarray]:
    """Generate a temporal sequence of synthetic satellite ocean frames.

    Each frame contains an animated water texture with multiple ship-like
    blobs moving along straight trajectories with small random noise.

    Args:
        config: Synthetic data generation parameters.

    Returns:
        List of BGR frames ready for the preprocessing pipeline.
    """
    rng    = np.random.default_rng(seed=99)
    h, w   = config["frame_size"]
    ships  = [SyntheticShip(i, (h, w), rng) for i in range(config["num_ships"])]
    frames: List[np.ndarray] = []

    for frame_idx in range(config["num_frames"]):
        gray = generate_water_texture(h, w, frame_idx)

        for ship in ships:
            if ship.is_visible((h, w)):
                gray = _draw_ship(gray, ship)
            ship.step()

        # Convert to BGR so the rest of the pipeline handles it uniformly
        frames.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    return frames


def save_synthetic_frames(frames: List[np.ndarray], data_dir: Path) -> None:
    """Persist generated frames as numbered PNGs in the data directory."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames, start=1):
        cv2.imwrite(str(data_dir / f"demo_frame_{i:03d}.png"), frame)
    print(f"Synthetic frames saved → {data_dir}/  ({len(frames)} frames)")


# ── Reporting ─────────────────────────────────────────────────────────────────

def _write_tracking_report(all_tracks: list, output_path: str, fps: int) -> None:
    """Write a plain-text per-ship tracking summary."""
    with open(output_path, "w") as fh:
        fh.write("=" * 50 + "\n")
        fh.write("  Satellite Ship Tracking System — Report\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(f"Total ships tracked : {len(all_tracks)}\n")
        fh.write(f"Output FPS          : {fps}\n\n")

        for track in sorted(all_tracks, key=lambda t: t.id):
            fh.write(f"Ship #{track.id}\n")
            fh.write(f"  Frames active : {track.frame_ids[0]} → {track.frame_ids[-1]}"
                     f" ({len(track.centers)} detections)\n")
            fh.write(f"  Avg speed     : {track.speed():.2f} px/frame"
                     f"  ({track.speed() * fps:.1f} px/sec)\n")
            sx, sy = track.centers[0]
            ex, ey = track.centers[-1]
            fh.write(f"  Start position: ({sx:.1f}, {sy:.1f})\n")
            fh.write(f"  End position  : ({ex:.1f}, {ey:.1f})\n")
            fh.write(f"  Heading       : {track.heading():.1f}°\n\n")

    print(f"Report saved       → {output_path}")


# ── Main demo pipeline ────────────────────────────────────────────────────────

def run_demo() -> None:
    """Execute the full demo pipeline end-to-end.

    Steps:
      1. Generate synthetic ocean frames with moving ships.
      2. Save frames to data/ as demo_frame_NNN.png.
      3. Preprocess: resize + CLAHE + frame alignment.
      4. Detect ships in each frame (classical CV).
      5. Track ships across frames (IoU-based greedy tracker).
      6. Annotate frames and export annotated + raw videos.
      7. Save per-frame PNGs, tracking report, and trajectory map.
    """
    print("\n" + "=" * 55)
    print("  Satellite Ship Tracking System — Demo")
    print("=" * 55 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fps = SYNTH_CONFIG["fps"]

    # ── 1. Generate synthetic frames ─────────────────────────────────────
    print("[1/6] Generating synthetic satellite frames …")
    raw_frames = generate_synthetic_frames(SYNTH_CONFIG)
    save_synthetic_frames(raw_frames, DATA_DIR)

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    print("[2/6] Preprocessing (CLAHE + frame alignment) …")
    processed = preprocess_frames(raw_frames, align=True)

    # ── 3. Detect ships ──────────────────────────────────────────────────
    print("[3/6] Detecting ships (classical CV) …")
    detector       = ShipDetector(mode="classical")
    all_detections = detector.detect_batch(processed)
    total_det      = sum(len(d) for d in all_detections)
    print(f"      Detections across all frames: {total_det}")

    # ── 4. Track ships ────────────────────────────────────────────────────
    print("[4/6] Running multi-object tracker …")
    mot = MultiObjectTracker()
    frame_active: List[list] = []
    for fid, dets in enumerate(all_detections):
        active = mot.update(dets, fid)
        frame_active.append(active)
    all_tracks = mot.all_tracks()
    print(f"      Unique ship tracks found: {len(all_tracks)}")

    # ── 5. Annotate frames ────────────────────────────────────────────────
    print("[5/6] Annotating frames …")
    annotated: List[np.ndarray] = []
    for fid, (frame, active) in enumerate(zip(processed, frame_active)):
        annotated.append(annotate_frame(frame, active, fid))

    # ── 6. Export outputs ─────────────────────────────────────────────────
    print("[6/6] Exporting outputs …")
    write_video(annotated,  str(OUTPUT_DIR / "demo_annotated.mp4"), fps=fps)
    write_video(processed,  str(OUTPUT_DIR / "demo_raw.mp4"),       fps=fps)
    export_frame_pngs(annotated, str(OUTPUT_DIR / "frames"), prefix="annotated")

    _write_tracking_report(all_tracks, str(OUTPUT_DIR / "tracking_report.txt"), fps)

    h, w = SYNTH_CONFIG["frame_size"]
    save_trajectory_map(all_tracks, str(OUTPUT_DIR / "trajectory_map.png"), frame_shape=(h, w))

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "─" * 45)
    print("  Done!  Outputs written to output/")
    print("─" * 45)
    print("  demo_annotated.mp4   — annotated video")
    print("  demo_raw.mp4         — clean video")
    print("  frames/              — per-frame PNGs")
    print("  tracking_report.txt  — per-ship stats")
    print("  trajectory_map.png   — trajectory visualisation")
    print()


if __name__ == "__main__":
    run_demo()
