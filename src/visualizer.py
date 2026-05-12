"""
Annotation rendering for satellite ship tracking output.

Draws bounding boxes, trajectory trails, velocity arrows, ship labels,
frame overlays, and a legend onto BGR frames.  Also generates a top-down
trajectory map using matplotlib.
"""

import os
from pathlib import Path

import cv2
import numpy as np

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_DIR / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg"))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from typing import Any, List, Optional, Tuple


# ── Low-level drawing primitives ──────────────────────────────────────────────

def draw_bbox(
    image: np.ndarray,
    bbox: List[int],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """Draw a filled-corner bounding box around a detected ship."""
    x, y, w, h = bbox
    corner = max(8, min(w, h) // 3)

    # Corner accent lines instead of a full rectangle — cleaner look
    for (px, py, dx, dy) in [
        (x,     y,     1,  1),
        (x + w, y,    -1,  1),
        (x,     y + h,  1, -1),
        (x + w, y + h, -1, -1),
    ]:
        cv2.line(image, (px, py), (px + dx * corner, py), color, thickness)
        cv2.line(image, (px, py), (px, py + dy * corner), color, thickness)

    # Light full rectangle for context
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
    return image


def draw_trail(
    image: np.ndarray,
    centers: List[Tuple[float, float]],
    color: Tuple[int, int, int],
    max_thickness: int = 3,
) -> np.ndarray:
    """Draw a fading polyline trail from oldest to newest position."""
    n = len(centers)
    if n < 2:
        return image
    for i in range(1, n):
        alpha     = i / n
        fade      = tuple(int(c * alpha) for c in color)
        thickness = max(1, int(max_thickness * alpha))
        p1 = (int(centers[i - 1][0]), int(centers[i - 1][1]))
        p2 = (int(centers[i][0]),     int(centers[i][1]))
        cv2.line(image, p1, p2, fade, thickness)
    return image


def draw_velocity_arrow(
    image: np.ndarray,
    center: Tuple[float, float],
    velocity: Tuple[float, float],
    color: Tuple[int, int, int],
    scale: float = 6.0,
) -> np.ndarray:
    """Draw a velocity vector arrow from the ship center."""
    cx, cy = int(center[0]), int(center[1])
    vx, vy = velocity
    ex, ey = int(cx + vx * scale), int(cy + vy * scale)
    if (ex, ey) != (cx, cy):
        cv2.arrowedLine(image, (cx, cy), (ex, ey), color, 2, tipLength=0.35)
    return image


def draw_ship_label(
    image: np.ndarray,
    bbox: List[int],
    track_id: int,
    speed: float,
    color: Tuple[int, int, int],
) -> np.ndarray:
    """Draw a pill-shaped label above the bounding box showing ID and speed."""
    x, y, w, _ = bbox
    label = f"#{track_id}  {speed:.1f}px/f"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
    pad = 3
    lx, ly = x, max(0, y - th - baseline - pad * 2)
    cv2.rectangle(image, (lx, ly), (lx + tw + pad * 2, ly + th + baseline + pad * 2), color, -1)
    cv2.putText(image, label, (lx + pad, ly + th + pad), font, scale, (0, 0, 0), thick)
    return image


def draw_frame_overlay(
    image: np.ndarray,
    frame_id: int,
    active_count: int,
    timestamp: Optional[str] = None,
) -> np.ndarray:
    """Draw a semi-transparent HUD showing frame number and ship count."""
    text = f"Frame {frame_id:03d}  |  Ships tracked: {active_count}"
    if timestamp:
        text += f"  |  {timestamp}"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    # Dark background bar
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (tw + 16, th + 16), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    cv2.putText(image, text, (8, th + 8), font, scale, (230, 230, 230), thick)
    return image


def draw_legend(
    image: np.ndarray,
    track_info: List[Tuple[int, Tuple[int, int, int]]],
) -> np.ndarray:
    """Draw a compact legend panel in the bottom-right corner."""
    if not track_info:
        return image
    h, w = image.shape[:2]
    row_h, pad, swatch = 20, 6, 14
    panel_h = len(track_info) * row_h + pad * 3 + 16
    panel_w = 148
    px = w - panel_w - 6
    py = h - panel_h - 6

    overlay = image.copy()
    cv2.rectangle(overlay, (px, py), (w - 6, h - 6), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    cv2.putText(image, "Active Ships", (px + pad, py + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    for i, (tid, color) in enumerate(track_info):
        ry = py + 22 + i * row_h
        cv2.rectangle(image, (px + pad, ry), (px + pad + swatch, ry + swatch - 2), color, -1)
        cv2.putText(image, f"Ship #{tid}", (px + pad + swatch + 6, ry + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
    return image


# ── High-level annotation ─────────────────────────────────────────────────────

def annotate_frame(
    image: np.ndarray,
    active_tracks: List[Any],
    frame_id: int,
    trail_length: int = 20,
) -> np.ndarray:
    """Compose all annotations onto a single frame.

    Draws: trails, bounding boxes, velocity arrows, ship labels, HUD, legend.

    Args:
        image:         Preprocessed BGR frame.
        active_tracks: Track objects visible in this frame.
        frame_id:      Zero-based frame index.
        trail_length:  How many historical positions to show in the trail.

    Returns:
        Annotated copy of the frame.
    """
    out = image.copy()

    legend_items: List[Tuple[int, Tuple[int, int, int]]] = []
    for track in active_tracks:
        out = draw_trail(out, track.trail(trail_length), track.color)
        out = draw_bbox(out, track.last_bbox, track.color)
        out = draw_velocity_arrow(out, track.last_center, track.velocity(), track.color)
        out = draw_ship_label(out, track.last_bbox, track.id, track.speed(), track.color)
        legend_items.append((track.id, track.color))

    out = draw_frame_overlay(out, frame_id, len(active_tracks))
    out = draw_legend(out, legend_items)
    return out


# ── Trajectory map ────────────────────────────────────────────────────────────

def save_trajectory_map(
    all_tracks: List[Any],
    output_path: str,
    frame_shape: Tuple[int, int],
) -> None:
    """Generate a top-down map of all ship trajectories and save as PNG.

    Start positions are shown as circles, end positions as triangles.

    Args:
        all_tracks:   All Track objects (active + finished) from the tracker.
        output_path:  Destination file path for the PNG.
        frame_shape:  (height, width) of frames, used to set axis limits.
    """
    h, w = frame_shape
    fig, ax = plt.subplots(figsize=(10, 10), dpi=130)
    ax.set_facecolor("#0a1628")
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)   # flip Y so screen-space matches image-space
    ax.set_title("Ship Trajectory Map", color="white", fontsize=15, pad=12)
    ax.set_xlabel("X (pixels)", color="#aaaaaa", fontsize=10)
    ax.set_ylabel("Y (pixels)", color="#aaaaaa", fontsize=10)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#445566")

    # Light grid for distance reference
    ax.grid(color="#1e3050", linestyle="--", linewidth=0.5, alpha=0.6)

    legend_patches: List[mpatches.Patch] = []
    for track in all_tracks:
        # Need at least 2 positions to draw a segment
        if len(track.centers) < 2:
            continue

        b, g, r = track.color
        rgb_norm = (r / 255.0, g / 255.0, b / 255.0)

        xs = [c[0] for c in track.centers]
        ys = [c[1] for c in track.centers]
        n  = len(xs)

        # Build a LineCollection so each segment can carry its own alpha,
        # fading from dim (oldest) to full opacity (newest) to show direction.
        pts      = np.array([xs, ys], dtype=float).T.reshape(-1, 1, 2)
        segments = np.concatenate([pts[:-1], pts[1:]], axis=1)   # shape (n-1, 2, 2)
        alphas   = np.linspace(0.15, 1.0, len(segments))
        seg_colors = [(*rgb_norm, float(a)) for a in alphas]     # RGBA per segment

        lc = LineCollection(segments, colors=seg_colors, linewidths=2.0, zorder=3)
        ax.add_collection(lc)

        # Small dot at every intermediate recorded position (shows tracking cadence)
        if n > 2:
            ax.scatter(xs[1:-1], ys[1:-1], color=rgb_norm,
                       s=14, alpha=0.55, zorder=4, linewidths=0)

        # Distinct start (circle) and end (triangle) markers
        ax.scatter(xs[0],  ys[0],  color=rgb_norm, marker="o",
                   s=80, zorder=6, edgecolors="white", linewidths=0.6)
        ax.scatter(xs[-1], ys[-1], color=rgb_norm, marker="^",
                   s=100, zorder=6, edgecolors="white", linewidths=0.6)

        legend_patches.append(mpatches.Patch(color=rgb_norm, label=f"Ship #{track.id}"))

    if legend_patches:
        leg = ax.legend(
            handles=legend_patches,
            facecolor="#1a2a3a",
            edgecolor="#445566",
            labelcolor="white",
            loc="upper right",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Trajectory map saved → {output_path}")
