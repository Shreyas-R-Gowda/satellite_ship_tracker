"""
Video construction and export from processed frames.

Builds MP4 files via OpenCV VideoWriter and can also export per-frame PNGs.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

CONFIG = {
    "fps":        8,
    "codec":      "mp4v",
    "frame_size": (800, 800),   # (width, height)
}


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """Normalise any frame to BGR for VideoWriter / cv2.imwrite.

    - Grayscale (H×W)      -> 3-channel BGR via GRAY2BGR.
    - 3-channel BGR frame  -> returned unchanged.

    OpenCV reads, draws, and writes in BGR order, so the pipeline keeps that
    convention end-to-end.
    """
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def create_writer(
    output_path: str,
    frame_size: Tuple[int, int],
    fps: int   = CONFIG["fps"],
    codec: str = CONFIG["codec"],
) -> cv2.VideoWriter:
    """Initialise and return an OpenCV VideoWriter.

    Args:
        output_path: Destination .mp4 file path.
        frame_size:  (width, height) in pixels.
        fps:         Output frames per second.
        codec:       FourCC codec string (e.g. 'mp4v', 'avc1').

    Returns:
        Opened cv2.VideoWriter instance.
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(
            f"VideoWriter failed to open '{output_path}'. "
            "Check that the codec is supported and the output directory exists."
        )
    return writer


def write_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int   = CONFIG["fps"],
    codec: str = CONFIG["codec"],
) -> None:
    """Write a list of frames to an MP4 video file.

    Args:
        frames:      Ordered list of BGR (or grayscale) frames.
        output_path: Destination path for the .mp4 file.
        fps:         Playback frame rate.
        codec:       FourCC codec string.
    """
    if not frames:
        raise ValueError("write_video: received empty frames list.")

    first = _ensure_bgr(frames[0])
    h, w  = first.shape[:2]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = create_writer(output_path, (w, h), fps, codec)
    for frame in frames:
        writer.write(_ensure_bgr(frame))
    writer.release()
    print(f"Video saved  → {output_path}  ({len(frames)} frames @ {fps} FPS)")


def write_comparison_video(
    raw_frames: List[np.ndarray],
    annotated_frames: List[np.ndarray],
    output_path: str,
    fps: int = 8,
    codec: str = CONFIG["codec"],
    label_left: str = "Raw",
    label_right: str = "Tracked",
) -> None:
    """Write a side-by-side raw/tracked comparison MP4."""
    if len(raw_frames) != len(annotated_frames):
        raise ValueError("write_comparison_video: frame lists must have the same length.")
    if not raw_frames:
        raise ValueError("write_comparison_video: received empty frames list.")

    combined_frames: List[np.ndarray] = []
    for raw, annotated in zip(raw_frames, annotated_frames):
        left = _ensure_bgr(raw)
        right = _ensure_bgr(annotated)
        combined = np.concatenate([left, right], axis=1)
        left_w = left.shape[1]

        cv2.line(combined, (left_w, 0), (left_w, combined.shape[0]), (255, 255, 255), 2)
        cv2.putText(
            combined,
            label_left,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            combined,
            label_right,
            (left_w + 10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        combined_frames.append(combined)

    first = combined_frames[0]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = create_writer(output_path, (first.shape[1], first.shape[0]), fps, codec)
    for frame in combined_frames:
        writer.write(frame)
    writer.release()
    print(f"Comparison video → {output_path}  ({len(raw_frames)} frames @ {fps} FPS)")


def export_frame_pngs(
    frames: List[np.ndarray],
    output_dir: str,
    prefix: str = "frame",
) -> None:
    """Save each frame as a numbered PNG file.

    Args:
        frames:     Ordered list of BGR (or grayscale) frames.
        output_dir: Directory to write PNG files into (created if absent).
        prefix:     Filename prefix; files are named {prefix}_001.png etc.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames, start=1):
        path = out_dir / f"{prefix}_{i:03d}.png"
        cv2.imwrite(str(path), _ensure_bgr(frame))

    print(f"Frames saved → {output_dir}/  ({len(frames)} PNGs)")
