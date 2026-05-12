"""Streamlit frontend for the satellite ship tracker."""

import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

from demo import DEMO_GSD_M_PER_PX, SYNTH_CONFIG, generate_synthetic_frames
from src.evaluator import compute_metrics
from src.preprocessor import load_images, preprocess_frames
from src.ship_detector import ShipDetector
from src.tracker import MultiObjectTracker
from src.video_builder import write_video
from src.visualizer import annotate_frame, save_trajectory_map


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR frame for Streamlit display."""
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _tracks_to_pred(all_tracks: list) -> dict:
    pred = {}
    for track in all_tracks:
        for fid, center in zip(track.frame_ids, track.centers):
            pred.setdefault(fid, []).append(center)
    return pred


def _run_pipeline(
    frames: List[np.ndarray],
    fps: int,
    trail_length: int,
    gt: Optional[dict] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], list, Optional[dict]]:
    processed = preprocess_frames(frames, align=True)
    detector = ShipDetector(mode="classical")
    all_detections = detector.detect_batch(processed)

    mot = MultiObjectTracker()
    frame_active = []
    for fid, dets in enumerate(all_detections):
        frame_active.append(mot.update(dets, fid))

    all_tracks = mot.all_tracks()
    annotated = [
        annotate_frame(frame, active, fid, trail_length=trail_length)
        for fid, (frame, active) in enumerate(zip(processed, frame_active))
    ]

    metrics = compute_metrics(gt, _tracks_to_pred(all_tracks)) if gt is not None else None
    return processed, annotated, all_tracks, metrics


def _show_metrics(metrics: Optional[dict]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    if metrics is None:
        c1.metric("Precision", "N/A")
        c2.metric("Recall", "N/A")
        c3.metric("F1", "N/A")
        c4.metric("Avg Error (px)", "N/A")
        return

    c1.metric("Precision", f"{metrics['precision']:.3f}")
    c2.metric("Recall", f"{metrics['recall']:.3f}")
    c3.metric("F1", f"{metrics['f1']:.3f}")
    c4.metric("Avg Error (px)", f"{metrics['avg_center_error_px']:.1f}")


def _tracking_report(all_tracks: list, fps: int) -> str:
    lines = [
        "Satellite Ship Tracking Report",
        "",
        f"Total ships tracked : {len(all_tracks)}",
        f"Output FPS          : {fps}",
        f"GSD assumed         : {DEMO_GSD_M_PER_PX} m/px",
        "",
    ]
    for track in sorted(all_tracks, key=lambda t: t.id):
        sx, sy = track.centers[0]
        ex, ey = track.centers[-1]
        lines.extend([
            f"Ship #{track.id}",
            f"  Frames active : {track.frame_ids[0]} -> {track.frame_ids[-1]} ({len(track.centers)} detections)",
            f"  Avg speed     : {track.speed():.2f} px/frame | {track.speed_knots(DEMO_GSD_M_PER_PX, fps):.2f} knots",
            f"  Start position: ({sx:.1f}, {sy:.1f})",
            f"  End position  : ({ex:.1f}, {ey:.1f})",
            f"  Heading       : {track.heading():.1f} deg",
            "",
        ])
    return "\n".join(lines)


def _save_trajectory(all_tracks: list, frame_shape: Tuple[int, int]) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    save_trajectory_map(all_tracks, tmp.name, frame_shape=frame_shape)
    return tmp.name


def _show_outputs(
    processed: List[np.ndarray],
    annotated: List[np.ndarray],
    all_tracks: list,
    metrics: Optional[dict],
    fps: int,
    show_report: bool = False,
) -> None:
    _show_metrics(metrics)

    if processed:
        h, w = processed[0].shape[:2]
        trajectory_path = _save_trajectory(all_tracks, (h, w))
        st.image(trajectory_path, caption="Trajectory Map")

    if annotated:
        mid = len(annotated) // 2
        st.image(_to_rgb(annotated[mid]), caption=f"Annotated Frame {mid:03d}")

    if show_report:
        st.text(_tracking_report(all_tracks, fps))


st.title("Satellite Ship Tracker")
st.caption("Detects and tracks ships in optical satellite image sequences")

with st.sidebar:
    st.header("Demo Settings")
    num_ships = st.slider("Number of Ships", 1, 8, 3)
    num_frames = st.slider("Number of Frames", 10, 40, 25)
    demo_fps = st.slider("FPS", 2, 15, 8)
    trail_length = st.slider("Trail Length", 5, 40, 20)

tab_demo, tab_upload = st.tabs(["Run Demo", "Upload Images"])

with tab_demo:
    if st.button("Generate & Track"):
        with st.spinner("Running pipeline..."):
            config = {
                **SYNTH_CONFIG,
                "num_ships": num_ships,
                "num_frames": num_frames,
                "fps": demo_fps,
            }
            raw_frames, gt = generate_synthetic_frames(config)
            processed, annotated, all_tracks, metrics = _run_pipeline(
                raw_frames,
                fps=demo_fps,
                trail_length=trail_length,
                gt=gt,
            )
        _show_outputs(processed, annotated, all_tracks, metrics, demo_fps, show_report=True)

with tab_upload:
    st.write("Upload satellite image frames in time order")
    uploaded_files = st.file_uploader(
        "Satellite image frames",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    upload_fps = st.slider("Upload FPS", 2, 15, 8)

    if st.button("Run Tracking"):
        if not uploaded_files:
            st.warning("Please upload at least one image frame.")
        else:
            with st.spinner("Running pipeline..."):
                temp_dir = Path(tempfile.mkdtemp())
                for file in uploaded_files:
                    (temp_dir / file.name).write_bytes(file.getbuffer())

                images, _ = load_images(str(temp_dir))
                processed, annotated, all_tracks, metrics = _run_pipeline(
                    images,
                    fps=upload_fps,
                    trail_length=trail_length,
                    gt=None,
                )

                video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                video_file.close()
                write_video(annotated, video_file.name, fps=upload_fps)
                video_bytes = Path(video_file.name).read_bytes()

            _show_outputs(processed, annotated, all_tracks, metrics, upload_fps)
            st.download_button(
                "Download Annotated Video",
                data=video_bytes,
                file_name="annotated_tracking.mp4",
                mime="video/mp4",
            )
