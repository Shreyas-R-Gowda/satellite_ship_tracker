"""
Satellite Ship Tracking System — CLI entry point.

Usage examples:
    python main.py --input data/ --output output/ --fps 8 --detector classical
    python main.py --input data/ --output output/ --fps 5 --detector yolo --save-frames
    python main.py --input data/ --no-align
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessor  import load_images, preprocess_frames
from src.ship_detector import ShipDetector
from src.tracker       import MultiObjectTracker
from src.video_builder import write_video, export_frame_pngs
from src.visualizer    import annotate_frame, save_trajectory_map


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Satellite Ship Tracking System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",       default="data/",      help="Folder containing satellite images")
    p.add_argument("--output",      default="output/",    help="Folder for all output files")
    p.add_argument("--fps",         type=int, default=8,  help="Output video frames per second")
    p.add_argument("--detector",    choices=["classical", "yolo"], default="classical",
                   help="Detection backend (classical CV or YOLOv8)")
    p.add_argument("--no-align",    action="store_true",  help="Skip ORB frame alignment")
    p.add_argument("--save-frames", action="store_true",  help="Export per-frame annotated PNGs")
    p.add_argument("--trail",       type=int, default=20, help="Trail length (frames) for visualisation")
    p.add_argument("--tile",        action="store_true",
                   help="Tile images larger than 1024x1024 before detection")
    return p.parse_args()


def _write_report(all_tracks: list, output_path: str, meta: dict) -> None:
    """Write per-ship summary to a plain-text file."""
    with open(output_path, "w") as fh:
        fh.write("=" * 50 + "\n")
        fh.write("  Satellite Ship Tracking Report\n")
        fh.write("=" * 50 + "\n\n")
        for k, v in meta.items():
            fh.write(f"{k:<22}: {v}\n")
        fh.write(f"{'Total ships tracked':<22}: {len(all_tracks)}\n\n")

        for track in sorted(all_tracks, key=lambda t: t.id):
            fh.write(f"Ship #{track.id}\n")
            fh.write(f"  Frames active : {track.frame_ids[0]} → {track.frame_ids[-1]}"
                     f" ({len(track.centers)} detections)\n")
            fh.write(f"  Avg speed     : {track.speed():.2f} px/frame\n")
            sx, sy = track.centers[0]
            ex, ey = track.centers[-1]
            fh.write(f"  Start position: ({sx:.1f}, {sy:.1f})\n")
            fh.write(f"  End position  : ({ex:.1f}, {ey:.1f})\n")
            fh.write(f"  Heading       : {track.heading():.1f}°\n\n")

    print(f"Report saved       → {output_path}")


def main() -> None:
    args       = parse_args()
    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"\nLoading images from {input_dir} …")
    images, names = load_images(str(input_dir))
    if not images:
        print(
            f"  No images found in '{input_dir}'.\n"
            "  Run  python demo.py  first to generate synthetic data."
        )
        sys.exit(1)
    print(f"  Loaded {len(images)} images: {names[0]} … {names[-1]}")

    # ── Preprocess ────────────────────────────────────────────────────────
    print("Preprocessing frames …")
    processed = preprocess_frames(images, align=not args.no_align)

    # ── Detect ────────────────────────────────────────────────────────────
    print(f"Detecting ships using '{args.detector}' detector …")
    detector = ShipDetector(mode=args.detector)
    from src.preprocessor import tile_large_image
    from src.ship_detector import merge_tile_detections
    all_detections = []
    for frame in processed:
        h, w = frame.shape[:2]
        if args.tile and (h > 1024 or w > 1024):
            tiles, tile_bboxes = tile_large_image(frame)
            tile_dets = [detector.detect(t) for t in tiles]
            dets = merge_tile_detections(tile_dets, tile_bboxes)
        else:
            dets = detector.detect(frame)
        all_detections.append(dets)
    total_det      = sum(len(d) for d in all_detections)
    print(f"  Total detections across all frames: {total_det}")

    # ── Track ─────────────────────────────────────────────────────────────
    print("Tracking ships across frames …")
    mot           = MultiObjectTracker()
    frame_active  = []
    for fid, dets in enumerate(all_detections):
        active = mot.update(dets, fid)
        frame_active.append(active)
    all_tracks = mot.all_tracks()
    print(f"  Unique ship tracks: {len(all_tracks)}")

    # ── Annotate ──────────────────────────────────────────────────────────
    print("Annotating frames …")
    annotated = [
        annotate_frame(frame, active, fid, trail_length=args.trail)
        for fid, (frame, active) in enumerate(zip(processed, frame_active))
    ]

    # ── Export ────────────────────────────────────────────────────────────
    print("Exporting …")
    write_video(annotated,  str(output_dir / "annotated.mp4"), fps=args.fps)
    write_video(processed,  str(output_dir / "raw.mp4"),       fps=args.fps)

    if args.save_frames:
        export_frame_pngs(annotated, str(output_dir / "frames"))

    h, w = processed[0].shape[:2]
    save_trajectory_map(all_tracks, str(output_dir / "trajectory_map.png"), frame_shape=(h, w))

    _write_report(
        all_tracks,
        str(output_dir / "tracking_report.txt"),
        meta={
            "Input folder":  str(input_dir),
            "Frames":        len(images),
            "Detector":      args.detector,
            "FPS":           args.fps,
            "Frame align":   not args.no_align,
        },
    )

    print(f"\nDone!  All outputs written to {output_dir}/\n")


if __name__ == "__main__":
    main()
