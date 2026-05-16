# Satellite Ship Tracker

Detects and tracks ships in optical satellite image sequences by converting ordered frames into a simulated video, estimating motion across time, and generating visual and numerical tracking outputs.

## Repository Description

Satellite ship tracking system built with computer vision and multi-object tracking. The project preprocesses optical satellite image frames, detects ship-like objects, tracks them across time using Kalman prediction and Hungarian assignment, and exports annotated videos, trajectory maps, and tracking reports.

## Problem Statement

Satellite imagery is often available as a sequence of still images rather than a directly usable video stream. This project turns those image frames into a video-like sequence, detects ships in each frame, tracks their movement across time, and summarizes the results through visualizations and metrics.

## Data Source

This repository currently uses synthetic optical satellite-style imagery generated in `demo.py`.

- `data/demo_frame_*.png`: default demo sequence
- `data/busy_port/`: denser synthetic scenario with more ships
- `data/open_ocean/`: sparse synthetic scenario with fewer ships

The CLI and Streamlit app also support running on user-provided real image frames in:

- `.png`
- `.jpg`
- `.jpeg`
- `.tif`
- `.tiff`

Important:

- Input frames should be ordered by time.
- Real-world speed estimation requires correct timestamps and ground sample distance metadata.
- The demo uses an assumed GSD of `3.0 m/px`.

## Features

- Synthetic dataset generation for repeatable demos
- Optical image preprocessing with resizing, CLAHE, and optional frame alignment
- Classical ship detection using thresholding, morphology, and contour filtering
- Optional YOLO detector hook
- Multi-object tracking with:
  - persistent ship IDs
  - Kalman filter motion prediction
  - Hungarian assignment for global matching
- Evaluation metrics on synthetic data:
  - precision
  - recall
  - F1 score
  - average center error
- Side-by-side comparison video export
- Large-image tiling support for frames above `1024x1024`
- Streamlit frontend for demo runs and uploaded image sequences

## Project Structure

```text
.
├── app.py                  # Streamlit frontend
├── demo.py                 # Synthetic demo pipeline and dataset generation
├── main.py                 # CLI entry point for image folders
├── requirements.txt
├── data/                   # Synthetic frame sequences
├── output/                 # Generated videos, frames, reports, maps
└── src/
    ├── evaluator.py        # Precision/recall/F1/error metrics
    ├── preprocessor.py     # Loading, CLAHE, alignment, tiling
    ├── ship_detector.py    # Detection and tile-merge logic
    ├── tracker.py          # Kalman + Hungarian multi-object tracker
    ├── video_builder.py    # Raw, annotated, comparison video export
    └── visualizer.py       # Bounding boxes, trails, overlays, map
```

## Architecture

```text
Input Image Frames
        ->
Preprocessing
        ->
Ship Detection
        ->
Multi-Object Tracking
        ->
Metrics and Motion Analysis
        ->
Video / Map / Report Outputs
```

### Data Flow

1. `load_images()` reads the frame sequence from a folder.
2. `preprocess_frames()` resizes, enhances contrast, and optionally aligns frames.
3. `ShipDetector` detects ship-like objects in each frame.
4. `MultiObjectTracker` assigns detections to persistent tracks across frames.
5. `annotate_frame()` draws bounding boxes, labels, trails, and velocity arrows.
6. `write_video()` and `write_comparison_video()` export playback-ready videos.
7. `save_trajectory_map()` and `compute_metrics()` summarize movement and accuracy.

## Outputs

Running the demo produces:

- `output/demo_raw.mp4`: simulated video generated from input frames
- `output/demo_annotated.mp4`: tracked video with IDs, trails, and velocity overlays
- `output/demo_comparison.mp4`: side-by-side raw vs tracked comparison
- `output/trajectory_map.png`: top-down ship path visualization
- `output/tracking_report.txt`: per-ship summary with speed and heading
- `output/frames/`: per-frame annotated PNG exports

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run the synthetic demo

```bash
python3 demo.py
```

This generates the demo sequence, runs tracking, computes evaluation metrics, and exports videos and maps to `output/`.

### Run tracking on an image folder

```bash
python3 main.py --input data/ --output output/ --fps 8 --detector classical --save-frames
```

### Run with tiling support

```bash
python3 main.py --input path/to/frames --output output/ --tile
```

### Launch the Streamlit app

```bash
streamlit run app.py
```

If you are using the local virtual environment directly:

```bash
.venv/bin/streamlit run app.py
```

## Example Demo Metrics

Typical synthetic demo output:

- Precision: `1.000`
- Recall: `0.827`
- F1 Score: `0.905`
- Avg center error: `5.9 px`
- Unique ship tracks: `3`

## Screenshots / Visualization

Good files to include in a presentation or GitHub preview:

- `output/demo_comparison.mp4`
- `output/demo_annotated.mp4`
- `output/trajectory_map.png`
- `output/tracking_report.txt`

Suggested presentation flow:

1. Show one raw input frame from `data/`
2. Show `demo_raw.mp4`
3. Show `demo_annotated.mp4`
4. Show `demo_comparison.mp4`
5. Show `trajectory_map.png`
6. Show `tracking_report.txt`

## Limitations

- The default detector is classical CV and works best when ships are bright and elongated against darker water.
- The demo is synthetic; real satellite data may require stronger detectors and cleaner metadata.
- Speed in knots is only physically meaningful when timestamps and GSD are realistic.
- The current pipeline tracks motion in image space, not latitude/longitude space.

## Future Scope

- Train a YOLO-based detector on real satellite ship datasets
- Add anomaly detection for unusual vessel motion
- Add route forecasting and motion prediction models
- Export structured outputs like CSV or GeoJSON
- Add georeferencing support for latitude/longitude trajectories
- Incorporate cloud/noise robustness for real scenes
- Extend to port congestion analysis and maritime activity monitoring

## Tech Stack

- Python
- OpenCV
- NumPy
- SciPy
- Matplotlib
- Streamlit

## License

Add a license here if you plan to make the repository public for reuse.
