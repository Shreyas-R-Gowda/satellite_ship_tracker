# Satellite Ship Tracker - Presentation Notes

## Problem Statement

Develop a system that converts optical satellite images into a video-like
sequence and tracks ship movement across the frames.

## Input

The system accepts a time-ordered folder of optical satellite images.

In this demo, the input images are generated synthetically:

- Folder: `data/`
- Format: numbered PNG frames, for example `demo_frame_001.png`
- Frame size: `800 x 800`
- Demo sequence length: `25` frames
- Simulated objects: `3` moving ships

For real data, the same pipeline can process `.jpg`, `.jpeg`, `.png`, `.tif`,
and `.tiff` files using:

```bash
python3 main.py --input data/ --output output/ --fps 8 --detector classical --save-frames
```

## Processing Pipeline

1. Load and sort satellite images by filename.
2. Resize frames and enhance contrast using CLAHE.
3. Optionally align frames using ORB feature matching and homography.
4. Detect ships using classical computer vision:
   adaptive thresholding, morphology, and contour filtering.
5. Track ships across frames using an IoU-based multi-object tracker.
6. Render annotations and export the simulated video outputs.

## Output

The project generates these outputs in `output/`:

- `demo_raw.mp4`: simulated clean satellite video created from image frames.
- `demo_annotated.mp4`: tracking video with ship IDs, boxes, trails, velocity
  arrows, and speed labels.
- `frames/`: annotated frame-by-frame PNG output.
- `trajectory_map.png`: top-down visualization of each ship path.
- `tracking_report.txt`: per-ship movement summary.

Current demo result:

- Total frames: `25`
- Output FPS: `8`
- Total ships tracked: `3`
- Output motion units: pixels/frame and pixels/sec

## Presentation Script

This system takes a folder of optical satellite images captured over time and
sorts them into a frame sequence. The frames are preprocessed to improve
contrast and reduce camera drift. Each frame is then analyzed to detect bright,
elongated ship-like objects on the ocean surface. The tracker links detections
across frames, assigns each ship a unique ID, and estimates its movement using
center-point history. Finally, the system exports a simulated video, an
annotated tracking video, a trajectory map, and a text report with speed,
heading, and start/end positions.

## Note

The current demo estimates speed in image-space units, not real-world nautical
units. To convert pixels/sec into meters/sec or knots, the system would need
georeferencing information such as ground sample distance and real timestamps.
