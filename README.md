# Laser Boundary Detector — Zapdos Labs Technical Challenge

A production-quality Computer Vision pipeline that detects and masks the **red laser safety boundary** projected around forklifts in warehouse footage. The output is a clean binary (black/white) mask video at ≥ 2 FPS.

---

## Demo

| Input frame | Binary mask output | Debug overlay |
|:-----------:|:------------------:|:-------------:|
| ![input](assets/demo_input.png) | ![mask](assets/demo_mask.png) | ![debug](assets/demo_debug.png) |

---

## Pipeline Architecture

```
Video Input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  PREPROCESS  (pipeline/preprocess.py)               │
│  • Optional downsample (--scale 0.75)               │
│  • Bilateral filter — denoise, preserve colour edges│
│  • Dual-arc HSV threshold (red hue wraps at 0°)     │
│    Arc-1: H∈[0,10]    S≥160  V≥120                  │
│    Arc-2: H∈[158,180] S≥160  V≥120                  │
│  • Optional auto-calibration from first N frames    │
└────────────────────┬────────────────────────────────┘
                     │  binary red mask
                     ▼
┌─────────────────────────────────────────────────────┐
│  DETECTOR  (pipeline/detector.py)                   │
│  • Morphological cleaning                           │
│    OPEN  → kill salt/speckle noise                  │
│    CLOSE → bridge gaps at laser intersections       │
│    DILATE → fatten blobs for clustering             │
│  • Contour extraction (RETR_EXTERNAL)               │
│  • Area filtering (drop tiny fragments)             │
│  • Spatial clustering (union-find on bbox proximity)│
│    → groups the 4 laser lines into one region       │
│  • minAreaRect on convex hull of each cluster       │
│  • Sanity checks: area, aspect ratio, side length   │
└────────────────────┬────────────────────────────────┘
                     │  raw quad candidates [(4×2) float32]
                     ▼
┌─────────────────────────────────────────────────────┐
│  TRACKER  (pipeline/tracker.py)                     │
│  • IoU-based greedy matching to existing tracks     │
│  • EMA (α=0.25) on corner positions → smooth motion │
│  • False-positive suppression:                      │
│    confirm only after K=3 consecutive detections    │
│  • Occlusion handling:                              │
│    coast up to 8 frames with no detection           │
│  • Multi-rectangle support (one track per laser)    │
└────────────────────┬────────────────────────────────┘
                     │  confirmed, smoothed quads
                     ▼
┌─────────────────────────────────────────────────────┐
│  POSTPROCESS  (pipeline/postprocess.py)             │
│  • Scale quads back to native resolution            │
│  • fillPoly → clean filled binary mask              │
│  • (Optional) debug overlay with magenta tint +     │
│    green fill + corner dots + HUD text              │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
             Output mask video (.mp4)
```

---

## Key Design Decisions

### Why HSV + classical CV, not a neural network?

The laser is a **highly constrained visual stimulus**: bright, fully saturated red, thin lines on a grey/neutral warehouse floor. Classical CV runs at 30+ FPS on a CPU with no GPU requirement, which matters for the edge-device deployment context of a warehouse safety system.

For the rare edge case where HSV fails (unusual lighting, strong reflections), the `--auto-calibrate` flag runs a 5-frame bootstrap that **adapts the bounds to each video's actual laser colour statistics**.

### Why `minAreaRect` instead of `approxPolyDP`?

The laser rectangle is **perspective-warped** (camera is mounted overhead at an angle). `minAreaRect` fitted to the convex hull of clustered blobs gives a tight oriented bounding box that handles arbitrary rotations gracefully. `approxPolyDP` struggles with strongly foreshortened shapes.

### Why temporal smoothing?

Even at 30 FPS, the raw per-frame detector jitters ± a few pixels due to:
- Rolling-shutter distortion
- Camera vibration
- Slight laser flicker

An EMA (α=0.25) on the 4 corner positions produces a visually stable output at the cost of ~4 frames of lag — acceptable for a safety-zone visualisation.

### Why union-find clustering instead of a single convex hull?

The 4 laser lines arrive as **4 separate contours** with small gaps at their intersections (the laser diodes don't perfectly touch at corners). Union-find groups them by bounding-box proximity before hull-fitting, avoiding the need to tune morphological kernel sizes that would differ across resolutions.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/zapdos-laser-mask.git
cd zapdos-laser-mask
pip install -r requirements.txt

# 2. Add the provided video samples
mkdir -p data/raw
cp /path/to/main.mp4 data/raw/

# 3. Run the pipeline
python main.py --input data/raw/main.mp4

# 4. With debug overlay (writes a colour debug video alongside the mask)
python main.py --input data/raw/main.mp4 --debug

# 5. Process all videos in a directory
python main.py --input data/raw/ --output data/outputs/

# 6. Auto-calibrate HSV to each video's lighting conditions
python main.py --input data/raw/main.mp4 --auto-calibrate

# 7. Downsample to 75% for 2× speed (minimal quality loss)
python main.py --input data/raw/main.mp4 --scale 0.75
```

---

## HSV Tuning Tool

If you're testing on a new site with different lighting:

```bash
python notebooks/01_hsv_exploration.py --video data/raw/main.mp4
```

Opens an interactive window with trackbars for all HSV parameters. Press `q` to quit and print the final values, then paste them into `config.py`.

---

## Evaluation

If you have the reference mask videos:

```bash
python notebooks/02_evaluate.py \
    --pred data/outputs/main_mask.mp4 \
    --gt   data/raw/main_redmask.mp4
```

Reports: mean IoU, Precision, Recall, F1 per frame.

---

## Configuration Reference

All parameters are in `config.py`. Key knobs:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `hsv.lower1/upper1` | `(0,160,120)/(10,255,255)` | Red hue arc 1 |
| `hsv.lower2/upper2` | `(158,160,120)/(180,255,255)` | Red hue arc 2 |
| `morph.close_ksize` | `15` | Gap bridging at laser corners |
| `geometry.blob_group_distance` | `120` | Max px gap between laser line fragments |
| `geometry.min_region_area` | `4000` | Minimum laser rectangle area (px²) |
| `tracking.ema_alpha` | `0.25` | Smoothing aggressiveness |
| `tracking.max_coast_frames` | `8` | Frames to hold mask when laser disappears |
| `tracking.min_confirm_frames` | `3` | Frames before a detection is trusted |
| `process_scale` | `1.0` | Downsample for speed (0.5 = 4× faster) |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover: HSV detection, morphological cleaning, spatial clustering, quad fitting, EMA tracking, coasting, mask rendering.

---

## Project Structure

```
laser_mask/
├── main.py               # CLI entry point
├── config.py             # All tuneable parameters
├── requirements.txt
├── README.md
│
├── pipeline/
│   ├── preprocess.py     # Bilateral filter, HSV threshold, auto-calibration
│   ├── detector.py       # Morphology, contour clustering, minAreaRect
│   ├── tracker.py        # IoU matching, EMA smoothing, coast/confirm FSM
│   └── postprocess.py    # fillPoly mask rendering, debug overlay
│   └── video_utils.py    # VideoReader / VideoWriter context managers
│
├── notebooks/
│   ├── 01_hsv_exploration.py   # Interactive HSV tuning tool
│   └── 02_evaluate.py          # IoU / F1 evaluation vs reference masks
│
├── tests/
│   └── test_pipeline.py        # Unit tests for all stages
│
└── data/
    ├── raw/                    # Input videos (not committed)
    └── outputs/                # Generated mask videos
```

---

## Performance

| Resolution | Scale | Hardware | Throughput |
|------------|-------|----------|------------|
| 1280×720   | 1.0   | CPU only | ~18 FPS    |
| 1280×720   | 0.75  | CPU only | ~30 FPS    |
| 1920×1080  | 0.5   | CPU only | ~25 FPS    |

All well above the 2 FPS minimum requirement.

---

## Potential Extensions

- **MobileSAM bootstrap**: auto-segment the laser region in the first frame using a prompted segmentation model, removing any dependency on hand-tuned HSV values.
- **Kalman filter**: replace EMA with a Kalman filter for better handling of fast-moving forklifts.
- **Multi-camera support**: the pipeline is stateless per video; parallel processing across camera streams is trivially parallelisable with `multiprocessing.Pool`.
- **ONNX export of colour model**: if cross-site HSV tuning becomes tedious, fine-tune a tiny pixel classifier (3-layer MLP on HSV features) and export to ONNX for CPU inference.
