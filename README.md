# Real-Time Forklift Laser Boundary Detection

A production-oriented Computer Vision pipeline for detecting and masking forklift safety laser boundaries in industrial warehouse environments.

The system processes raw warehouse video streams and generates clean binary mask videos representing the forklift safety laser region in real time.

Designed for:
- real-time CPU inference
- offline/on-prem deployment
- noisy industrial environments
- low-latency safety monitoring systems

The pipeline avoids heavy deep learning models and instead combines:
- HSV color segmentation
- morphological image processing
- geometry-aware quadrilateral fitting
- temporal tracking and smoothing

to achieve stable and explainable detections at high throughput.

## Core Features

- Real-time CPU-only processing
- Classical CV pipeline optimized for industrial environments
- Dual-arc HSV laser detection
- Morphological noise suppression
- Perspective-robust quadrilateral fitting
- Temporal tracking with EMA smoothing
- False-positive suppression using finite-state confirmation logic
- Multi-video batch processing
- Auto-calibration for varying warehouse lighting conditions
- Debug overlay generation
- Unit-tested modular architecture

## Pipeline Architecture

```text
Input Video
    ↓
Frame Sampling
    ↓
Bilateral Filtering
    ↓
HSV Red Segmentation
    ↓
Morphological Cleanup
    ↓
Contour Extraction
    ↓
Spatial Clustering
    ↓
Rotated Quad Fitting
    ↓
Temporal Tracking
    ↓
Binary Mask Rendering

## Performance Benchmarks

Tested on:
- Intel i5 10th Gen CPU
- CPU-only inference
- 1248×720 warehouse footage

| Video | Resolution | Frames | Throughput |
|---|---|---|---|
| clip1.mp4 | 1248×720 | 63 | 51.4 FPS |
| clip1 (2).mp4 | 1248×720 | 63 | 37.0 FPS |
| clip1 (3).mp4 | 1248×720 | 63 | 39.6 FPS |
| clip2.mp4 | 1248×720 | 62 | 72.9 FPS |
| clip2 (2).mp4 | 1248×720 | 62 | 60.0 FPS |
| clip2 (3).mp4 | 1248×720 | 62 | 58.6 FPS |

All significantly exceed the 2 FPS minimum requirement specified in the challenge.

## Observed Failure Cases

The current pipeline performs robustly under most warehouse conditions, but several challenging edge cases remain:

- Strong reflective warehouse flooring can occasionally generate transient false positives.
- Severe motion blur may temporarily reduce contour connectivity.
- Partial laser occlusions can shrink the detected quadrilateral region.
- Extremely saturated ambient red lighting may interfere with HSV segmentation.
- Overlapping forklift laser regions may merge under aggressive clustering settings.

Mitigation strategies currently implemented:
- Temporal confirmation FSM
- EMA smoothing
- Area and aspect-ratio filtering
- Spatial clustering constraints
- Morphological cleanup pipeline

## Engineering Tradeoffs

This project intentionally prioritizes:

- real-time CPU performance
- deterministic behavior
- explainability
- ease of debugging
- low deployment complexity

over:
- large deep learning segmentation models
- GPU dependency
- training-heavy approaches

While neural segmentation models may improve robustness in rare edge cases, the classical CV approach offers:
- significantly higher throughput
- simpler deployment
- easier tuning
- stronger interpretability
- lower operational cost

## Current Limitations

- HSV thresholds may require retuning for dramatically different lighting environments.
- Perspective distortion handling relies on minAreaRect rather than full homography reconstruction.
- The tracker currently uses EMA smoothing rather than predictive motion modeling.
- Extreme occlusions may temporarily destabilize quadrilateral fitting.

## Potential Future Improvements

- Adaptive HSV threshold refinement
- Perspective-aware homography stabilization
- Kalman-filter-based motion prediction
- Multi-camera synchronization
- Edge-aware contour reconstruction
- Lightweight learned pixel classifier for cross-site robustness
- Dynamic kernel sizing based on distance estimation

## Design Philosophy

The pipeline was designed around a core principle:

> Prefer lightweight, explainable, and deterministic computer vision systems when the visual problem is sufficiently constrained.

Forklift safety lasers are:
- highly saturated
- geometrically structured
- visually distinct

making them well-suited for a hybrid classical CV approach rather than large neural segmentation models.

## Why No Deep Learning?

Deep learning segmentation models (e.g. SAM, YOLO segmentation, Mask R-CNN) were intentionally not used as the primary solution because:

- the dataset size is extremely limited
- inference latency is significantly higher
- deployment complexity increases
- interpretability decreases
- deterministic debugging becomes harder

The current classical CV pipeline already achieves:
- real-time throughput
- stable masks
- strong explainability
- low compute requirements

which aligns well with industrial edge deployment requirements.

## Technical Highlights

- 70+ FPS throughput on CPU-only inference
- Perspective-robust quadrilateral fitting
- Temporal FSM-based false-positive suppression
- Union-find contour clustering
- Modular testable architecture
- Interactive HSV tuning tool
- Auto-calibration pipeline
- Synthetic unit-test framework

## Test Environment

- CPU: Intel i5 10th Gen
- RAM: 8 GB DDR4
- GPU: RTX 1650 Ti (unused)
- OS: Windows 10
- Python: 3.x

