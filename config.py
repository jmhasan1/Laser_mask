"""
config.py — Central configuration for the red-laser boundary detection pipeline.

All tuneable knobs live here so nothing is buried in code.
"""

from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np

# ---------------------------------------------------------------------------
# HSV colour bounds for red laser (two hue wraps required in OpenCV HSV)
# ---------------------------------------------------------------------------
# Red occupies two arcs in HSV hue space: [0-10] and [160-180].
# These values were derived by sampling real laser pixels across the provided clips.
# Saturation / value floors are intentionally aggressive to reject warm-white
# reflections and orange warehouse markings.

@dataclass
class HSVBounds:
    lower1: Tuple[int, int, int] = (0,   160, 120)   # hue wrap: lower arc
    upper1: Tuple[int, int, int] = (10,  255, 255)
    lower2: Tuple[int, int, int] = (158, 160, 120)   # hue wrap: upper arc
    upper2: Tuple[int, int, int] = (180, 255, 255)


# ---------------------------------------------------------------------------
# Morphological cleaning
# ---------------------------------------------------------------------------
@dataclass
class MorphConfig:
    # Small open to kill salt noise before merging blobs
    open_ksize: int = 3
    open_iter: int  = 1

    # Close to bridge gaps in the laser line (laser has dark gaps at intersections)
    close_ksize: int = 15
    close_iter: int  = 2

    # Dilate slightly to connect near-touching segments before hull fitting
    dilate_ksize: int = 5
    dilate_iter: int  = 1


# ---------------------------------------------------------------------------
# Contour / geometry filtering
# ---------------------------------------------------------------------------
@dataclass
class GeometryConfig:
    # Minimum area of raw red blobs to keep (px²) — removes isolated dust
    min_blob_area: int = 150

    # After grouping blobs by proximity, the merged convex region must be
    # at least this large (px²) before we attempt rectangle fitting
    min_region_area: int = 4_000

    # Maximum aspect ratio of bounding rect (long_side / short_side).
    # Very elongated shapes are laser lines, not rectangles — we skip them.
    max_aspect_ratio: float = 8.0

    # approxPolyDP epsilon as fraction of arc length
    poly_epsilon_frac: float = 0.04

    # Minimum side length for the fitted quadrilateral (px)
    min_side_length: int = 40

    # Group blobs closer than this (px) into the same candidate region
    blob_group_distance: int = 120


# ---------------------------------------------------------------------------
# Temporal smoothing / tracking
# ---------------------------------------------------------------------------
@dataclass
class TrackingConfig:
    # Exponential moving average weight for corner positions
    # Higher → more responsive; lower → smoother
    ema_alpha: float = 0.25

    # If IOU between current and previous quad drops below this, treat as
    # a new detection rather than updating the running average
    iou_reset_threshold: float = 0.10

    # Number of frames a tracker can "coast" without a detection before
    # it is retired (handles laser turning off momentarily)
    max_coast_frames: int = 8

    # Minimum number of consecutive detections before a track is confirmed
    # (avoids promoting transient false positives)
    min_confirm_frames: int = 3


# ---------------------------------------------------------------------------
# Output video
# ---------------------------------------------------------------------------
@dataclass
class OutputConfig:
    target_fps: float = 15.0          # ≥ 2 FPS as required; we target 15
    fourcc: str = "mp4v"
    suffix: str = "_mask"             # appended to input filename
    show_preview: bool = False        # set True for interactive debug window
    draw_debug_overlay: bool = False  # writes a colour debug video alongside


# ---------------------------------------------------------------------------
# Optional: lightweight ML fallback (MobileSAM / YOLO-World prompt)
# ---------------------------------------------------------------------------
@dataclass
class MLConfig:
    # Whether to run a one-shot ML bootstrap on the first N frames to
    # auto-calibrate HSV bounds instead of relying on hard-coded values
    use_ml_bootstrap: bool = False
    bootstrap_frames: int = 5
    model_name: str = "mobile_sam"    # "mobile_sam" | "yolo_world"


# ---------------------------------------------------------------------------
# Master config (compose all sub-configs)
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    hsv:      HSVBounds     = field(default_factory=HSVBounds)
    morph:    MorphConfig   = field(default_factory=MorphConfig)
    geometry: GeometryConfig= field(default_factory=GeometryConfig)
    tracking: TrackingConfig= field(default_factory=TrackingConfig)
    output:   OutputConfig  = field(default_factory=OutputConfig)
    ml:       MLConfig      = field(default_factory=MLConfig)

    # Frame downsample factor for processing (1 = native, 2 = half-res).
    # Detection runs at downsampled res; mask is upsampled back to native.
    process_scale: float = 1.0


# Singleton-style default — import and mutate as needed
DEFAULT_CONFIG = PipelineConfig()
