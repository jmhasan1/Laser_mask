"""
pipeline/preprocess.py

Handles per-frame preprocessing:
  - Optional resize for speed
  - Bilateral filtering to smooth the image while preserving laser edges
  - Dual-range HSV thresholding for red (hue wraps around 0° in OpenCV)
  - Optional auto-calibration of HSV bounds from a reference frame
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, Tuple
# from config import PipelineConfig


def resize_frame(
    frame: np.ndarray,
    scale: float,
) -> Tuple[np.ndarray, float]:
    """Downsample frame for faster processing.  Returns (resized, actual_scale)."""
    if scale == 1.0:
        return frame, 1.0
    h, w = frame.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR), scale


def preprocess_frame(frame: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """
    Light denoising that preserves colour edges.

    Bilateral filter with small diameter removes sensor noise without blurring
    the sharp red/non-red boundary that our HSV threshold depends on.
    """
    # Bilateral: d=5 is fast; sigmaColor / sigmaSpace tuned empirically
    return cv2.bilateralFilter(frame, d=5, sigmaColor=35, sigmaSpace=35)


def extract_red_mask(frame_bgr: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """
    Dual-range HSV threshold that captures red laser pixels.

    Red hue in OpenCV HSV wraps around 0°:
      arc-1: [0  … 10 ]   (orange-red)
      arc-2: [158… 180]   (magenta-red)

    Both arcs are merged with bitwise OR.

    Returns a uint8 binary mask (255 = red laser, 0 = background).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    h = cfg.hsv
    mask1 = cv2.inRange(hsv,
                        np.array(h.lower1, dtype=np.uint8),
                        np.array(h.upper1, dtype=np.uint8))
    mask2 = cv2.inRange(hsv,
                        np.array(h.lower2, dtype=np.uint8),
                        np.array(h.upper2, dtype=np.uint8))

    return cv2.bitwise_or(mask1, mask2)


# ---------------------------------------------------------------------------
# Auto-calibration (optional): samples a handful of early frames to tighten
# the HSV bounds around the actual laser colour, rejecting red objects in the
# scene (e.g. safety cones, forklift bodywork) that are outside the expected
# brightness range of a laser line.
# ---------------------------------------------------------------------------

def _sample_laser_pixels(
    hsv_frame: np.ndarray,
    rough_mask: np.ndarray,
    max_samples: int = 2000,
) -> Optional[np.ndarray]:
    """Return HSV values of pixels inside rough_mask, up to max_samples."""
    ys, xs = np.where(rough_mask > 0)
    if len(ys) == 0:
        return None
    idx = np.random.choice(len(ys), size=min(len(ys), max_samples), replace=False)
    pixels = hsv_frame[ys[idx], xs[idx]]  # shape (N, 3)
    return pixels


def auto_calibrate_hsv(
    frames_bgr: list[np.ndarray],
    cfg: PipelineConfig,
    percentile_shrink: float = 5.0,
) -> PipelineConfig:
    """
    Refine HSV bounds by sampling laser pixels across a few seed frames.

    Strategy:
      1. Apply the current (wide) bounds to each seed frame.
      2. Collect HSV pixel statistics from accepted pixels.
      3. Shrink the bounds to [p5, p95] of observed S and V channels.
         (Hue is kept wide to handle laser bloom variations.)

    This tightens specificity without manual work, and adapts to the
    exact lighting conditions of a given warehouse site.
    """
    all_pixels = []
    for frame in frames_bgr:
        preprocessed = preprocess_frame(frame, cfg)
        rough_mask   = extract_red_mask(preprocessed, cfg)
        hsv          = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)
        pixels       = _sample_laser_pixels(hsv, rough_mask)
        if pixels is not None:
            all_pixels.append(pixels)

    if not all_pixels:
        print("[auto_calibrate] No red pixels found in seed frames — keeping defaults.")
        return cfg

    all_pixels = np.vstack(all_pixels)  # (N_total, 3)
    s_vals = all_pixels[:, 1]
    v_vals = all_pixels[:, 2]

    s_lo = max(0,   int(np.percentile(s_vals, percentile_shrink)))
    s_hi = min(255, int(np.percentile(s_vals, 100 - percentile_shrink)))
    v_lo = max(0,   int(np.percentile(v_vals, percentile_shrink)))

    # Update both arcs symmetrically
    cfg.hsv.lower1 = (cfg.hsv.lower1[0], s_lo, v_lo)
    cfg.hsv.upper1 = (cfg.hsv.upper1[0], s_hi, 255)
    cfg.hsv.lower2 = (cfg.hsv.lower2[0], s_lo, v_lo)
    cfg.hsv.upper2 = (cfg.hsv.upper2[0], s_hi, 255)

    print(f"[auto_calibrate] Refined HSV: S∈[{s_lo},{s_hi}]  V≥{v_lo}")
    return cfg