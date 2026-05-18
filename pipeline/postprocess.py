"""
pipeline/postprocess.py

Converts confirmed, smoothed quadrilaterals into clean binary masks and
(optionally) a colour debug overlay video.

Output mask conventions
-----------------------
* Background  → 0   (black)
* Laser region → 255 (white)

The mask is always a *filled* polygon, not just the boundary outline.
Filled polygons are noise-free by construction — no morphological clean-up
needed after this stage.

Debug overlay
-------------
When enabled, writes a second video with:
  - Original frame BGR
  - Detected quads drawn as coloured outlines (green = confirmed)
  - HSV red-pixel mask blended in magenta
  - Frame number and FPS counter
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Optional, Tuple
from config import PipelineConfig


# ---------------------------------------------------------------------------
# Binary mask rendering
# ---------------------------------------------------------------------------

def render_binary_mask(
    quads: List[np.ndarray],
    frame_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Rasterise all confirmed quads into a single-channel binary mask.

    Parameters
    ----------
    quads       : list of (4, 2) float32 arrays — ordered quad corners
    frame_shape : (H, W, C) of the *original* (not downscaled) frame

    Returns
    -------
    mask : uint8 array of shape (H, W), values 0 or 255
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for quad in quads:
        pts = quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

    return mask


# ---------------------------------------------------------------------------
# Scale quad from processing resolution to output resolution
# ---------------------------------------------------------------------------

def scale_quad(quad: np.ndarray, scale: float) -> np.ndarray:
    """
    When processing was done at a down-scaled resolution, multiply corner
    coordinates back to native resolution.
    """
    if scale == 1.0:
        return quad
    return quad / scale


def scale_quads(quads: List[np.ndarray], scale: float) -> List[np.ndarray]:
    return [scale_quad(q, scale) for q in quads]


# ---------------------------------------------------------------------------
# Debug overlay renderer
# ---------------------------------------------------------------------------

def render_debug_overlay(
    frame_bgr: np.ndarray,
    red_mask: np.ndarray,
    binary_mask: np.ndarray,
    quads: List[np.ndarray],
    frame_idx: int,
    fps: float,
) -> np.ndarray:
    """
    Produce a rich debug frame blending original image, red pixel mask,
    and detected quads.  Useful for diagnosing false positives / negatives.
    """
    overlay = frame_bgr.copy()

    # 1. Blend detected red pixels as magenta tint
    if red_mask is not None:
        magenta_layer = np.zeros_like(overlay)
        magenta_layer[red_mask > 0] = (180, 0, 180)
        cv2.addWeighted(magenta_layer, 0.4, overlay, 0.6, 0, overlay)

    # 2. Draw output binary mask as semi-transparent green fill
    if binary_mask is not None:
        green_layer = np.zeros_like(overlay)
        green_layer[binary_mask > 0] = (0, 200, 0)
        cv2.addWeighted(green_layer, 0.25, overlay, 0.75, 0, overlay)

    # 3. Draw quad outlines
    for quad in quads:
        pts = quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=True,
                      color=(0, 255, 0), thickness=2)
        # Draw corner dots
        for pt in quad.astype(np.int32):
            cv2.circle(overlay, tuple(pt), 5, (0, 255, 255), -1)

    # 4. HUD text
    h, w = overlay.shape[:2]
    cv2.putText(overlay, f"Frame {frame_idx:04d}  |  {fps:.1f} FPS",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(overlay, f"Tracks: {len(quads)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                cv2.LINE_AA)

    return overlay