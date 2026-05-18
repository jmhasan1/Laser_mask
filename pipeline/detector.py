"""
pipeline/detector.py

Core geometry stage: takes a binary red-pixel mask and returns zero or more
detected laser rectangles as ordered 4-point quadrilaterals (np.ndarray
shape (4, 2), dtype float32).

Detection flow per frame:
  1.  Morphological cleaning (open → close → dilate)
  2.  Connected component / contour extraction
  3.  Blob filtering (area, position)
  4.  Spatial clustering — nearby blobs form a single candidate region
  5.  Convex hull of each cluster
  6.  minAreaRect → rotated rectangle quad
  7.  Geometric sanity checks (area, aspect ratio, side length)

Design choices
--------------
* We deliberately do NOT run a neural network per frame.  The scene has a
  highly constrained appearance (bright saturated red lines on a grey floor),
  so classical CV is both faster and more deterministic.

* minAreaRect is preferred over approxPolyDP because the laser rectangle
  can be significantly perspective-warped; fitting an oriented rectangle
  to the convex hull is more stable than polygon simplification.

* Spatial clustering (union-find on bounding-box proximity) handles the common
  case where the four laser lines are detected as four separate contours with
  small gaps between them.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Optional, Tuple
from config import PipelineConfig


# ---------------------------------------------------------------------------
# 1. Morphological cleaning
# ---------------------------------------------------------------------------

def clean_mask(mask: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """
    Progressive morphological pipeline:
      OPEN  — removes isolated speckle (salt noise) without shrinking lines
      CLOSE — bridges gaps in the laser line (intersections appear dark)
      DILATE — slightly fattens blobs so nearby fragments merge in clustering
    """
    mc = cfg.morph

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mc.open_ksize,  mc.open_ksize))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT,    (mc.close_ksize, mc.close_ksize))
    k_dil   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mc.dilate_ksize, mc.dilate_ksize))

    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open,  iterations=mc.open_iter)
    out = cv2.morphologyEx(out,  cv2.MORPH_CLOSE, k_close, iterations=mc.close_iter)
    out = cv2.dilate(out, k_dil, iterations=mc.dilate_iter)
    return out


# ---------------------------------------------------------------------------
# 2. Blob extraction
# ---------------------------------------------------------------------------

def extract_blobs(
    cleaned_mask: np.ndarray,
    min_area: int,
) -> List[np.ndarray]:
    """
    Extract contours, filter by area, return list of contour arrays.
    Uses RETR_EXTERNAL so we only get outermost contours.
    """
    contours, _ = cv2.findContours(
        cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [c for c in contours if cv2.contourArea(c) >= min_area]


# ---------------------------------------------------------------------------
# 3. Spatial clustering (union-find)
# ---------------------------------------------------------------------------

def _bounding_box(contour: np.ndarray) -> Tuple[int, int, int, int]:
    """Returns (x, y, w, h) bounding box."""
    return cv2.boundingRect(contour)


def _boxes_nearby(
    b1: Tuple[int, int, int, int],
    b2: Tuple[int, int, int, int],
    threshold: int,
) -> bool:
    """True if the gap between the two bounding boxes is ≤ threshold pixels."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    # Compute gap along each axis (negative gap = overlap)
    gap_x = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
    gap_y = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
    return (gap_x ** 2 + gap_y ** 2) ** 0.5 <= threshold


def cluster_blobs(
    contours: List[np.ndarray],
    distance: int,
) -> List[List[np.ndarray]]:
    """
    Group contours into clusters using a simple union-find keyed on bounding-box
    proximity.  Two blobs belong to the same cluster if their bounding boxes are
    within `distance` pixels of each other (Euclidean gap metric).

    Returns a list of groups, each group being a list of contours.
    """
    n = len(contours)
    if n == 0:
        return []

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int):
        parent[find(i)] = find(j)

    boxes = [_bounding_box(c) for c in contours]
    for i in range(n):
        for j in range(i + 1, n):
            if _boxes_nearby(boxes[i], boxes[j], distance):
                union(i, j)

    groups: dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return [[contours[i] for i in idxs] for idxs in groups.values()]


# ---------------------------------------------------------------------------
# 4. Quadrilateral fitting
# ---------------------------------------------------------------------------

def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    Standard convention for perspective transforms.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)           # smallest sum → TL, largest sum → BR
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis=1)      # smallest diff → TR, largest diff → BL
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def fit_quad_to_cluster(
    cluster: List[np.ndarray],
    cfg: PipelineConfig,
) -> Optional[np.ndarray]:
    """
    Given a cluster of contours (all belonging to one laser rectangle):
      1. Merge all points into one cloud.
      2. Compute convex hull.
      3. Fit minAreaRect → returns a tight oriented bounding box.
      4. Sanity-check area, aspect ratio, and minimum side length.
      5. Return ordered 4-point quad or None if checks fail.
    """
    gc = cfg.geometry

    # Merge all contour points
    all_pts = np.vstack(cluster).reshape(-1, 1, 2)

    # Convex hull → compact point set for rect fitting
    hull = cv2.convexHull(all_pts)
    hull_area = cv2.contourArea(hull)

    if hull_area < gc.min_region_area:
        return None

    # minAreaRect gives (centre, (w,h), angle)
    rect = cv2.minAreaRect(hull)
    box  = cv2.boxPoints(rect)       # 4 corners, float32
    w, h = rect[1]

    long_side  = max(w, h)
    short_side = min(w, h)

    if short_side < 1:
        return None
    if long_side / short_side > gc.max_aspect_ratio:
        return None  # degenerate line, not a rectangle
    if short_side < gc.min_side_length:
        return None

    return _order_quad_points(box)


# ---------------------------------------------------------------------------
# 5. Public API
# ---------------------------------------------------------------------------

def detect_laser_quads(
    red_mask: np.ndarray,
    cfg: PipelineConfig,
) -> List[np.ndarray]:
    """
    Full detection pipeline: mask → list of ordered 4-point quads.

    Each quad is np.ndarray shape (4, 2), dtype float32,
    ordered [TL, TR, BR, BL] in pixel coordinates.
    """
    cleaned  = clean_mask(red_mask, cfg)
    blobs    = extract_blobs(cleaned, cfg.geometry.min_blob_area)

    if not blobs:
        return []

    clusters = cluster_blobs(blobs, cfg.geometry.blob_group_distance)

    quads = []
    for cluster in clusters:
        quad = fit_quad_to_cluster(cluster, cfg)
        if quad is not None:
            quads.append(quad)

    return quads