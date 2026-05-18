"""
pipeline/tracker.py

Temporal tracking layer that sits between the per-frame detector and the
mask renderer.  It solves three key problems:

  1. JITTER — laser lines vibrate slightly across frames (camera vibration,
     rolling shutter).  An EMA on corner positions produces smooth motion.

  2. FLICKER — the detector misses the rectangle on some frames (motion blur,
     partial occlusion).  A "coasting" mechanism holds the last good quad for
     up to N frames before retiring the track.

  3. FALSE-POSITIVE SUPPRESSION — a new detection must persist for at least
     K frames before it is "confirmed" and appears in the output.

Architecture
------------
Each active laser boundary is represented as a Track object.  Each frame, the
detector produces a list of candidate quads.  We match them to existing tracks
using IoU (of the filled polygon areas) and update accordingly.

The tracker is intentionally lightweight: for the typical case of 1–3 laser
rectangles in the frame, O(N²) matching is negligible.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Optional, Tuple
from config import PipelineConfig


# ---------------------------------------------------------------------------
# Utility: polygon IoU on a shared canvas
# ---------------------------------------------------------------------------

def _poly_iou(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    shape: Tuple[int, int],
) -> float:
    """
    Rasterise two quadrilaterals and compute pixel-level IoU.
    shape = (height, width).
    """
    def fill(pts: np.ndarray) -> np.ndarray:
        canvas = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(canvas, [pts.astype(np.int32)], 255)
        return canvas

    m_a = fill(pts_a)
    m_b = fill(pts_b)

    inter = np.count_nonzero(cv2.bitwise_and(m_a, m_b))
    union = np.count_nonzero(cv2.bitwise_or(m_a, m_b))
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Track object
# ---------------------------------------------------------------------------

class Track:
    """
    Represents a single tracked laser boundary across time.

    State machine:
      TENTATIVE → (min_confirm_frames detections) → CONFIRMED
      CONFIRMED → (max_coast_frames misses) → LOST (removed)
    """

    _id_counter = 0

    def __init__(self, quad: np.ndarray, cfg: PipelineConfig):
        Track._id_counter += 1
        self.track_id        = Track._id_counter
        self.quad            = quad.copy()          # smoothed corners (float32)
        self._cfg            = cfg
        self.hit_streak      = 1                    # consecutive detections
        self.coast_frames    = 0                    # consecutive misses
        # Confirm immediately if threshold is ≤ 1 (first detection counts)
        self.confirmed       = (cfg.tracking.min_confirm_frames <= 1)

    # ------------------------------------------------------------------
    def update(self, quad: np.ndarray):
        """Incorporate a new detection via EMA corner smoothing."""
        alpha = self._cfg.tracking.ema_alpha
        self.quad         = alpha * quad + (1 - alpha) * self.quad
        self.hit_streak  += 1
        self.coast_frames = 0
        if self.hit_streak >= self._cfg.tracking.min_confirm_frames:
            self.confirmed = True

    def coast(self):
        """Advance one frame with no matching detection."""
        self.coast_frames += 1
        self.hit_streak    = 0

    @property
    def is_lost(self) -> bool:
        return self.coast_frames > self._cfg.tracking.max_coast_frames

    @property
    def output_quad(self) -> Optional[np.ndarray]:
        """Returns the smoothed quad if confirmed, else None."""
        return self.quad.copy() if self.confirmed else None


# ---------------------------------------------------------------------------
# Multi-object tracker
# ---------------------------------------------------------------------------

class LaserTracker:
    """
    Frame-by-frame multi-track manager.

    Call update(quads, frame_shape) once per frame.
    Call active_quads() to retrieve confirmed, smoothed rectangles.
    """

    def __init__(self, cfg: PipelineConfig):
        self._cfg    = cfg
        self._tracks: List[Track] = []
        self._frame_shape: Tuple[int, int] = (720, 1280)  # updated each call

    # ------------------------------------------------------------------
    def update(
        self,
        detected_quads: List[np.ndarray],
        frame_shape: Tuple[int, int],
    ):
        """
        Match detected quads to existing tracks (greedy IoU matching),
        update matched tracks, coast unmatched tracks, spawn new tracks.
        """
        self._frame_shape = frame_shape
        tc = self._cfg.tracking

        if not self._tracks:
            # Cold start: spawn a track for every detection
            for q in detected_quads:
                self._tracks.append(Track(q, self._cfg))
            return

        # Build IoU matrix: rows = tracks, cols = detections
        n_tracks = len(self._tracks)
        n_dets   = len(detected_quads)

        if n_dets == 0:
            for t in self._tracks:
                t.coast()
            self._prune()
            return

        iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
        for ti, track in enumerate(self._tracks):
            for di, quad in enumerate(detected_quads):
                iou_matrix[ti, di] = _poly_iou(track.quad, quad, frame_shape)

        # Greedy matching (good enough for ≤ ~5 rectangles)
        matched_tracks = set()
        matched_dets   = set()

        # Sort potential pairs by IoU descending
        pairs = sorted(
            [(iou_matrix[ti, di], ti, di)
             for ti in range(n_tracks)
             for di in range(n_dets)],
            reverse=True,
        )

        for iou, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            if iou < tc.iou_reset_threshold:
                break  # remaining pairs are below threshold
            self._tracks[ti].update(detected_quads[di])
            matched_tracks.add(ti)
            matched_dets.add(di)

        # Coast unmatched tracks
        for ti, track in enumerate(self._tracks):
            if ti not in matched_tracks:
                track.coast()

        # Spawn new tracks for unmatched detections
        for di, quad in enumerate(detected_quads):
            if di not in matched_dets:
                self._tracks.append(Track(quad, self._cfg))

        self._prune()

    # ------------------------------------------------------------------
    def _prune(self):
        self._tracks = [t for t in self._tracks if not t.is_lost]

    # ------------------------------------------------------------------
    def active_quads(self) -> List[np.ndarray]:
        """Return list of confirmed, smoothed quads for the current frame."""
        result = []
        for t in self._tracks:
            q = t.output_quad
            if q is not None:
                result.append(q)
        return result

    def reset(self):
        self._tracks = []
        Track._id_counter = 0