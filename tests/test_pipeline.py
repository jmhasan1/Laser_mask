"""
tests/test_pipeline.py

Unit tests for each pipeline stage using synthetic frames.
Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import pytest

from config import PipelineConfig
from pipeline.preprocess import extract_red_mask, preprocess_frame
from pipeline.detector   import clean_mask, extract_blobs, cluster_blobs, fit_quad_to_cluster, detect_laser_quads
from pipeline.tracker    import LaserTracker, Track
from pipeline.postprocess import render_binary_mask, scale_quads


# ---------------------------------------------------------------------------
# Helpers: synthetic frame generators
# ---------------------------------------------------------------------------

def make_black_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def draw_red_rectangle(frame, tl, br, thickness=4):
    """Draw a bright red rectangle (BGR) on frame in-place."""
    cv2.rectangle(frame, tl, br, color=(0, 0, 220), thickness=thickness)
    return frame


def make_red_rect_frame(h=480, w=640, tl=(80, 60), br=(400, 300)):
    frame = make_black_frame(h, w)
    draw_red_rectangle(frame, tl, br)
    return frame


# ---------------------------------------------------------------------------
# preprocess tests
# ---------------------------------------------------------------------------

class TestRedMask:

    def test_detects_bright_red(self):
        frame = make_red_rect_frame()
        cfg   = PipelineConfig()
        mask  = extract_red_mask(frame, cfg)
        # Should have substantial white pixels where the rectangle is
        assert mask.max() == 255, "Mask should contain white pixels"
        white_count = np.count_nonzero(mask)
        assert white_count > 100, f"Expected > 100 red pixels, got {white_count}"

    def test_black_frame_gives_empty_mask(self):
        frame = make_black_frame()
        cfg   = PipelineConfig()
        mask  = extract_red_mask(frame, cfg)
        assert np.all(mask == 0), "Black frame should produce zero mask"

    def test_blue_frame_gives_empty_mask(self):
        frame = np.full((480, 640, 3), (200, 0, 0), dtype=np.uint8)  # blue
        cfg   = PipelineConfig()
        mask  = extract_red_mask(frame, cfg)
        assert np.count_nonzero(mask) == 0, "Blue frame should produce zero mask"

    def test_bilateral_preserves_shape(self):
        frame  = make_red_rect_frame()
        cfg    = PipelineConfig()
        result = preprocess_frame(frame, cfg)
        assert result.shape == frame.shape


# ---------------------------------------------------------------------------
# detector tests
# ---------------------------------------------------------------------------

class TestDetector:

    def _get_mask_for_rect(self, tl=(80, 60), br=(400, 300)):
        frame = make_red_rect_frame(tl=tl, br=br)
        cfg   = PipelineConfig()
        mask  = extract_red_mask(frame, cfg)
        return mask, cfg

    def test_clean_mask_reduces_noise(self):
        # Create a mostly-noise mask with one real blob
        rng  = np.random.default_rng(42)
        mask = np.zeros((480, 640), dtype=np.uint8)
        # Sprinkle salt noise
        noise_yx = rng.integers(0, [480, 640], size=(200, 2))
        mask[noise_yx[:, 0], noise_yx[:, 1]] = 255
        # One real blob
        cv2.rectangle(mask, (100, 100), (300, 250), 255, -1)

        cfg     = PipelineConfig()
        cleaned = clean_mask(mask, cfg)
        blobs   = extract_blobs(cleaned, min_area=cfg.geometry.min_blob_area)
        # Should have 1 blob (the real rectangle), not 200 noise dots
        assert len(blobs) == 1

    def test_detect_single_laser_rectangle(self):
        frame = make_black_frame(480, 640)
        # Draw 4 red lines forming a rectangle (simulates laser outline)
        cv2.line(frame, (100, 80),  (400, 80),  (0, 0, 220), 6)  # top
        cv2.line(frame, (100, 280), (400, 280), (0, 0, 220), 6)  # bottom
        cv2.line(frame, (100, 80),  (100, 280), (0, 0, 220), 6)  # left
        cv2.line(frame, (400, 80),  (400, 280), (0, 0, 220), 6)  # right

        cfg   = PipelineConfig()
        mask  = extract_red_mask(frame, cfg)
        quads = detect_laser_quads(mask, cfg)
        assert len(quads) == 1, f"Expected 1 quad, got {len(quads)}"

    def test_quad_has_four_corners(self):
        frame = make_black_frame(480, 640)
        cv2.line(frame, (100, 80),  (400, 80),  (0, 0, 220), 6)
        cv2.line(frame, (100, 280), (400, 280), (0, 0, 220), 6)
        cv2.line(frame, (100, 80),  (100, 280), (0, 0, 220), 6)
        cv2.line(frame, (400, 80),  (400, 280), (0, 0, 220), 6)

        cfg   = PipelineConfig()
        mask  = extract_red_mask(frame, cfg)
        quads = detect_laser_quads(mask, cfg)
        assert quads[0].shape == (4, 2)

    def test_no_detection_on_black_frame(self):
        frame = make_black_frame()
        cfg   = PipelineConfig()
        mask  = extract_red_mask(frame, cfg)
        quads = detect_laser_quads(mask, cfg)
        assert quads == []


# ---------------------------------------------------------------------------
# tracker tests
# ---------------------------------------------------------------------------

class TestTracker:

    def _make_quad(self, x=100, y=100, w=200, h=150):
        return np.array([
            [x,     y    ],
            [x + w, y    ],
            [x + w, y + h],
            [x,     y + h],
        ], dtype=np.float32)

    def test_track_confirmed_after_k_frames(self):
        cfg = PipelineConfig()
        cfg.tracking.min_confirm_frames = 3
        tracker = LaserTracker(cfg)

        quad  = self._make_quad()
        shape = (480, 640)

        # Feed same quad for 3 frames
        for _ in range(3):
            tracker.update([quad], shape)

        active = tracker.active_quads()
        assert len(active) == 1, "Track should be confirmed after 3 hits"

    def test_track_not_confirmed_before_k_frames(self):
        cfg = PipelineConfig()
        cfg.tracking.min_confirm_frames = 5
        tracker = LaserTracker(cfg)

        quad  = self._make_quad()
        shape = (480, 640)

        for _ in range(2):
            tracker.update([quad], shape)

        active = tracker.active_quads()
        assert len(active) == 0, "Track should not be confirmed yet"

    def test_track_coasts_on_miss(self):
        cfg = PipelineConfig()
        cfg.tracking.min_confirm_frames = 1
        cfg.tracking.max_coast_frames   = 5
        tracker = LaserTracker(cfg)

        quad  = self._make_quad()
        shape = (480, 640)

        # Confirm the track
        tracker.update([quad], shape)
        assert len(tracker.active_quads()) == 1

        # Miss 4 frames — should still be active
        for _ in range(4):
            tracker.update([], shape)
        assert len(tracker.active_quads()) == 1

        # Miss past max_coast_frames — should be lost
        for _ in range(6):
            tracker.update([], shape)
        assert len(tracker.active_quads()) == 0

    def test_ema_smoothing_reduces_jitter(self):
        cfg = PipelineConfig()
        cfg.tracking.min_confirm_frames = 1
        cfg.tracking.ema_alpha = 0.5
        tracker = LaserTracker(cfg)

        quad1 = self._make_quad(100, 100)
        quad2 = self._make_quad(200, 100)  # shifted 100 px right
        shape = (480, 640)

        tracker.update([quad1], shape)
        q_after_1 = tracker.active_quads()[0].copy()

        tracker.update([quad2], shape)
        q_after_2 = tracker.active_quads()[0].copy()

        # With alpha=0.5, the position should be between quad1 and quad2
        expected_x = 0.5 * 200 + 0.5 * 100
        assert abs(q_after_2[0, 0] - expected_x) < 5, \
            f"EMA not working correctly: got {q_after_2[0, 0]:.1f}, expected ≈ {expected_x}"


# ---------------------------------------------------------------------------
# postprocess tests
# ---------------------------------------------------------------------------

class TestPostprocess:

    def _make_quad(self, x=100, y=80, w=250, h=180):
        return np.array([
            [x,     y    ],
            [x + w, y    ],
            [x + w, y + h],
            [x,     y + h],
        ], dtype=np.float32)

    def test_render_mask_fills_polygon(self):
        quad  = self._make_quad()
        shape = (480, 640, 3)
        mask  = render_binary_mask([quad], shape)

        assert mask.shape == (480, 640)
        # Interior pixel should be white
        cx, cy = 225, 170  # inside the quad
        assert mask[cy, cx] == 255

        # Exterior pixel should be black
        assert mask[10, 10] == 0

    def test_render_mask_is_binary(self):
        quad = self._make_quad()
        mask = render_binary_mask([quad], (480, 640, 3))
        unique = np.unique(mask)
        assert set(unique).issubset({0, 255}), f"Non-binary values: {unique}"

    def test_scale_quad_identity(self):
        quad   = self._make_quad()
        scaled = scale_quads([quad], scale=1.0)[0]
        np.testing.assert_array_equal(quad, scaled)

    def test_scale_quad_upscale(self):
        quad   = self._make_quad(50, 40, 125, 90)
        scaled = scale_quads([quad], scale=0.5)[0]
        # scale=0.5 means processing was at half-res; upscale by 2×
        np.testing.assert_allclose(scaled, quad * 2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])