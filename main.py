"""
main.py — Red laser boundary detection pipeline entry point.

Usage
-----
# Process a single video with defaults:
    python main.py --input data/raw/main.mp4

# Process a single video with debug overlay:
    python main.py --input data/raw/main.mp4 --debug

# Process all videos in a directory:
    python main.py --input data/raw/ --output data/outputs/

# Run with auto-calibration enabled (fits HSV to each video's first frames):
    python main.py --input data/raw/main.mp4 --auto-calibrate

# Override specific config values on the command line:
    python main.py --input data/raw/main.mp4 --fps 10 --scale 0.75

Architecture recap (one line per stage)
---------------------------------------
VideoReader  → frame
preprocess   → bilateral-filtered BGR + binary red mask (HSV threshold)
detector     → raw candidate quads (morphology + contour clustering + minAreaRect)
tracker      → temporally smoothed & confirmed quads (EMA + coast)
postprocess  → filled binary mask frame
VideoWriter  → output mask video
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from config import PipelineConfig, DEFAULT_CONFIG
from pipeline.preprocess  import preprocess_frame, extract_red_mask, auto_calibrate_hsv
from pipeline.detector    import detect_laser_quads
from pipeline.tracker     import LaserTracker
from pipeline.postprocess import render_binary_mask, render_debug_overlay, scale_quads
from pipeline.video_utils import VideoReader, VideoWriter, build_output_path


# ---------------------------------------------------------------------------
# Core per-video processing function
# ---------------------------------------------------------------------------

def process_video(
    input_path: Path,
    output_dir: Path,
    cfg: PipelineConfig,
    auto_calibrate: bool = False,
    write_debug: bool = False,
) -> Path:
    """
    Full pipeline for a single input video.

    Returns the path to the output mask video.
    """
    print(f"\n{'='*60}")
    print(f"  Input : {input_path}")
    print(f"{'='*60}")

    out_mask_path  = build_output_path(input_path, output_dir, cfg.output.suffix)
    out_debug_path = build_output_path(input_path, output_dir, "_debug")

    with VideoReader(input_path) as reader:
        src_fps  = reader.fps
        height   = reader.height
        width    = reader.width

        print(f"  Source: {width}×{height} @ {src_fps:.2f} fps  "
              f"({reader.frame_count} frames)")

        target_fps = min(src_fps, cfg.output.target_fps)

        # --- Optional: auto-calibrate HSV bounds from first N frames --------
        if auto_calibrate or cfg.ml.use_ml_bootstrap:
            seed_frames = reader.read_first_n(cfg.ml.bootstrap_frames)
            if cfg.process_scale != 1.0:
                seed_frames = [
                    cv2.resize(f,
                               (int(f.shape[1]*cfg.process_scale),
                                int(f.shape[0]*cfg.process_scale)))
                    for f in seed_frames
                ]
            cfg = auto_calibrate_hsv(seed_frames, cfg)

        # Determine processing resolution
        proc_h = int(height * cfg.process_scale)
        proc_w = int(width  * cfg.process_scale)
        proc_shape = (proc_h, proc_w)

        # Initialise tracker (fresh for each video)
        tracker = LaserTracker(cfg)

        # Open writers
        with VideoWriter(out_mask_path, target_fps, width, height) as mask_writer, \
             (VideoWriter(out_debug_path, target_fps, width, height)
              if write_debug else _NullWriter()) as debug_writer:

            t0 = time.time()
            frame_count = 0

            for kept_idx, frame_bgr in reader.frames(target_fps, verbose=True):

                # 1. Optionally downscale for speed
                if cfg.process_scale != 1.0:
                    proc_frame = cv2.resize(frame_bgr, (proc_w, proc_h))
                else:
                    proc_frame = frame_bgr

                # 2. Preprocess (bilateral denoise)
                filtered = preprocess_frame(proc_frame, cfg)

                # 3. HSV red mask
                red_mask = extract_red_mask(filtered, cfg)

                # 4. Detect quads in processed-resolution space
                raw_quads = detect_laser_quads(red_mask, cfg)

                # 5. Temporal tracking (EMA + coast)
                tracker.update(raw_quads, proc_shape)
                smooth_quads = tracker.active_quads()

                # 6. Scale quads back to native resolution
                native_quads = scale_quads(smooth_quads, cfg.process_scale)

                # 7. Render binary mask at native resolution
                binary_mask = render_binary_mask(native_quads, frame_bgr.shape)

                # 8. Write mask video
                mask_writer.write(binary_mask)

                # 9. Write debug overlay (optional)
                if write_debug:
                    native_red = (
                        cv2.resize(red_mask, (width, height))
                        if cfg.process_scale != 1.0
                        else red_mask
                    )
                    elapsed = time.time() - t0 + 1e-6
                    current_fps = (kept_idx + 1) / elapsed
                    dbg = render_debug_overlay(
                        frame_bgr, native_red, binary_mask,
                        native_quads, kept_idx, current_fps,
                    )
                    debug_writer.write(dbg)

                # 10. Optional live preview
                if cfg.output.show_preview:
                    preview = np.hstack([
                        cv2.resize(frame_bgr, (640, 360)),
                        cv2.cvtColor(
                            cv2.resize(binary_mask, (640, 360)),
                            cv2.COLOR_GRAY2BGR,
                        ),
                    ])
                    cv2.imshow("Laser Mask Preview", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1

        elapsed = time.time() - t0
        print(f"\n  ✓  {frame_count} frames in {elapsed:.1f}s  "
              f"({frame_count/elapsed:.1f} fps throughput)")
        print(f"  Mask  → {out_mask_path}")
        if write_debug:
            print(f"  Debug → {out_debug_path}")

    if cfg.output.show_preview:
        cv2.destroyAllWindows()

    return out_mask_path


# ---------------------------------------------------------------------------
# Null writer (context manager no-op for when debug is disabled)
# ---------------------------------------------------------------------------

class _NullWriter:
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def write(self, _): pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Red laser boundary mask generator for forklift safety zones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  "-i", required=True,
                   help="Path to input video file or directory of videos.")
    p.add_argument("--output", "-o", default="data/outputs/",
                   help="Directory for output mask videos.")
    p.add_argument("--fps",    type=float, default=DEFAULT_CONFIG.output.target_fps,
                   help="Target output FPS (must be ≥ 2).")
    p.add_argument("--scale",  type=float, default=DEFAULT_CONFIG.process_scale,
                   help="Downsample factor for processing (1.0 = native res).")
    p.add_argument("--auto-calibrate", action="store_true",
                   help="Auto-tune HSV bounds from the first few frames.")
    p.add_argument("--debug",  action="store_true",
                   help="Write a colour debug overlay video alongside the mask.")
    p.add_argument("--preview", action="store_true",
                   help="Show live preview window during processing.")
    return p


def main():
    args = build_parser().parse_args()

    if args.fps < 2.0:
        print("ERROR: --fps must be ≥ 2 (challenge requirement).", file=sys.stderr)
        sys.exit(1)

    # Build config from args
    cfg = PipelineConfig()
    cfg.output.target_fps      = args.fps
    cfg.process_scale          = args.scale
    cfg.output.show_preview    = args.preview
    cfg.output.draw_debug_overlay = args.debug

    input_path  = Path(args.input)
    output_dir  = Path(args.output)

    # Collect video files to process
    if input_path.is_dir():
        video_files = sorted(input_path.glob("*.mp4")) + \
                      sorted(input_path.glob("*.avi")) + \
                      sorted(input_path.glob("*.mov"))
        if not video_files:
            print(f"No video files found in {input_path}", file=sys.stderr)
            sys.exit(1)
    elif input_path.is_file():
        video_files = [input_path]
    else:
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLaser Boundary Detector")
    print(f"  Processing {len(video_files)} video(s)")
    print(f"  Target FPS: {cfg.output.target_fps}")
    print(f"  Process scale: {cfg.process_scale}")

    outputs = []
    for vf in video_files:
        out = process_video(
            input_path   = vf,
            output_dir   = output_dir,
            cfg          = cfg,
            auto_calibrate = args.auto_calibrate,
            write_debug  = args.debug,
        )
        outputs.append(out)

    print(f"\n{'='*60}")
    print(f"Done. {len(outputs)} mask video(s) written to {output_dir}")


if __name__ == "__main__":
    main()