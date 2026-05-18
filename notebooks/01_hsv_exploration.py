"""
notebooks/01_hsv_exploration.py

Interactive HSV threshold explorer.

Run:  python notebooks/01_hsv_exploration.py --video data/raw/main.mp4

Opens a window with trackbars for H-lo/hi, S-lo/hi, V-lo/hi for both
red hue arcs so you can tune thresholds visually before committing them
to config.py.

Press 'q' to quit and print the final values.
Press 's' to save the current frame + mask as PNG snapshots.
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np


def nothing(_): pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", "-v", required=True)
    ap.add_argument("--frame", "-f", type=int, default=30,
                    help="Start frame index")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open {args.video}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)

    # ---- Create control window ----
    cv2.namedWindow("HSV Controls", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Output",       cv2.WINDOW_NORMAL)

    # Arc 1 (hue 0-10)
    cv2.createTrackbar("H1 lo", "HSV Controls", 0,   10,  nothing)
    cv2.createTrackbar("H1 hi", "HSV Controls", 10,  10,  nothing)
    # Arc 2 (hue 158-180)
    cv2.createTrackbar("H2 lo", "HSV Controls", 158, 180, nothing)
    cv2.createTrackbar("H2 hi", "HSV Controls", 180, 180, nothing)
    # Saturation
    cv2.createTrackbar("S lo",  "HSV Controls", 160, 255, nothing)
    cv2.createTrackbar("S hi",  "HSV Controls", 255, 255, nothing)
    # Value
    cv2.createTrackbar("V lo",  "HSV Controls", 120, 255, nothing)
    cv2.createTrackbar("V hi",  "HSV Controls", 255, 255, nothing)

    snapshot_idx = 0

    while True:
        # Read current frame (loop-back at end)
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h1lo = cv2.getTrackbarPos("H1 lo", "HSV Controls")
        h1hi = cv2.getTrackbarPos("H1 hi", "HSV Controls")
        h2lo = cv2.getTrackbarPos("H2 lo", "HSV Controls")
        h2hi = cv2.getTrackbarPos("H2 hi", "HSV Controls")
        slo  = cv2.getTrackbarPos("S lo",  "HSV Controls")
        shi  = cv2.getTrackbarPos("S hi",  "HSV Controls")
        vlo  = cv2.getTrackbarPos("V lo",  "HSV Controls")
        vhi  = cv2.getTrackbarPos("V hi",  "HSV Controls")

        mask1 = cv2.inRange(hsv, np.array([h1lo, slo, vlo]), np.array([h1hi, shi, vhi]))
        mask2 = cv2.inRange(hsv, np.array([h2lo, slo, vlo]), np.array([h2hi, shi, vhi]))
        mask  = cv2.bitwise_or(mask1, mask2)

        # Visualise: side-by-side original + mask overlay
        overlay = frame.copy()
        overlay[mask > 0] = (0, 0, 255)
        combined = np.hstack([
            cv2.resize(frame,   (640, 360)),
            cv2.resize(overlay, (640, 360)),
            cv2.cvtColor(cv2.resize(mask, (640, 360)), cv2.COLOR_GRAY2BGR),
        ])

        px_count = np.count_nonzero(mask)
        cv2.putText(combined, f"Red pixels: {px_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("Output", combined)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"snapshot_{snapshot_idx:03d}_frame.png", frame)
            cv2.imwrite(f"snapshot_{snapshot_idx:03d}_mask.png",  mask)
            print(f"[snapshot {snapshot_idx}] saved")
            snapshot_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\nFinal HSV values:")
    print(f"  lower1 = ({h1lo}, {slo}, {vlo})")
    print(f"  upper1 = ({h1hi}, {shi}, {vhi})")
    print(f"  lower2 = ({h2lo}, {slo}, {vlo})")
    print(f"  upper2 = ({h2hi}, {shi}, {vhi})")
    print("\nPaste these into config.py → HSVBounds")


if __name__ == "__main__":
    main()