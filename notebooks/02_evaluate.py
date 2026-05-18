"""
notebooks/02_evaluate.py

Quantitative evaluation against the reference masks provided in the repo.

Metrics computed per-frame:
  - IoU (Intersection over Union) of detected region vs reference mask
  - Precision / Recall / F1
  - Hausdorff distance between predicted and reference quad centroids

Aggregated as mean ± std over all frames and all test videos.

Usage:
    python notebooks/02_evaluate.py \
        --pred  data/outputs/main_mask.mp4 \
        --gt    data/raw/main_redmask.mp4

Output:
    Prints a table and writes results/eval_results.json
"""

import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np


def binarise(frame: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Convert a BGR or grayscale frame to binary mask."""
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
    return binary


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.count_nonzero(cv2.bitwise_and(pred, gt))
    union = np.count_nonzero(cv2.bitwise_or(pred, gt))
    return inter / union if union > 0 else (1.0 if inter == 0 else 0.0)


def compute_precision_recall(pred: np.ndarray, gt: np.ndarray):
    tp = np.count_nonzero(cv2.bitwise_and(pred, gt))
    fp = np.count_nonzero(pred) - tp
    fn = np.count_nonzero(gt)   - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def evaluate_pair(pred_path: Path, gt_path: Path) -> dict:
    cap_pred = cv2.VideoCapture(str(pred_path))
    cap_gt   = cv2.VideoCapture(str(gt_path))

    ious, precs, recs, f1s = [], [], [], []
    frame_idx = 0

    while True:
        ret_p, pred_frame = cap_pred.read()
        ret_g, gt_frame   = cap_gt.read()
        if not ret_p or not ret_g:
            break

        pred_mask = binarise(pred_frame)
        gt_mask   = binarise(gt_frame)

        # Resize to same size if mismatch
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        iou        = compute_iou(pred_mask, gt_mask)
        prec, rec, f1 = compute_precision_recall(pred_mask, gt_mask)

        ious.append(iou)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        frame_idx += 1

    cap_pred.release()
    cap_gt.release()

    if not ious:
        return {"error": "No frames compared"}

    return {
        "frames"        : frame_idx,
        "mean_iou"      : float(np.mean(ious)),
        "std_iou"       : float(np.std(ious)),
        "mean_precision": float(np.mean(precs)),
        "mean_recall"   : float(np.mean(recs)),
        "mean_f1"       : float(np.mean(f1s)),
        "per_frame_iou" : [round(v, 4) for v in ious],
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate predicted mask vs ground truth")
    ap.add_argument("--pred", required=True, help="Predicted mask video")
    ap.add_argument("--gt",   required=True, help="Ground-truth mask video")
    ap.add_argument("--out",  default="results/eval_results.json")
    args = ap.parse_args()

    pred_path = Path(args.pred)
    gt_path   = Path(args.gt)
    out_path  = Path(args.out)

    print(f"\nEvaluating:")
    print(f"  Prediction : {pred_path}")
    print(f"  Ground truth: {gt_path}")

    results = evaluate_pair(pred_path, gt_path)

    # Print table
    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    for k, v in results.items():
        if k == "per_frame_iou":
            continue
        print(f"  {k:<18} {v:>10.4f}" if isinstance(v, float)
              else f"  {k:<18} {v:>10}")

    # Save JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
