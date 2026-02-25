#!/usr/bin/env python3
"""Evaluate CLIP mosaic detector against mosaic ground truth.

Two evaluation modes:
  (default) Full-image CLIP scoring on each of the 122 images.
  --regions  Scores each YOLO/DINO detection crop; evaluates per-image.

Usage:
    venv/bin/python scripts/evaluate_clip_mosaic.py
    venv/bin/python scripts/evaluate_clip_mosaic.py --regions
    venv/bin/python scripts/evaluate_clip_mosaic.py --regions --threshold 0.05
    venv/bin/python scripts/evaluate_clip_mosaic.py --no-cache
"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image, ImageOps

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frame_prep.clip_detector import CLIPMosaicDetector

GROUND_TRUTH_PATH = Path("test_real_images/mosaic_ground_truth.json")
IMAGE_DIR = Path("test_real_images/input")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CLIP mosaic detector")
    p.add_argument(
        "--regions",
        action="store_true",
        help="Score YOLO/DINO detection crops instead of full images",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Single-point evaluation at this threshold (default: sweep)",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached CLIP scores and recompute",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-image scores",
    )
    return p.parse_args()


def load_ground_truth(path: Path):
    with open(path) as f:
        return json.load(f)


def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return ImageOps.exif_transpose(img)


def iou(a, b):
    """IoU between two (x1,y1,x2,y2) boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def gt_boxes_pixels(gt_entry, img_w, img_h):
    """Convert normalised GT mosaic boxes to pixel coords."""
    boxes = []
    for b in gt_entry.get("boxes", []):
        boxes.append((
            int(b["x1_norm"] * img_w),
            int(b["y1_norm"] * img_h),
            int(b["x2_norm"] * img_w),
            int(b["y2_norm"] * img_h),
        ))
    return boxes


# ---------------------------------------------------------------------------
# Full-image evaluation
# ---------------------------------------------------------------------------

def score_all_images(gt, image_dir, detector, verbose):
    """Return list of (filename, true_label, clip_score)."""
    results = []
    filenames = sorted(gt.keys())

    for i, filename in enumerate(filenames):
        img_path = image_dir / filename
        if not img_path.exists():
            print(f"  [SKIP] {filename} not found")
            continue

        label = gt[filename]["has_mosaic"]
        img = load_image(img_path)
        s = detector._score_cached(img, "full")
        results.append((filename, label, s))

        if verbose:
            flag = "mosaic" if label else "non-mosaic"
            print(f"  [{i+1:3d}/{len(filenames)}] {filename:40s}  {flag:10s}  score={s:+.4f}")
        else:
            print(f"\r  Scoring {i+1}/{len(filenames)}...", end="", flush=True)

    if not verbose:
        print()
    return results


# ---------------------------------------------------------------------------
# Region-based evaluation
# ---------------------------------------------------------------------------

def score_regions(gt, image_dir, detector, verbose):
    """Run YOLO/DINO on each image, score every detection crop with CLIP.

    Returns:
        per_image_results: list of (filename, has_mosaic, max_score, scores_list)
            max_score = max CLIP score across all detection crops (or full image
            if no detections).
        per_crop_results: list of (filename, det_class, crop_is_mosaic, score)
            crop_is_mosaic = True if the crop has IoU >= 0.15 with a GT mosaic box
    """
    from frame_prep.detector import OptimizedEnsembleDetector

    det_engine = OptimizedEnsembleDetector()

    filenames = sorted(gt.keys())
    per_image = []
    per_crop = []

    for i, filename in enumerate(filenames):
        img_path = image_dir / filename
        if not img_path.exists():
            print(f"  [SKIP] {filename} not found")
            continue

        img = load_image(img_path)
        w, h = img.size
        has_mosaic = gt[filename]["has_mosaic"]
        gt_px = gt_boxes_pixels(gt[filename], w, h)

        print(f"\r  [{i+1}/{len(filenames)}] {filename:40s}", end="", flush=True)

        # Run YOLO/DINO (cached)
        try:
            detections = det_engine.detect(img, verbose=False, image_path=str(img_path))
        except Exception as e:
            print(f"\n  [ERROR] {filename}: {e}")
            continue

        crop_scores = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            if x2c <= x1c or y2c <= y1c:
                continue

            crop = img.crop((x1c, y1c, x2c, y2c))
            region_tag = f"{x1c},{y1c},{x2c},{y2c}"
            s = detector._score_cached(crop, region_tag)

            # Classify crop: does it overlap with a GT mosaic box?
            crop_is_mosaic = False
            if has_mosaic and gt_px:
                for gt_box in gt_px:
                    if iou((x1c, y1c, x2c, y2c), gt_box) >= 0.15:
                        crop_is_mosaic = True
                        break
            elif has_mosaic and not gt_px:
                # GT says mosaic but no specific box — whole image is mosaic
                crop_is_mosaic = True

            crop_scores.append(s)
            per_crop.append((filename, det.class_name, crop_is_mosaic, s))

        # If no detections, fall back to full image
        if not crop_scores:
            s = detector._score_cached(img, "full")
            crop_scores = [s]

        max_score = max(crop_scores)
        per_image.append((filename, has_mosaic, max_score, crop_scores))

        if verbose:
            flag = "mosaic" if has_mosaic else "non-mosaic"
            n = len(detections)
            print(f"\n    {flag:10s}  {n} dets  max={max_score:+.4f}  all={[f'{s:+.4f}' for s in crop_scores]}")

    print()
    return per_image, per_crop


# ---------------------------------------------------------------------------
# Shared reporting helpers
# ---------------------------------------------------------------------------

def print_histogram(results, label="Score", bins=10):
    mosaic_scores = [s for _, label_, s in results if label_]
    non_mosaic_scores = [s for _, label_, s in results if not label_]

    all_scores = [s for _, _, s in results]
    if not all_scores:
        return
    lo, hi = min(all_scores), max(all_scores)
    step = (hi - lo) / bins if hi != lo else 1

    print(f"\n{label} distribution  (range: {lo:+.4f} … {hi:+.4f})")
    print(f"{'Bucket':>14}  {'Mosaic':>8}  {'Non-mosaic':>10}")
    print("-" * 40)
    for i in range(bins):
        low = lo + i * step
        high = lo + (i + 1) * step
        m_count = sum(1 for s in mosaic_scores if low <= s < high)
        nm_count = sum(1 for s in non_mosaic_scores if low <= s < high)
        print(f"  [{low:+.3f},{high:+.3f})  {'█' * m_count:<16} {'░' * nm_count}")

    print()
    if mosaic_scores:
        print(f"  Mosaics ({len(mosaic_scores)}):     mean={sum(mosaic_scores)/len(mosaic_scores):+.4f}  "
              f"min={min(mosaic_scores):+.4f}  max={max(mosaic_scores):+.4f}")
    if non_mosaic_scores:
        print(f"  Non-mosaics ({len(non_mosaic_scores)}): mean={sum(non_mosaic_scores)/len(non_mosaic_scores):+.4f}  "
              f"min={min(non_mosaic_scores):+.4f}  max={max(non_mosaic_scores):+.4f}")


def sweep_thresholds(results, label=""):
    all_scores = sorted(set(s for _, _, s in results))
    lo = min(all_scores)
    thresholds = sorted(set(
        list(all_scores)
        + [round(lo + i * 0.005, 4) for i in range(60)]
    ))

    n_pos = sum(1 for _, label_, _ in results if label_)
    n_neg = sum(1 for _, label_, _ in results if not label_)

    header = f"Threshold sweep{' (' + label + ')' if label else ''}  ({n_pos} positives, {n_neg} negatives)"
    print(f"\n{header}")
    print(f"{'Threshold':>10}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}  "
          f"{'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print("-" * 65)

    best_f1, best_thr = -1, None
    prev_row = None

    for thr in thresholds:
        tp = sum(1 for _, lbl, s in results if lbl and s >= thr)
        fp = sum(1 for _, lbl, s in results if not lbl and s >= thr)
        fn = sum(1 for _, lbl, s in results if lbl and s < thr)
        tn = sum(1 for _, lbl, s in results if not lbl and s < thr)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        row = (tp, fp, fn, tn)
        if row != prev_row:
            print(f"  {thr:>8.4f}  {tp:>4}  {fp:>4}  {fn:>4}  {tn:>4}    {prec:>5.1%}  {rec:>5.1%}  {f1:>5.3f}")
            prev_row = row
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return best_thr, best_f1


def print_breakdown(results, threshold, label="image"):
    missed = [(fn, s) for fn, lbl, s in results if lbl and s < threshold]
    fps = [(fn, s) for fn, lbl, s in results if not lbl and s >= threshold]
    tps = [(fn, s) for fn, lbl, s in results if lbl and s >= threshold]

    print(f"\n--- At threshold={threshold:.4f} ---")
    print(f"  Detected ({len(tps)} TP {label}s):")
    for fn, s in sorted(tps, key=lambda x: -x[1]):
        print(f"    {fn:42s}  score={s:+.4f}")
    print(f"\n  Missed ({len(missed)} FN {label}s):")
    for fn, s in sorted(missed, key=lambda x: x[1]):
        print(f"    {fn:42s}  score={s:+.4f}")
    print(f"\n  False positives ({len(fps)} FP {label}s):")
    for fn, s in sorted(fps, key=lambda x: -x[1]):
        print(f"    {fn:42s}  score={s:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("Loading ground truth...")
    gt = load_ground_truth(GROUND_TRUTH_PATH)
    n_mosaic = sum(1 for v in gt.values() if v["has_mosaic"])
    n_total = len(gt)
    print(f"  {n_total} images, {n_mosaic} mosaics, {n_total - n_mosaic} non-mosaics")

    detector = CLIPMosaicDetector(threshold=0.0)

    print(f"\nLoading CLIP model ({CLIPMosaicDetector.MODEL_ID})...")
    detector._load_clip()
    print("  Model loaded.")

    if args.no_cache:
        from frame_prep.detector import CACHE_DIR
        import shutil
        cache_dir = CACHE_DIR / "clip_mosaic"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print("  CLIP cache cleared.")

    if args.regions:
        # ---- Region-based evaluation ----
        print("\nRunning YOLO/DINO + CLIP region scoring...")
        per_image, per_crop = score_regions(gt, IMAGE_DIR, detector, verbose=args.verbose)

        # Per-crop discrimination: mosaic-overlapping crops vs others
        crop_results = [(fn, is_mosaic, s) for fn, _, is_mosaic, s in per_crop]
        n_mc = sum(1 for _, m, _ in crop_results if m)
        n_nmc = sum(1 for _, m, _ in crop_results if not m)
        print(f"\n  {len(crop_results)} total crops: {n_mc} mosaic-overlapping, {n_nmc} non-mosaic")
        if n_mc == 0:
            print("  WARNING: no mosaic-overlapping crops found — GT IoU threshold may be too strict,")
            print("  or YOLO/DINO never detects anything overlapping the mosaic regions.")
        else:
            print_histogram(crop_results, label="Per-crop score", bins=10)
            best_thr, best_f1 = sweep_thresholds(crop_results, label="per-crop")
            print(f"\nBest per-crop F1={best_f1:.3f} at threshold={best_thr:.4f}")
            if args.threshold is not None:
                print_breakdown(crop_results, args.threshold, label="crop")
            else:
                print_breakdown(crop_results, best_thr, label="crop")

        # Per-image: fires if max crop score > threshold
        img_results = [(fn, has_mosaic, max_score) for fn, has_mosaic, max_score, _ in per_image]
        print("\n\n=== Per-image analysis (max score across all crops) ===")
        print_histogram(img_results, label="Per-image max-crop score", bins=10)
        best_thr_img, best_f1_img = sweep_thresholds(img_results, label="per-image")
        print(f"\nBest per-image F1={best_f1_img:.3f} at threshold={best_thr_img:.4f}")
        thr = args.threshold if args.threshold is not None else best_thr_img
        print_breakdown(img_results, thr, label="image")

    else:
        # ---- Full-image evaluation ----
        print("\nScoring full images...")
        results = score_all_images(gt, IMAGE_DIR, detector, verbose=args.verbose)
        print(f"  Scored {len(results)} images.")
        print_histogram(results, label="Full-image score")
        if args.threshold is not None:
            print_breakdown(results, args.threshold)
        else:
            best_thr, best_f1 = sweep_thresholds(results, label="full-image")
            print(f"\nBest F1={best_f1:.3f} at threshold={best_thr:.4f}")
            print_breakdown(results, best_thr)


if __name__ == "__main__":
    main()
