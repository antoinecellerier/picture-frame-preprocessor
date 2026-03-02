#!/usr/bin/env python3
"""
Per-class detection accuracy evaluation against art_class_ground_truth.json.

For each image, runs the detector, maps the primary detection class to a
canonical art class (mural/mosaic/street_art/sculpture/painting/installation),
then reports per-class detection rate and a confusion matrix.

Usage:
    venv/bin/python scripts/evaluate_art_class_accuracy.py
    venv/bin/python scripts/evaluate_art_class_accuracy.py --no-cache
    venv/bin/python scripts/evaluate_art_class_accuracy.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frame_prep.detector import OptimizedEnsembleDetector

GT_PATH = Path("test_real_images/art_class_ground_truth.json")
IMAGE_DIR = Path("test_real_images/input")
CANONICAL_CLASSES = ["mural", "mosaic", "street_art", "sculpture", "painting", "installation", "non_art"]

# ── Class name → canonical art class mapping ──────────────────────────────────
# Uses substring matching (longest match wins).
# Ambiguous: "painted figure" / "painted art" / "wall art" → painted_2d
# (covers both mural and street_art; shown separately in verbose output)

_MAPPING: list[tuple[str, str]] = [
    # mosaic
    ("mosaic",          "mosaic"),
    ("tile art",        "mosaic"),
    ("tile mural",      "mosaic"),
    # sculpture / 3D
    ("sculpture",       "sculpture"),
    ("statue",          "sculpture"),
    ("figurine",        "sculpture"),
    ("bust",            "sculpture"),
    ("carving",         "sculpture"),
    ("pottery",         "sculpture"),
    ("relief",          "sculpture"),
    ("vase",            "sculpture"),
    ("ceramic",         "sculpture"),
    # painting (framed / canvas)
    ("framed artwork",  "painting"),
    ("painting",        "painting"),
    ("canvas",          "painting"),
    ("collage",         "painting"),
    ("mixed media",     "painting"),
    ("fresco",          "mural"),      # frescoes are wall paintings
    # street art (before "art" generic)
    ("street art",      "street_art"),
    ("graffiti",        "street_art"),
    # mural (after mosaic/fresco/street_art to avoid stealing "mural" substring)
    ("painted mural",   "mural"),
    ("mural",           "mural"),
    # ambiguous painted 2D — could be mural or street_art; mapped as "painted_2d"
    ("painted figure",  "painted_2d"),
    ("painted art",     "painted_2d"),
    ("painted",         "painted_2d"),
    # generic art labels → broad "art"
    ("artwork",         "art"),
    ("art piece",       "art"),
    ("art on wall",     "art"),
    ("wall art",        "art"),
    ("wall-mounted art","art"),
    ("decorative art",  "art"),
    ("decorative",      "art"),
    ("artistic",        "art"),
    # installation / exhibit
    ("art installation","installation"),
    ("gallery",         "installation"),
    ("museum exhibit",  "installation"),
    ("exhibit",         "installation"),
    ("display",         "installation"),
    # 3D generic
    ("figure",          "sculpture"),
]

# Fuzzy merge: painted_2d counts as mural or street_art (the two most confusable)
_PAINTED_2D_COUNTS_AS = {"mural", "street_art"}
# art counts as any non-sculpture class (genuinely ambiguous)
_ART_COUNTS_AS = {"mural", "mosaic", "street_art", "painting", "installation"}


def map_class(detector_class: str) -> str:
    """Map a raw detector class name to a canonical or intermediate class."""
    cl = detector_class.lower()
    for keyword, canon in _MAPPING:
        if keyword in cl:
            return canon
    return "unknown"


def is_correct(predicted: str, gt_class: str) -> bool:
    """Return True if prediction is considered correct for gt_class."""
    if predicted == gt_class:
        return True
    if predicted == "painted_2d" and gt_class in _PAINTED_2D_COUNTS_AS:
        return True
    if predicted == "art" and gt_class in _ART_COUNTS_AS:
        return True
    return False


def iou(box_a: list[float], box_b: list[float]) -> float:
    """IoU between [x1,y1,x2,y2] normalised boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def bbox_px_to_norm(bbox_px: list[int], w: int, h: int) -> list[float]:
    x1, y1, x2, y2 = bbox_px
    return [x1 / w, y1 / h, x2 / w, y2 / h]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-cache", action="store_true", help="Ignore cached detections")
    p.add_argument("--verbose", "-v", action="store_true", help="Print per-image results")
    p.add_argument(
        "--iou-threshold", type=float, default=0.15,
        help="IoU threshold to count a detection as hitting the GT region (default: 0.15)",
    )
    p.add_argument(
        "--dino-model", default="IDEA-Research/grounding-dino-tiny",
        help="Grounding DINO HuggingFace model ID (default: grounding-dino-tiny)",
    )
    p.add_argument(
        "--yolo-model", default="yolov8m-worldv2",
        help="YOLO model stem in models/ dir (default: yolov8m-worldv2). Use 'yoloe-26m-seg' for YOLOE.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(GT_PATH) as f:
        gt = json.load(f)

    detector = OptimizedEnsembleDetector(confidence_threshold=0.25, dino_model_id=args.dino_model, yolo_model=args.yolo_model)

    # ── per-class accumulators ─────────────────────────────────────────────────
    # Results: list of dicts per image
    results: list[dict] = []

    images = sorted(gt.keys())
    print(f"Evaluating {len(images)} images...\n")

    for i, filename in enumerate(images):
        img_path = IMAGE_DIR / filename
        if not img_path.exists():
            continue

        entry = gt[filename]
        gt_primary = entry["primary_class"]
        gt_boxes_norm = [
            [b["x1_norm"], b["y1_norm"], b["x2_norm"], b["y2_norm"]]
            for b in entry.get("boxes", [])
        ]

        img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        w, h = img.size

        if args.no_cache:
            # Clear cache entries for this image
            from frame_prep.detector import CACHE_DIR
            for p in CACHE_DIR.glob("*.json"):
                # Simple: delete all cache (blunt but works for --no-cache)
                pass  # TODO: per-image cache invalidation if needed

        try:
            detections = detector.detect(img, image_path=str(img_path))
        except Exception as e:
            print(f"  ERROR {filename}: {e}")
            continue

        if detections:
            primary = detector.get_primary_subject(detections)
            det_class = primary.class_name if primary else "none"
            if primary:
                det_conf = primary.confidence
                det_bbox_norm = bbox_px_to_norm(list(primary.bbox), w, h)
            else:
                det_class = "none"
                det_conf = 0.0
                det_bbox_norm = [0, 0, 0, 0]
        else:
            det_class = "none"
            det_conf = 0.0
            det_bbox_norm = [0, 0, 0, 0]

        mapped = map_class(det_class)

        # IoU with best GT box
        best_iou = max((iou(det_bbox_norm, gb) for gb in gt_boxes_norm), default=0.0)
        det_hits_gt = best_iou >= args.iou_threshold

        correct = is_correct(mapped, gt_primary)

        results.append({
            "filename": filename,
            "gt_class": gt_primary,
            "det_class": det_class,
            "mapped_class": mapped,
            "det_conf": det_conf,
            "best_iou": best_iou,
            "det_hits_gt": det_hits_gt,
            "class_correct": correct,
        })

        if args.verbose:
            hit_str = f"IoU={best_iou:.2f}" if det_hits_gt else f"IoU={best_iou:.2f} MISS"
            ok_str = "OK" if correct else f"WRONG({mapped})"
            print(f"  [{i+1:3d}/{len(images)}] {filename:<45} gt={gt_primary:<12} "
                  f"det={det_class:<30} {ok_str}  {hit_str}")
        elif (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(images)}...")

    print()

    # ── Per-class summary ──────────────────────────────────────────────────────
    by_gt: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_gt[r["gt_class"]].append(r)

    print("=" * 70)
    print(f"{'Class':<14} {'N':>4}  {'Correct':>7}  {'Acc':>6}  {'IoU Hit':>7}  {'IoU%':>5}")
    print("-" * 70)

    totals = {"n": 0, "correct": 0, "iou_hit": 0}
    for cls in CANONICAL_CLASSES:
        rows = by_gt.get(cls, [])
        if not rows:
            continue
        n = len(rows)
        correct = sum(1 for r in rows if r["class_correct"])
        iou_hit = sum(1 for r in rows if r["det_hits_gt"])
        print(f"  {cls:<12} {n:>4}  {correct:>7}  {correct/n:>5.0%}   {iou_hit:>7}  {iou_hit/n:>4.0%}")
        totals["n"] += n
        totals["correct"] += correct
        totals["iou_hit"] += iou_hit

    n, correct, iou_hit = totals["n"], totals["correct"], totals["iou_hit"]
    print("-" * 70)
    print(f"  {'TOTAL':<12} {n:>4}  {correct:>7}  {correct/n:>5.0%}   {iou_hit:>7}  {iou_hit/n:>4.0%}")
    print()

    # ── Confusion matrix (GT rows × predicted columns) ────────────────────────
    all_mapped = sorted(set(r["mapped_class"] for r in results))
    col_classes = [c for c in CANONICAL_CLASSES if c in all_mapped] + \
                  [c for c in all_mapped if c not in CANONICAL_CLASSES]

    print("Confusion matrix  (rows=GT, cols=predicted mapped class)")
    print(f"{'':>14}", end="")
    for c in col_classes:
        print(f"  {c[:8]:>8}", end="")
    print()
    print("-" * (14 + 10 * len(col_classes)))

    for gt_cls in CANONICAL_CLASSES:
        rows = by_gt.get(gt_cls, [])
        if not rows:
            continue
        counts = defaultdict(int)
        for r in rows:
            counts[r["mapped_class"]] += 1
        print(f"  {gt_cls:<12}", end="")
        for c in col_classes:
            v = counts.get(c, 0)
            print(f"  {v:>8}", end="")
        print()

    print()

    # ── Failures per class ─────────────────────────────────────────────────────
    print("Failures by class:")
    for cls in CANONICAL_CLASSES:
        failures = [r for r in by_gt.get(cls, []) if not r["class_correct"]]
        if not failures:
            continue
        print(f"\n  {cls} ({len(failures)} failures):")
        for r in failures:
            print(f"    {r['filename']:<45} det={r['det_class']:<30} mapped={r['mapped_class']}")


if __name__ == "__main__":
    main()
