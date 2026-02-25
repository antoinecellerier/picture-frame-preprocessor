#!/usr/bin/env python3
"""Evaluate OWLv2 open-vocabulary detector for mosaic detection.

OWLv2 takes text queries and returns bounding boxes directly, unlike CLIP
which scores whole-image or region similarity.  This script tests whether
OWLv2 can localise mosaics better than YOLO/DINO (F1=0.529 with low-conf
candidate pass).

Strategy: run OWLv2 ONCE per image with ALL unique queries combined across all
prompt sets; cache the combined result; evaluate each prompt set by filtering
detections to that set's queries.  This avoids 4× redundant model inference.

Baseline for comparison:
  YOLO/DINO conf=0.10: TP=18/28 (64%), FP=22/94 (23%), F1=0.529

Usage:
    venv/bin/python scripts/evaluate_owlv2_mosaic.py
    venv/bin/python scripts/evaluate_owlv2_mosaic.py --model google/owlv2-large-patch14-finetuned
    venv/bin/python scripts/evaluate_owlv2_mosaic.py --threshold 0.05 --verbose
    venv/bin/python scripts/evaluate_owlv2_mosaic.py --prompt-set narrow
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

GROUND_TRUTH_PATH = Path("test_real_images/mosaic_ground_truth.json")
IMAGE_DIR = Path("test_real_images/input")
CACHE_DIR = Path("cache/owlv2")

DEFAULT_MODEL = "google/owlv2-base-patch16-finetuned"

# Prompt sets — each is a subset of ALL_PROMPTS.
# All prompts are inferred in one combined pass; results filtered per set.
PROMPT_SETS = {
    "narrow": [
        "mosaic",
    ],
    "medium": [
        "mosaic",
        "tile artwork",
        "decorative tile",
    ],
    "wide": [
        "mosaic",
        "tile artwork",
        "decorative tile",
        "ceramic tile art",
        "stone mosaic",
        "tile mural",
    ],
    "art_only": [
        "mosaic artwork",
        "tile mosaic art",
    ],
}

# Union of all unique prompts — used for the single combined inference pass
ALL_PROMPTS = sorted(set(q for qs in PROMPT_SETS.values() for q in qs))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate OWLv2 for mosaic detection")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"OWLv2 model ID (default: {DEFAULT_MODEL})")
    p.add_argument("--prompt-set", default=None,
                   choices=list(PROMPT_SETS.keys()),
                   help="Named prompt set to evaluate (default: all)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Single-point confidence threshold (default: sweep)")
    p.add_argument("--iou-threshold", type=float, default=0.15,
                   help="IoU threshold for a detection to count as TP (default: 0.15)")
    p.add_argument("--no-cache", action="store_true",
                   help="Ignore cached results and recompute all inferences")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-image detections")
    return p.parse_args()


# ─────────────────────────────────────────────
# IMAGE / CACHE HELPERS
# ─────────────────────────────────────────────

def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return ImageOps.exif_transpose(img)


def _cache_key(img_path: Path, model_id: str, prompts: list) -> str:
    stat = img_path.stat()
    key = (f"{img_path.absolute()}:{stat.st_size}:{stat.st_mtime}"
           f":{model_id}:{json.dumps(sorted(prompts))}")
    return hashlib.sha256(key.encode()).hexdigest()[:20]


def _load_cached(img_path: Path, model_id: str, prompts: list):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(img_path, model_id, prompts)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None


def _save_cached(img_path: Path, model_id: str, prompts: list, dets: list):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(img_path, model_id, prompts)
    cache_file = CACHE_DIR / f"{key}.json"
    with open(cache_file, "w") as f:
        json.dump(dets, f)


# ─────────────────────────────────────────────
# IoU / GT helpers
# ─────────────────────────────────────────────

def iou(a, b):
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
    boxes = []
    for b in gt_entry.get("boxes", []):
        boxes.append((
            int(b["x1_norm"] * img_w),
            int(b["y1_norm"] * img_h),
            int(b["x2_norm"] * img_w),
            int(b["y2_norm"] * img_h),
        ))
    return boxes


# ─────────────────────────────────────────────
# OWLv2 MODEL
# ─────────────────────────────────────────────

def load_owlv2(model_id: str):
    from transformers import Owlv2Processor, Owlv2ForObjectDetection

    print(f"  Loading OWLv2 ({model_id})...")
    try:
        processor = Owlv2Processor.from_pretrained(model_id, local_files_only=True)
        model = Owlv2ForObjectDetection.from_pretrained(model_id, local_files_only=True)
    except OSError:
        print("  Not cached locally — downloading from HuggingFace...")
        processor = Owlv2Processor.from_pretrained(model_id)
        model = Owlv2ForObjectDetection.from_pretrained(model_id)

    model.eval()
    print("  Model loaded.")
    return processor, model


def run_owlv2(image: Image.Image, prompts: list, processor, model,
              img_path: Path = None, model_id: str = "") -> list:
    """Run OWLv2 with all prompts; return list of (x1,y1,x2,y2,score,label).

    All detections (threshold=0.0) are returned so callers can sweep.
    Caches per (image path, model, sorted prompts).
    """
    import torch

    if img_path is not None:
        cached = _load_cached(img_path, model_id, prompts)
        if cached is not None:
            return [tuple(d) for d in cached]

    texts = [prompts]
    inputs = processor(text=texts, images=[image], return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        threshold=0.0,
        target_sizes=[image.size[::-1]],
        text_labels=[prompts],
    )[0]

    detections = []
    for score, label, box in zip(
        results["scores"].tolist(),
        results["text_labels"],
        results["boxes"].tolist(),
    ):
        x1, y1, x2, y2 = [int(round(c)) for c in box]
        detections.append((x1, y1, x2, y2, score, label))

    if img_path is not None:
        _save_cached(img_path, model_id, prompts, detections)

    return detections


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def infer_all_images(gt, image_dir, all_prompts, processor, model,
                     model_id, verbose):
    """Run OWLv2 with all_prompts on every image; return {filename: [dets]}."""
    filenames = sorted(gt.keys())
    all_dets = {}
    n_cached = 0

    for i, filename in enumerate(filenames):
        img_path = image_dir / filename
        if not img_path.exists():
            print(f"  [SKIP] {filename} not found")
            continue

        cached = _load_cached(img_path, model_id, all_prompts)
        if cached is not None:
            all_dets[filename] = [tuple(d) for d in cached]
            n_cached += 1
            print(f"\r  [{i+1:3d}/{len(filenames)}] {filename:40s} (cached)", end="", flush=True)
            continue

        print(f"\r  [{i+1:3d}/{len(filenames)}] {filename:40s}", end="", flush=True)
        img = load_image(img_path)
        dets = run_owlv2(img, all_prompts, processor, model,
                         img_path=img_path, model_id=model_id)
        all_dets[filename] = dets

    print(f"\n  Done. {n_cached}/{len(filenames)} from cache.")
    return all_dets


def score_image(filename, dets_all, has_mosaic, gt_px, prompt_set, iou_threshold):
    """Given all detections for one image, compute the score for a prompt set.

    Returns a float score (positive = detection fired; negative = GT missed).
    """
    # Filter detections to prompts in this set
    dets = [d for d in dets_all if d[5] in prompt_set]

    if not dets:
        return 0.0

    if has_mosaic and gt_px:
        # For mosaic images with GT boxes: score = max confidence among GT-overlapping dets
        best = 0.0
        for x1, y1, x2, y2, score, _ in dets:
            for gt_box in gt_px:
                if iou((x1, y1, x2, y2), gt_box) >= iou_threshold:
                    best = max(best, score)
                    break
        if best > 0.0:
            return best
        # No overlap — mosaic missed; signal miss with negative
        return -max(d[4] for d in dets)
    else:
        # Non-mosaic images or whole-image mosaics: any detection counts
        return max(d[4] for d in dets)


def evaluate_prompt_set_from_cache(gt, image_dir, all_dets, prompts,
                                   iou_threshold, verbose):
    """Evaluate a named prompt set using pre-computed all_dets dict."""
    prompt_set = set(prompts)
    results = []

    for filename, has_mosaic_data in sorted(gt.items()):
        img_path = image_dir / filename
        if not img_path.exists():
            continue

        has_mosaic = has_mosaic_data["has_mosaic"]
        dets_all = all_dets.get(filename, [])

        # Need image size for GT box conversion
        img = load_image(img_path)
        w, h = img.size
        gt_px = gt_boxes_pixels(has_mosaic_data, w, h)

        score = score_image(filename, dets_all, has_mosaic, gt_px,
                            prompt_set, iou_threshold)

        results.append((filename, has_mosaic, score, dets_all))

        if verbose:
            flag = "mosaic" if has_mosaic else "non-mosaic"
            top = sorted(
                [d for d in dets_all if d[5] in prompt_set],
                key=lambda d: -d[4]
            )[:3]
            det_str = "  ".join(f"{d[5]}@{d[4]:.3f}" for d in top)
            print(f"    {filename:42s} {flag:10s}  score={score:+.4f}  {det_str}")

    return results


# ─────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────

def sweep_thresholds(results, label=""):
    flat = [(fn, lbl, max(s, 0.0)) for fn, lbl, s, *_ in results]

    all_scores = sorted(set(s for _, _, s in flat if s > 0))
    if not all_scores:
        print("  No positive scores — all detections are misses.")
        return None, 0.0

    lo, hi = min(all_scores), max(all_scores)
    step = (hi - lo) / 40 if hi != lo else 0.01
    thresholds = sorted(set(
        all_scores + [round(lo + i * step, 4) for i in range(40)]
    ))

    n_pos = sum(1 for _, lbl, _ in flat if lbl)
    n_neg = sum(1 for _, lbl, _ in flat if not lbl)

    header = f"Threshold sweep{' (' + label + ')' if label else ''}  ({n_pos} pos, {n_neg} neg)"
    print(f"\n{header}")
    print(f"{'Threshold':>10}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}  "
          f"{'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print("-" * 65)

    best_f1, best_thr = -1, None
    prev_row = None

    for thr in thresholds:
        tp = sum(1 for _, lbl, s in flat if lbl and s >= thr)
        fp = sum(1 for _, lbl, s in flat if not lbl and s >= thr)
        fn = sum(1 for _, lbl, s in flat if lbl and s < thr)
        tn = sum(1 for _, lbl, s in flat if not lbl and s < thr)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        row = (tp, fp, fn, tn)
        if row != prev_row:
            print(f"  {thr:>8.4f}  {tp:>4}  {fp:>4}  {fn:>4}  {tn:>4}    "
                  f"{prec:>5.1%}  {rec:>5.1%}  {f1:>5.3f}")
            prev_row = row
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    return best_thr, best_f1


def print_breakdown(results, threshold, iou_threshold):
    flat = [(fn, lbl, max(s, 0.0)) for fn, lbl, s, *_ in results]
    missed = [(fn, s) for fn, lbl, s in flat if lbl and s < threshold]
    fps    = [(fn, s) for fn, lbl, s in flat if not lbl and s >= threshold]
    tps    = [(fn, s) for fn, lbl, s in flat if lbl and s >= threshold]

    print(f"\n--- At threshold={threshold:.4f} (IoU>={iou_threshold}) ---")
    print(f"  Detected ({len(tps)} TP):")
    for fn, s in sorted(tps, key=lambda x: -x[1]):
        print(f"    {fn:42s}  score={s:+.4f}")
    print(f"\n  Missed ({len(missed)} FN):")
    for fn, s in sorted(missed, key=lambda x: x[1]):
        print(f"    {fn:42s}  score={s:+.4f}")
    print(f"\n  False positives ({len(fps)} FP):")
    for fn, s in sorted(fps, key=lambda x: -x[1]):
        print(f"    {fn:42s}  score={s:+.4f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    print("Loading ground truth...")
    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)
    n_mosaic = sum(1 for v in gt.values() if v["has_mosaic"])
    n_total  = len(gt)
    print(f"  {n_total} images, {n_mosaic} mosaics, {n_total - n_mosaic} non-mosaics")
    print(f"  Combined prompts ({len(ALL_PROMPTS)}): {ALL_PROMPTS}")

    if args.no_cache and CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print("  OWLv2 cache cleared.")

    print(f"\nLoading model: {args.model}")
    processor, model = load_owlv2(args.model)

    print("\n" + "=" * 70)
    print("BASELINE (YOLO/DINO conf=0.10): TP=18/28 (64%), FP=22/94 (23%), F1=0.529")
    print("=" * 70)

    # ── Single inference pass for all images ──
    print(f"\nRunning OWLv2 on {n_total} images (combined {len(ALL_PROMPTS)}-prompt pass)...")
    all_dets = infer_all_images(
        gt, IMAGE_DIR, ALL_PROMPTS, processor, model,
        model_id=args.model, verbose=args.verbose,
    )

    # ── Evaluate each prompt set ──
    prompt_sets_to_eval = (
        {args.prompt_set: PROMPT_SETS[args.prompt_set]}
        if args.prompt_set
        else PROMPT_SETS
    )

    all_best = []
    for name, prompts in prompt_sets_to_eval.items():
        print(f"\n{'=' * 70}")
        print(f"Prompt set '{name}': {prompts}")
        print("=" * 70)

        results = evaluate_prompt_set_from_cache(
            gt, IMAGE_DIR, all_dets, prompts,
            iou_threshold=args.iou_threshold,
            verbose=args.verbose,
        )

        if args.threshold is not None:
            print_breakdown(results, args.threshold, args.iou_threshold)
            flat = [(fn, lbl, max(s, 0.0)) for fn, lbl, s, *_ in results]
            thr = args.threshold
            tp = sum(1 for _, lbl, s in flat if lbl and s >= thr)
            fp = sum(1 for _, lbl, s in flat if not lbl and s >= thr)
            fn_ = sum(1 for _, lbl, s in flat if lbl and s < thr)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn_) if (tp + fn_) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            print(f"\n  Prec={prec:.1%}  Rec={rec:.1%}  F1={f1:.3f}")
            all_best.append((name, thr, f1, tp, fp))
        else:
            best_thr, best_f1 = sweep_thresholds(results, label=name)
            if best_thr is not None:
                print(f"\nBest F1={best_f1:.3f} at threshold={best_thr:.4f}")
                print_breakdown(results, best_thr, args.iou_threshold)
                flat = [(fn, lbl, max(s, 0.0)) for fn, lbl, s, *_ in results]
                tp = sum(1 for _, lbl, s in flat if lbl and s >= best_thr)
                fp = sum(1 for _, lbl, s in flat if not lbl and s >= best_thr)
                all_best.append((name, best_thr, best_f1, tp, fp))

    if len(all_best) > 1:
        print(f"\n{'=' * 70}")
        print("SUMMARY — best F1 per prompt set:")
        print(f"{'Prompt set':>12}  {'Thr':>6}  {'F1':>6}  {'TP':>4}  {'FP':>4}")
        print("-" * 45)
        for name, thr, f1, tp, fp in sorted(all_best, key=lambda x: -x[2]):
            print(f"  {name:>10}  {thr:>6.4f}  {f1:>6.3f}  {tp:>4}  {fp:>4}")
        print(f"\nBaseline (YOLO/DINO conf=0.10):  F1=0.529  TP=18  FP=22")


if __name__ == "__main__":
    main()
