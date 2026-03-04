#!/usr/bin/env python3
"""Evaluate Qwen3-VL for art subject detection on failing images.

The pipeline currently hits 107/122 (88%) IoU.  15 images still miss.
This script tests whether Qwen3-VL's explicit 2D grounding capability
can find objects YOLO/DINO miss, and whether its VQA mode can distinguish
real people from depicted persons (category C misses).

Modes:
  grounding  Ask the model to locate art objects; compare bbox vs GT.
  vqa        Ask "real person or depicted in artwork?" for category C images.
  both       Run both (default).

Usage:
    venv/bin/python scripts/evaluate_qwen3vl.py --verbose
    venv/bin/python scripts/evaluate_qwen3vl.py --mode grounding --all-images
    venv/bin/python scripts/evaluate_qwen3vl.py --model Qwen/Qwen3-VL-2B-Instruct
    venv/bin/python scripts/evaluate_qwen3vl.py --images DSC_4162.JPG DSC_4168.JPG
    venv/bin/python scripts/evaluate_qwen3vl.py --no-cache --verbose
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

GROUND_TRUTH_PATH = Path("test_real_images/art_class_ground_truth.json")
IMAGE_DIR = Path("test_real_images/input")
CACHE_DIR = Path("cache/qwen3vl")

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

# The 15 images that currently miss IoU threshold (stem-based matching)
MISS_STEMS = [
    # A: borderline box
    "DSC_4101_0", "DSC_4388",
    # B: wrong detection beats correct one
    "DSC_4371", "DSC_4020", "DSC_4385", "DSC_3401", "20200525",
    # C: person-as-art (VQA target)
    "20210910", "DSC_0001_BURST",
    # D: detection failures
    "20210911", "DSC_4162", "DSC_4168", "DSC_4291", "DSC_4311", "DSC_4312",
]

# Stems of category C images (real person vs depicted person)
CATEGORY_C_STEMS = ["20210910", "DSC_0001_BURST"]

# Art classes to probe in grounding mode
ART_CLASSES = [
    "artwork",
    "painting",
    "mural",
    "mosaic",
    "sculpture",
    "street art",
    "art installation",
]

GROUNDING_PROMPT = (
    "Locate every instance that belongs to the following categories: {classes}.\n"
    "Output a JSON list where each item has \"bbox_2d\": [x1, y1, x2, y2] "
    "with coordinates in range 0-1000 and a \"label\" field."
)

VQA_PROMPT = (
    "I will show you an image. Some people in the image may be real people "
    "standing in a gallery, while others may be people DEPICTED in artwork "
    "(e.g., painted in a mural, sculpted in stone, shown in a painting). "
    "Look at the most prominent figure in this image and answer: "
    "Is this figure a real person or a person depicted in artwork?\n"
    "Answer with exactly one of: \"real person\" or \"depicted in artwork\""
)

IOU_HIT_THRESHOLD = 0.5
MAX_NEW_TOKENS_GROUNDING = 1024
MAX_NEW_TOKENS_VQA = 64

# Max image size for inference (pixels on longest side).
# Full-resolution images (3000-4000px) generate thousands of visual tokens
# and make CPU inference impractical (hours per image).
# 1024px is a good balance for CPU; use 2048+ with GPU.
DEFAULT_MAX_IMAGE_SIZE = 1024


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Qwen3-VL for art detection")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Qwen3-VL model ID (default: {DEFAULT_MODEL})")
    p.add_argument("--mode", choices=["grounding", "vqa", "both"], default="both",
                   help="Evaluation mode (default: both)")
    p.add_argument("--images", nargs="+", metavar="FILENAME",
                   help="Specific image filenames to test (overrides default miss set)")
    p.add_argument("--all-images", action="store_true",
                   help="Run on all 122 GT images instead of the 15-image miss set")
    p.add_argument("--max-image-size", type=int, default=DEFAULT_MAX_IMAGE_SIZE,
                   metavar="PX",
                   help=f"Resize images to at most this many pixels on the longest side "
                        f"before inference (default: {DEFAULT_MAX_IMAGE_SIZE}). "
                        f"Full-res (3000-4000px) is impractical on CPU.")
    p.add_argument("--no-cache", action="store_true",
                   help="Ignore cached results and force re-inference")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print raw model output per image")
    return p.parse_args()


# ─────────────────────────────────────────────
# IMAGE / CACHE HELPERS
# ─────────────────────────────────────────────

def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return ImageOps.exif_transpose(img)


def resize_image(img: Image.Image, max_size: int) -> Image.Image:
    """Resize so the longest side is at most max_size; preserve aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    if w >= h:
        new_w, new_h = max_size, int(h * max_size / w)
    else:
        new_w, new_h = int(w * max_size / h), max_size
    return img.resize((new_w, new_h), Image.LANCZOS)


def _cache_key(img_path: Path, model_id: str, prompt_mode: str,
               max_image_size: int) -> str:
    stat = img_path.stat()
    key = (f"{img_path.absolute()}:{stat.st_size}:{stat.st_mtime}"
           f":{model_id}:{prompt_mode}:{max_image_size}")
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def _load_cached(img_path: Path, model_id: str, prompt_mode: str,
                 max_image_size: int):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(img_path, model_id, prompt_mode, max_image_size)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None


def _save_cached(img_path: Path, model_id: str, prompt_mode: str,
                 max_image_size: int, result: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(img_path, model_id, prompt_mode, max_image_size)
    cache_file = CACHE_DIR / f"{key}.json"
    with open(cache_file, "w") as f:
        json.dump(result, f)


# ─────────────────────────────────────────────
# IoU / GT HELPERS
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


def gt_boxes_norm(gt_entry):
    """Return GT boxes as (x1,y1,x2,y2) in 0-1 normalized coords."""
    return [
        (b["x1_norm"], b["y1_norm"], b["x2_norm"], b["y2_norm"])
        for b in gt_entry.get("boxes", [])
    ]


def best_iou_norm(pred_norm, gt_boxes_norm_list):
    """Compute best IoU between a predicted box (0-1) and GT boxes (0-1)."""
    if not gt_boxes_norm_list:
        return 0.0
    return max(iou(pred_norm, gt) for gt in gt_boxes_norm_list)


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

def load_model(model_id: str):
    """Load Qwen3-VL (or compatible VL model).

    Tries, in order:
      1. Qwen3VLForConditionalGeneration (exact class for Qwen3-VL)
      2. AutoModelForVision2Seq (generic VLM fallback)
    For each, first attempts local HF cache, then downloads if needed.
    """
    print(f"  Loading model: {model_id} ...")

    def _try_load(model_cls_name: str):
        import transformers
        from transformers import AutoProcessor

        model_cls = getattr(transformers, model_cls_name, None)
        if model_cls is None:
            raise ImportError(f"{model_cls_name} not found in installed transformers")

        try:
            processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            model = model_cls.from_pretrained(
                model_id, torch_dtype="auto", device_map="auto",
                local_files_only=True,
            )
        except (OSError, EnvironmentError):
            print("  Not cached locally — downloading from HuggingFace (this may take a while)...")
            processor = AutoProcessor.from_pretrained(model_id)
            model = model_cls.from_pretrained(
                model_id, torch_dtype="auto", device_map="auto",
            )
        return processor, model

    for cls_name in ("Qwen3VLForConditionalGeneration", "Qwen3_5ForConditionalGeneration",
                     "Qwen2_5_VLForConditionalGeneration"):
        try:
            processor, model = _try_load(cls_name)
            print(f"  Loaded with {cls_name}.")
            model.eval()
            return processor, model
        except Exception as e:
            print(f"  {cls_name} failed: {e}")

    raise RuntimeError(f"Could not load model {model_id} with any known class")


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

def run_inference(image: Image.Image, prompt: str, processor, model,
                  max_new_tokens: int) -> str:
    """Run a single VL inference; return raw text output."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # apply_chat_template builds the text with image placeholders
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Qwen3-VL processor expects `images` kwarg; other VLMs may differ
    try:
        inputs = processor(
            text=text,
            images=[image],
            return_tensors="pt",
        )
    except TypeError:
        # Fallback: some processors use positional or different kwargs
        inputs = processor(images=[image], text=text, return_tensors="pt")

    # Move all tensor inputs to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    input_len = inputs.get("input_ids", inputs.get("decoder_input_ids",
                           next(iter(inputs.values())))).shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    generated = output_ids[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────
# BBOX PARSING
# ─────────────────────────────────────────────

def parse_grounding_output(raw: str) -> list[dict]:
    """Parse Qwen3-VL grounding output → list of {"bbox_2d": [x1,y1,x2,y2], "label": str}.

    Coordinates are expected in 0-1000 range.
    Tries several patterns:
      1. JSON objects with "bbox_2d" key
      2. Full JSON array
      3. Inline [x1, y1, x2, y2] tuples with surrounding label context
    """
    boxes = []

    # Pattern 1: individual JSON objects {..."bbox_2d": [...], "label": "..."}
    # Also handles objects without "label"
    obj_pattern = re.compile(
        r'\{[^{}]*"bbox_2d"\s*:\s*\[([^\]]+)\][^{}]*\}', re.DOTALL
    )
    label_pattern = re.compile(r'"label"\s*:\s*"([^"]+)"')

    for m in obj_pattern.finditer(raw):
        obj_str = m.group(0)
        coords_str = m.group(1)
        try:
            coords = [float(c.strip()) for c in coords_str.split(",")]
            if len(coords) == 4:
                label_m = label_pattern.search(obj_str)
                label = label_m.group(1) if label_m else "unknown"
                boxes.append({"bbox_2d": [int(c) for c in coords], "label": label})
        except ValueError:
            continue

    if boxes:
        return boxes

    # Pattern 2: try full JSON array parse (handles markdown code blocks too)
    json_match = re.search(r'\[[\s\S]*\]', raw)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "bbox_2d" in item:
                        coords = item["bbox_2d"]
                        if isinstance(coords, list) and len(coords) == 4:
                            boxes.append({
                                "bbox_2d": [int(c) for c in coords],
                                "label": item.get("label", "unknown"),
                            })
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    return boxes


def boxes_to_norm(boxes_1000: list[dict]) -> list[tuple]:
    """Convert 0-1000 bbox list to 0-1 normalized tuples (x1,y1,x2,y2)."""
    result = []
    for b in boxes_1000:
        coords = b["bbox_2d"]
        result.append(tuple(c / 1000.0 for c in coords))
    return result


# ─────────────────────────────────────────────
# IMAGE SELECTION
# ─────────────────────────────────────────────

def select_images(gt: dict, args) -> list[str]:
    """Return list of GT filenames to evaluate."""
    if args.images:
        # Explicit list — match against GT keys
        requested = set(args.images)
        matched = [k for k in gt if k in requested]
        missing = requested - set(matched)
        if missing:
            print(f"  Warning: {missing} not found in GT, skipping")
        return sorted(matched)

    if args.all_images:
        return sorted(gt.keys())

    # Default: the 15 miss images (match by stem prefix)
    result = []
    for k in sorted(gt.keys()):
        stem = k.rsplit(".", 1)[0]  # strip extension
        if any(stem == s or stem.startswith(s) for s in MISS_STEMS):
            result.append(k)
    return result


def is_category_c(filename: str) -> bool:
    stem = filename.rsplit(".", 1)[0]
    return any(stem == s or stem.startswith(s) for s in CATEGORY_C_STEMS)


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def eval_grounding(filename, gt_entry, img, processor, model,
                   model_id, no_cache, verbose, img_path, max_image_size):
    """Run grounding mode on one image; return result dict."""
    cached = None if no_cache else _load_cached(img_path, model_id, "grounding",
                                                max_image_size)

    if cached is not None:
        raw = cached["raw"]
        boxes_parsed = cached["boxes"]
        from_cache = True
    else:
        img_resized = resize_image(img, max_image_size)
        classes_str = ", ".join(ART_CLASSES)
        prompt = GROUNDING_PROMPT.format(classes=classes_str)
        raw = run_inference(img_resized, prompt, processor, model,
                            MAX_NEW_TOKENS_GROUNDING)
        boxes_parsed = parse_grounding_output(raw)
        _save_cached(img_path, model_id, "grounding", max_image_size,
                     {"raw": raw, "boxes": boxes_parsed})
        from_cache = False

    gt_boxes = gt_boxes_norm(gt_entry)
    pred_norm = boxes_to_norm(boxes_parsed)

    best_iou_val = 0.0
    best_label = boxes_parsed[0].get("label", "") if boxes_parsed else ""
    for box_norm, box_raw in zip(pred_norm, boxes_parsed):
        score = best_iou_norm(box_norm, gt_boxes)
        if score > best_iou_val:
            best_iou_val = score
            best_label = box_raw.get("label", "")

    hit = best_iou_val >= IOU_HIT_THRESHOLD

    if verbose:
        cache_tag = " (cached)" if from_cache else ""
        print(f"\n  [{filename}]{cache_tag}")
        print(f"    GT class: {gt_entry.get('primary_class', '?')}  "
              f"GT boxes: {len(gt_boxes)}")
        print(f"    Raw output: {raw[:300]}{'...' if len(raw) > 300 else ''}")
        print(f"    Parsed boxes ({len(boxes_parsed)}): "
              f"{[b['label'] for b in boxes_parsed[:5]]}")
        print(f"    Best IoU: {best_iou_val:.3f}  label={best_label!r}  "
              f"hit={'YES' if hit else 'NO'}")

    return {
        "hit": hit,
        "best_iou": best_iou_val,
        "best_label": best_label,
        "n_boxes": len(boxes_parsed),
        "from_cache": from_cache,
    }


def eval_vqa(filename, gt_entry, img, processor, model,
             model_id, no_cache, verbose, img_path, max_image_size):
    """Run VQA mode on one image; return result dict."""
    cached = None if no_cache else _load_cached(img_path, model_id, "vqa",
                                                max_image_size)

    if cached is not None:
        raw = cached["raw"]
        from_cache = True
    else:
        img_resized = resize_image(img, max_image_size)
        raw = run_inference(img_resized, VQA_PROMPT, processor, model,
                            MAX_NEW_TOKENS_VQA)
        _save_cached(img_path, model_id, "vqa", max_image_size, {"raw": raw})
        from_cache = False

    answer_lower = raw.lower()
    depicted = "depicted in artwork" in answer_lower
    real = "real person" in answer_lower
    if depicted:
        answer = "depicted in artwork"
    elif real:
        answer = "real person"
    else:
        answer = f"ambiguous: {raw[:80]}"

    if verbose:
        cache_tag = " (cached)" if from_cache else ""
        print(f"\n  [{filename}]{cache_tag}")
        print(f"    GT class: {gt_entry.get('primary_class', '?')}")
        print(f"    VQA answer: {raw!r}")
        print(f"    Parsed: {answer}")

    return {
        "answer": answer,
        "depicted": depicted,
        "from_cache": from_cache,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    print("Loading ground truth...")
    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)
    print(f"  {len(gt)} images in GT")

    filenames = select_images(gt, args)
    print(f"  Evaluating {len(filenames)} images: {filenames[:5]}{'...' if len(filenames) > 5 else ''}")
    print(f"  Max image size: {args.max_image_size}px (images resized before inference)")

    if args.no_cache and CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print("  Qwen3-VL cache cleared.")

    do_grounding = args.mode in ("grounding", "both")
    do_vqa = args.mode in ("vqa", "both")

    print(f"\nLoading model: {args.model}")
    processor, model = load_model(args.model)

    print(f"\n{'=' * 70}")
    print(f"Baseline: 107/122 (88%) IoU hits with YOLO/DINO ensemble")
    print(f"{'=' * 70}")

    grounding_results = []
    vqa_results = []

    for i, filename in enumerate(filenames):
        gt_entry = gt[filename]
        img_path = IMAGE_DIR / filename

        if not img_path.exists():
            print(f"  [SKIP] {filename} — file not found")
            continue

        print(f"\r  [{i+1:3d}/{len(filenames)}] {filename}", end="", flush=True)
        img = load_image(img_path)

        if do_grounding:
            res = eval_grounding(filename, gt_entry, img, processor, model,
                                 args.model, args.no_cache, args.verbose,
                                 img_path, args.max_image_size)
            grounding_results.append((filename, gt_entry, res))

        if do_vqa and is_category_c(filename):
            res = eval_vqa(filename, gt_entry, img, processor, model,
                           args.model, args.no_cache, args.verbose,
                           img_path, args.max_image_size)
            vqa_results.append((filename, gt_entry, res))

    # ── Grounding summary ──────────────────────────────────────────────────
    if do_grounding and grounding_results:
        hits = sum(1 for _, _, r in grounding_results if r["hit"])
        total = len(grounding_results)

        print(f"\n\n{'=' * 70}")
        print(f"GROUNDING RESULTS  (IoU threshold={IOU_HIT_THRESHOLD})")
        print(f"{'=' * 70}")
        print(f"{'Filename':45s}  {'Class':12s}  {'Hit':4s}  {'IoU':6s}  {'Label'}")
        print("-" * 90)
        for filename, gt_entry, r in grounding_results:
            flag = "HIT " if r["hit"] else "miss"
            cls = gt_entry.get("primary_class", "?")
            print(f"  {filename:43s}  {cls:12s}  {flag}  {r['best_iou']:.3f}  {r['best_label']}")

        print(f"\nGrounding hits: {hits}/{total}")
        if hits > 0:
            print(f"If these were previously failing images, new total would be "
                  f"~{107 + hits}/122 ({(107 + hits)/122*100:.1f}%)")
            print("(actual gain depends on whether the correct subject is selected)")

    # ── VQA summary ───────────────────────────────────────────────────────
    if do_vqa:
        if vqa_results:
            depicted_correct = sum(1 for _, _, r in vqa_results if r["depicted"])

            print(f"\n{'=' * 70}")
            print(f"VQA RESULTS  (category C: person-as-art images)")
            print(f"{'=' * 70}")
            print(f"{'Filename':45s}  {'Class':12s}  {'Answer'}")
            print("-" * 85)
            for filename, gt_entry, r in vqa_results:
                cls = gt_entry.get("primary_class", "?")
                print(f"  {filename:43s}  {cls:12s}  {r['answer']}")

            print(f"\nVQA: {depicted_correct}/{len(vqa_results)} answered 'depicted in artwork' "
                  f"(correct for category C)")
        else:
            print(f"\nVQA: no category C images in the selected set")
            print(f"  Category C stems: {CATEGORY_C_STEMS}")
            print(f"  Selected files: {filenames}")


if __name__ == "__main__":
    main()
