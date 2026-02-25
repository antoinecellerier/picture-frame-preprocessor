#!/usr/bin/env python3
"""Generate initial art_class_ground_truth.json from existing annotation data.

Reads:
  - test_real_images/ground_truth_annotations.json  (correct_detections class names)
  - test_real_images/mosaic_ground_truth.json        (authoritative mosaic labels)

Writes:
  - test_real_images/art_class_ground_truth.json

Art classes:
  painting     - 2-D art: framed paintings, portraits, drawings, figurative art
  mural        - Large-scale wall art, frescoes, narrative wall paintings
  mosaic       - Tile / ceramic / stone / glass mosaic art
  sculpture    - 3D free-standing: statues, busts, figurines, reliefs
  ceramic      - Vases, pottery, decorative vessels
  street_art   - Outdoor / graffiti / public street art
  installation - Mixed media, exhibit case, art installation
  non_art      - No art subject (would be filtered by pipeline)
  unknown      - Not yet annotated (to be filled by annotator tool)

Usage:
    venv/bin/python scripts/generate_art_class_gt.py
    venv/bin/python scripts/generate_art_class_gt.py --output path/to/output.json
"""

import argparse
import json
from pathlib import Path

GT_PATH = Path("test_real_images/ground_truth_annotations.json")
MOSAIC_GT_PATH = Path("test_real_images/mosaic_ground_truth.json")
OUTPUT_PATH = Path("test_real_images/art_class_ground_truth.json")

# Map detected class names → canonical art class (best-effort, to be verified by human)
# 'unknown' means we can't confidently infer from the class name alone.
CLASS_INFERENCE = {
    # mosaics
    "mosaic": "mosaic",
    "tile art": "mosaic",
    "mosaic mural art": "mosaic",
    # murals (large scale wall paintings — different from framed paintings)
    "mural": "mural",
    "mural wall art": "mural",
    # 2D paintings / drawings
    "painted figure": "painting",
    "painting": "painting",
    "framed artwork": "painting",
    "art": "painting",
    "artwork": "painting",
    "decorative art": "painting",
    "decorative": "painting",
    "wall art": "painting",
    # sculptures / 3D objects
    "sculpture": "sculpture",
    "sculpture statue": "sculpture",
    "sculpture on pedestal": "sculpture",
    "statue on display": "sculpture",
    "figure figne": "sculpture",      # DSC_4291 — clay figurine
    "figure figurine": "sculpture",
    "painted figure figurine": "sculpture",
    "painted figurene": "sculpture",
    "sculpture painted figure": "sculpture",
    "statue painted figure": "sculpture",
    "sculpture statue painted figure": "sculpture",
    # ceramics
    "vase": "ceramic",
    # installations
    "art installation": "installation",
    "artistic object": "installation",
    # street art / outdoor
    "art installation street art": "street_art",
    "street art": "street_art",
    # ambiguous class combos → unknown, let human decide
    "art piece": "unknown",
    "exhibit": "unknown",
}

# COCO/YOLO classes that indicate "no art" or misdetections
NON_ART_CLASSES = {
    "person", "bird", "elephant", "horse", "kite", "bench",
    "traffic light", "tv", "suitcase", "handbag", "skis",
    "decorated sign",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(OUTPUT_PATH))
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing file (default: merge with existing)")
    args = p.parse_args()

    output_path = Path(args.output)

    print(f"Reading {GT_PATH}...")
    with open(GT_PATH) as f:
        gt_list = json.load(f)

    print(f"Reading {MOSAIC_GT_PATH}...")
    with open(MOSAIC_GT_PATH) as f:
        mosaic_gt = json.load(f)

    # Load existing output if merging
    existing = {}
    if not args.overwrite and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing annotations from {output_path}")

    result = {}
    stats = {"mosaic": 0, "inferred": 0, "unknown": 0, "kept_existing": 0}

    for entry in gt_list:
        fn = entry["filename"]

        # Keep existing human annotation if present and not unknown
        if fn in existing and existing[fn].get("primary_class", "unknown") != "unknown":
            result[fn] = existing[fn]
            stats["kept_existing"] += 1
            continue

        primary_class = "unknown"
        confidence = "low"
        notes = ""

        # --- Mosaic GT is authoritative ---
        if fn in mosaic_gt and mosaic_gt[fn]["has_mosaic"]:
            primary_class = "mosaic"
            confidence = "high"
            stats["mosaic"] += 1

        # --- Infer from correct_detections class names ---
        elif entry.get("correct_detections"):
            # Take the highest-confidence detection's class name
            best = max(entry["correct_detections"], key=lambda d: d.get("confidence", 0))
            cn = best.get("class_name", "").lower().strip()

            if cn in NON_ART_CLASSES:
                # Non-art COCO class detected as primary — likely misdetection
                primary_class = "unknown"
                notes = f"model detected: {cn} (may be misdetection or non-art)"
                stats["unknown"] += 1
            elif cn in CLASS_INFERENCE:
                primary_class = CLASS_INFERENCE[cn]
                confidence = "medium"
                notes = f"inferred from detected class: {cn}"
                stats["inferred"] += 1
            else:
                primary_class = "unknown"
                notes = f"unrecognised detected class: {cn}"
                stats["unknown"] += 1
        else:
            stats["unknown"] += 1

        result[fn] = {
            "primary_class": primary_class,
            "confidence": confidence,  # 'high' (human), 'medium' (inferred), 'low' (unknown)
            "notes": notes,
        }

    print(f"\nResults:")
    print(f"  Total images:     {len(result)}")
    print(f"  Mosaic (auth.):   {stats['mosaic']}")
    print(f"  Class inferred:   {stats['inferred']}")
    print(f"  Unknown/needs human: {stats['unknown']}")
    print(f"  Kept existing:    {stats['kept_existing']}")

    # Count by class
    from collections import Counter
    class_counts = Counter(v["primary_class"] for v in result.values())
    print("\nClass distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls:15s}: {count}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"\nWritten to {output_path}")
    need_annotation = sum(1 for v in result.values() if v["primary_class"] == "unknown")
    print(f"Images needing human annotation: {need_annotation}")


if __name__ == "__main__":
    main()
