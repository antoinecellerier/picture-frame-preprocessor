#!/usr/bin/env python3
"""Generate visual review composites for Sonnet-based crop inspection.

For each test image, creates a composite showing:
  [Original with detection boxes + GT] | [Primary crop] | [Multi-crops...]

Also writes metadata JSON so the reviewing agent knows what to look for.

Usage:
    venv/bin/python scripts/generate_visual_review.py [--filter multi-crop|misses|all] [--images IMG1 IMG2 ...]
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

INPUT_DIR = BASE / "test_real_images" / "input"
GT_PATH = BASE / "test_real_images" / "ground_truth_annotations.json"
ART_CLASS_GT = BASE / "test_real_images" / "art_class_ground_truth.json"
OUTPUT_DIR = Path("/tmp/visual_review")

# Composite layout
THUMB_H = 600       # Height of each panel in the composite
LABEL_H = 18        # Height reserved for text labels
PAD = 4             # Padding between panels
BG = (30, 30, 30)   # Background color


def load_gt():
    """Load ground truth annotations and art class labels."""
    gt_boxes = {}
    if GT_PATH.exists():
        with open(GT_PATH) as f:
            data = json.load(f)
            for entry in data:
                # Use manual_boxes[0].bbox as the ground truth box
                boxes = entry.get("manual_boxes", [])
                if boxes:
                    gt_boxes[entry["filename"]] = boxes[0]["bbox"]

    art_classes = {}
    if ART_CLASS_GT.exists():
        with open(ART_CLASS_GT) as f:
            data = json.load(f)
            # Format: {"filename": {"primary_class": "mural", ...}, ...}
            if isinstance(data, dict):
                for fname, info in data.items():
                    art_classes[fname] = info.get("primary_class", "")
            else:
                for entry in data:
                    art_classes[entry["filename"]] = entry.get("art_class", "")

    return gt_boxes, art_classes


def resize_to_height(img, target_h):
    """Resize image to target height, preserving aspect ratio."""
    w, h = img.size
    scale = target_h / h
    return img.resize((int(w * scale), target_h), Image.LANCZOS)


def _get_font(size=12):
    """Get a TrueType font, falling back to default."""
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def draw_box(draw, bbox, color, label=None, width=2, font=None):
    """Draw a labeled bounding box with a readable text background."""
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        f = font or _get_font(12)
        tb = draw.textbbox((x1 + 2, y1 + 2), label, font=f)
        pad = 3
        draw.rectangle([tb[0] - pad, tb[1] - pad, tb[2] + pad, tb[3] + pad],
                        fill=(0, 0, 0, 180))
        draw.text((x1 + 2, y1 + 2), label, fill=color, font=f)


def create_scene_composite(original, multi_crops, detections,
                           gt_bbox, art_class, filename, overlaps=None):
    """Create a scene overview composite for missed opportunity assessment.

    Shows the original photo with detection boxes, GT box, and crop region
    outlines — but NO crop panels. This is used to find art subjects that
    aren't being captured at all.
    """
    panel_h = THUMB_H
    font = _get_font(13)
    label_font = _get_font(14)

    orig_panel = resize_to_height(original.copy(), panel_h)
    orig_w, orig_h = original.size
    scale = panel_h / orig_h
    draw = ImageDraw.Draw(orig_panel)

    # Draw GT box in green
    if gt_bbox:
        gx1, gy1, gx2, gy2 = gt_bbox
        draw.rectangle(
            [gx1 * scale, gy1 * scale, gx2 * scale, gy2 * scale],
            outline=(0, 220, 0), width=3)
        draw_box(draw, (gx1 * scale, gy1 * scale, gx2 * scale, gy2 * scale),
                 (0, 220, 0), "GT", width=3, font=font)

    # Draw detection boxes
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        is_primary = det.get('is_primary', False)
        color = (255, 255, 0) if is_primary else (100, 100, 255)
        label = f"{det['class'][:20]} {det['conf']:.2f}"
        draw_box(draw, (x1 * scale, y1 * scale, x2 * scale, y2 * scale),
                 color, label, font=font)

    # Draw crop region outlines
    crop_colors = [
        (255, 128, 0), (0, 200, 255), (255, 0, 255),
        (128, 255, 0), (255, 200, 0),
    ]
    for i, entry in enumerate(multi_crops or []):
        eff = entry[4] if len(entry) > 4 else None
        if eff is None:
            continue
        ex1, ey1, ex2, ey2 = eff
        color = crop_colors[i % len(crop_colors)]
        draw.rectangle(
            [ex1 * scale, ey1 * scale, ex2 * scale, ey2 * scale],
            outline=color, width=2)
        draw_box(draw, (ex1 * scale, ey2 * scale - 18, ex2 * scale, ey2 * scale),
                 color, f"C{i+1}", width=0, font=font)

    # Assemble
    tb = label_font.getbbox("Ay")
    label_row_h = (tb[3] - tb[1]) + 6
    overlap_lines = []
    if overlaps:
        overlap_lines = [f"{k}: {int(v*100)}%" for k, v in overlaps.items()]
    bottom_row_h = label_row_h if overlap_lines else 0
    total_h = label_row_h + panel_h + bottom_row_h

    composite = Image.new("RGB", (orig_panel.width, total_h), BG)
    composite.paste(orig_panel, (0, label_row_h))
    cdraw = ImageDraw.Draw(composite)
    header = f"{filename}  [{art_class}]  {orig_w}x{orig_h}"
    cdraw.text((2, 2), header, fill=(200, 200, 200), font=label_font)

    if overlap_lines:
        overlap_text = "  |  ".join(overlap_lines)
        cdraw.text((4, label_row_h + panel_h + 2), overlap_text,
                   fill=(180, 180, 180), font=label_font)

    return composite


def run(filter_mode="all", image_names=None):
    """Generate visual review composites."""
    from frame_prep.cli import create_detector, create_cropper
    from frame_prep.pipeline import run_detection_pipeline
    from frame_prep import defaults

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detector = create_detector(
        single_model=False, ensemble=False, model=None,
        confidence=None, no_two_pass=False, verbose=False)
    cropper = create_cropper(
        width=defaults.TARGET_WIDTH, height=defaults.TARGET_HEIGHT,
        zoom=defaults.ZOOM_FACTOR)

    gt_boxes, art_classes = load_gt()

    # Collect image paths
    all_images = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))

    if image_names:
        # Filter to specific images
        name_set = set(image_names)
        all_images = [p for p in all_images
                      if p.name in name_set or p.stem in name_set]

    metadata = []
    for i, img_path in enumerate(all_images):
        name = img_path.name
        img = ImageOps.exif_transpose(Image.open(img_path))
        w, h = img.size

        # Run detection pipeline
        result = run_detection_pipeline(
            img, detector, cropper, image_path=str(img_path))
        dets = result.filtered_detections
        focal_dets = result.focal_detections
        primary = result.primary

        # Generate crops
        primary_crop = None
        multi_crop_data = []
        if primary and result.art_score >= defaults.MIN_ART_SCORE:
            # Primary crop
            primary_crop = cropper.crop_image(
                img, dets, focal_detections=focal_dets)
            primary_zoom = cropper.last_zoom_applied

            # Multi-crops — include effective visible region for each
            multi_results = cropper.crop_all_subjects(
                img, dets, focal_detections=focal_dets)
            for crop_img, det, zoom in multi_results:
                cw = cropper._calculate_crop_window((w, h), det.center)
                eff = cropper._effective_visible_region(cw, det.bbox, det.center)
                multi_crop_data.append((crop_img, zoom, det.class_name, det.center, eff))

        # Check GT
        gt_bbox = gt_boxes.get(name)
        art_class = art_classes.get(name, "unknown")
        fills_frame = (primary is not None and
                       cropper.primary_fills_frame(primary.bbox, (w, h)))

        # Filter
        has_multi = len(multi_crop_data) >= 2
        is_miss = gt_bbox is not None and primary is not None
        # Simple IoU check for miss detection
        if is_miss and gt_bbox:
            from frame_prep.detector import calculate_iou
            crop_box = cropper.last_crop_box
            if crop_box:
                is_miss = calculate_iou(crop_box, gt_bbox) < 0.3

        if filter_mode == "multi-crop" and not has_multi:
            continue
        if filter_mode == "misses" and not is_miss:
            continue

        # Build detection metadata
        det_info = []
        for d in dets:
            det_info.append({
                'class': d.class_name,
                'conf': round(d.confidence, 3),
                'bbox': list(d.bbox),
                'is_primary': d is primary,
                'area_pct': round((d.bbox[2]-d.bbox[0]) * (d.bbox[3]-d.bbox[1]) / (w*h) * 100, 1),
            })

        # Compute pairwise overlaps first (needed for scene image)
        eff_regions_pre = [e[4] for e in multi_crop_data if len(e) > 4 and e[4]]
        overlaps_pre = {}
        for ci in range(len(eff_regions_pre)):
            for cj in range(ci + 1, len(eff_regions_pre)):
                a, b = eff_regions_pre[ci], eff_regions_pre[cj]
                ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
                ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area_a = (a[2]-a[0]) * (a[3]-a[1])
                    area_b = (b[2]-b[0]) * (b[3]-b[1])
                    union = area_a + area_b - inter
                    iou = round(inter / union, 2) if union > 0 else 0
                else:
                    iou = 0
                overlaps_pre[f"C{ci+1}-C{cj+1}"] = iou

        # Save scene image with overlaps
        scene_img = create_scene_composite(
            img, multi_crop_data, det_info, gt_bbox, art_class, name,
            overlaps=overlaps_pre)
        scene_path = OUTPUT_DIR / f"{img_path.stem}_scene.jpg"
        if scene_img:
            scene_img.save(str(scene_path), quality=85)

        # Save each crop as a separate file at target resolution with label
        crop_label_font = _get_font(11)
        crop_paths = []
        for ci, entry_data in enumerate(multi_crop_data):
            raw_crop = entry_data[0]
            # Resize to target dimensions like the preprocessor does
            if raw_crop.size != (defaults.TARGET_WIDTH, defaults.TARGET_HEIGHT):
                crop_img = raw_crop.resize(
                    (defaults.TARGET_WIDTH, defaults.TARGET_HEIGHT), Image.LANCZOS)
            else:
                crop_img = raw_crop.copy()
            zoom = entry_data[1]
            cls = entry_data[2]
            eff = entry_data[4] if len(entry_data) > 4 else None
            # Draw label at top of crop image
            draw = ImageDraw.Draw(crop_img)
            label = f"C{ci+1}: {cls[:20]} {zoom:.1f}x"
            if eff:
                label += f" [{eff[0]},{eff[1]},{eff[2]},{eff[3]}]"
            tb = draw.textbbox((4, 4), label, font=crop_label_font)
            draw.rectangle([tb[0]-2, tb[1]-2, tb[2]+2, tb[3]+2], fill=(0, 0, 0, 180))
            draw.text((4, 4), label, fill=(255, 255, 255), font=crop_label_font)
            crop_file = OUTPUT_DIR / f"{img_path.stem}_C{ci+1}.jpg"
            crop_img.save(str(crop_file), quality=90)
            crop_paths.append({
                'path': str(crop_file),
                'label': f"C{ci+1}",
                'class': cls,
                'zoom': round(zoom, 2),
                'region': list(eff) if eff else None,
            })

        overlaps = overlaps_pre

        entry = {
            'filename': name,
            'crop_files': crop_paths,
            'scene_image': str(scene_path) if scene_img else None,
            'overlaps': overlaps,
            'size': [w, h],
            'art_class': art_class,
            'primary': {
                'class': primary.class_name,
                'confidence': round(primary.confidence, 3),
                'bbox': list(primary.bbox),
            } if primary else None,
            'primary_fills_frame': fills_frame,
            'n_detections': len(dets),
            'n_focal': len(focal_dets),
            'n_crops': len(multi_crop_data),
            'crop_details': [
                {'class': entry[2], 'zoom': round(entry[1], 2)}
                for entry in multi_crop_data
            ],
            'gt_bbox': gt_bbox,
            'art_score': round(result.art_score, 3),
            'vlm_primary': result.vlm_primary,
        }
        metadata.append(entry)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(all_images)}...")

    # Write metadata
    meta_path = OUTPUT_DIR / "review_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nGenerated {len(metadata)} review composites in {OUTPUT_DIR}/")
    print(f"Metadata: {meta_path}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", choices=["all", "multi-crop", "misses"],
                        default="all")
    parser.add_argument("--images", nargs="*", help="Specific image names")
    args = parser.parse_args()
    run(filter_mode=args.filter, image_names=args.images)
