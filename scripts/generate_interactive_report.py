#!/usr/bin/env python3
"""Generate interactive HTML report for testing the updated preprocessor with feedback capability."""

import json
import sys
from pathlib import Path
import base64
from PIL import Image, ImageDraw, ImageFont, ImageOps
import io
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from frame_prep.detector import ArtFeatureDetector, EnsembleDetector, OptimizedEnsembleDetector
from frame_prep.cropper import SmartCropper
from frame_prep import defaults
from frame_prep.defaults import MIN_ART_SCORE


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def draw_boxes_on_image(image_path, detections, ground_truth_boxes=None, primary=None, max_width=800):
    """Draw detected and ground truth bounding boxes on image."""
    try:
        img = Image.open(image_path).convert('RGB')
        # Handle EXIF rotation
        img = ImageOps.exif_transpose(img)

        scale = 1.0
        if img.width > max_width:
            scale = max_width / img.width
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)

        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = font

        # Draw ground truth boxes in blue
        if ground_truth_boxes:
            for gt_box in ground_truth_boxes:
                bbox = [int(coord * scale) for coord in gt_box]
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255), width=3)
                draw.text((x1, y2 + 5), "Ground Truth", fill=(0, 0, 255), font=small_font)

        # Draw detected boxes in green
        if detections:
            for det in detections:  # Show all detections
                bbox = [int(coord * scale) for coord in det.bbox]
                x1, y1, x2, y2 = bbox

                # Check if this is the primary detection (by bbox match)
                is_primary = primary is not None and det.bbox == primary.bbox

                # Primary detection gets thicker border and brighter color
                width = 4 if is_primary else 2
                color = (0, 255, 0) if is_primary else (0, 200, 0)

                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

                label = f"{det.class_name} {det.confidence:.2f}"
                if is_primary:
                    label = "PRIMARY: " + label

                text_bbox = draw.textbbox((x1, y1-20), label, font=font)
                draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2],
                             fill=color)
                draw.text((x1, y1-20), label, fill=(255, 255, 255), font=font)

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        img_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_data}"
    except Exception as e:
        print(f"Error drawing boxes on {image_path}: {e}")
        return None


def run_detection(image_path, detector, verbose=False):
    """Run detection on an image and return results."""
    try:
        img = Image.open(image_path)
        # Handle EXIF rotation
        img = ImageOps.exif_transpose(img)

        # Pass image_path for cache lookups
        try:
            detections = detector.detect(img, verbose=verbose, image_path=image_path)
        except TypeError:
            detections = detector.detect(img, verbose=verbose)

        # Get primary by smart selection algorithm (with score)
        primary = None
        art_score = 0.0
        if detections and hasattr(detector, 'get_primary_subject_with_score'):
            primary, art_score = detector.get_primary_subject_with_score(detections)
        elif detections:
            primary = detector.get_primary_subject(detections)

        # Get primary by confidence (old method) for comparison
        primary_by_confidence = detections[0] if detections else None

        # Check if selection algorithm chose a different primary
        selection_changed = False
        if primary and primary_by_confidence:
            selection_changed = primary.bbox != primary_by_confidence.bbox

        return {
            'all_detections': detections,
            'primary': primary,
            'primary_by_confidence': primary_by_confidence,
            'selection_changed': selection_changed,
            'count': len(detections),
            'art_score': art_score,
        }
    except Exception as e:
        print(f"Error detecting in {image_path}: {e}")
        return {'all_detections': [], 'primary': None, 'primary_by_confidence': None,
                'selection_changed': False, 'count': 0, 'art_score': 0.0}


def check_accuracy(primary, ground_truth_boxes, iou_threshold=0.3):
    """Check if primary detection matches ground truth."""
    if not primary or not ground_truth_boxes:
        return False, 0.0

    best_iou = 0.0

    for gt_box in ground_truth_boxes:
        iou = calculate_iou(primary.bbox, gt_box)
        best_iou = max(best_iou, iou)

    return best_iou >= iou_threshold, best_iou


def generate_result_image(image_path, detections, cropper, max_width=400):
    """Generate the cropped result image for comparison."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = ImageOps.exif_transpose(img)

        # Run the actual cropping logic
        cropped = cropper.crop_image(img, detections, strategy='smart')

        # Get crop info
        zoom_applied = cropper.last_zoom_applied

        # Resize for display
        scale = 1.0
        if cropped.width > max_width:
            scale = max_width / cropped.width
            new_size = (int(cropped.width * scale), int(cropped.height * scale))
            cropped = cropped.resize(new_size, Image.LANCZOS)

        # Add zoom annotation
        draw = ImageDraw.Draw(cropped)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = ImageFont.load_default()

        zoom_text = f"Zoom: {zoom_applied:.2f}x"
        text_bbox = draw.textbbox((5, 5), zoom_text, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2],
                       fill=(0, 0, 0, 180))
        draw.text((5, 5), zoom_text, fill=(255, 255, 255), font=font)

        buffer = io.BytesIO()
        cropped.save(buffer, format='JPEG', quality=90)
        img_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_data}", zoom_applied
    except Exception as e:
        print(f"Error generating result for {image_path}: {e}")
        return None, 1.0


def generate_multi_crop_images(image_path, detections, cropper, max_width=250):
    """Generate cropped images for all viable art subjects (multi-crop display).

    Returns list of (data_uri, zoom_applied, class_name) tuples, or empty list
    if fewer than 2 viable subjects.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = ImageOps.exif_transpose(img)

        multi_results = cropper.crop_all_subjects(img, detections)
        if len(multi_results) < 2:
            return []

        output = []
        for cropped, det, zoom_applied in multi_results:
            # Resize for display
            if cropped.width > max_width:
                scale = max_width / cropped.width
                new_size = (int(cropped.width * scale), int(cropped.height * scale))
                cropped = cropped.resize(new_size, Image.LANCZOS)

            # Add annotation
            draw = ImageDraw.Draw(cropped)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
            except Exception:
                font = ImageFont.load_default()

            label = f"{det.class_name} ({zoom_applied:.1f}x)"
            text_bbox = draw.textbbox((5, 5), label, font=font)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2],
                           fill=(0, 0, 0, 180))
            draw.text((5, 5), label, fill=(255, 255, 255), font=font)

            buffer = io.BytesIO()
            cropped.save(buffer, format='JPEG', quality=90)
            img_data = base64.b64encode(buffer.getvalue()).decode()
            output.append((f"data:image/jpeg;base64,{img_data}", zoom_applied, det.class_name))

        return output
    except Exception as e:
        print(f"Error generating multi-crop for {image_path}: {e}")
        return []


def generate_report():
    """Generate interactive HTML report."""
    print("Loading ground truth annotations...")
    with open('test_real_images/ground_truth_annotations.json', 'r') as f:
        ground_truth = json.load(f)

    input_dir = Path('test_real_images/input')
    results = []

    # Create detector once (reused for all images, with caching)
    detector = OptimizedEnsembleDetector(
        confidence_threshold=defaults.CONFIDENCE_THRESHOLD,
        merge_threshold=defaults.MERGE_THRESHOLD,
        two_pass=defaults.TWO_PASS
    )

    # Create cropper for generating result images
    cropper = SmartCropper(
        target_width=defaults.TARGET_WIDTH,
        target_height=defaults.TARGET_HEIGHT,
        zoom_factor=defaults.ZOOM_FACTOR,
        use_saliency_fallback=defaults.USE_SALIENCY_FALLBACK
    )

    # Build config dict for report display and feedback export traceability
    config = {
        'detector': 'OptimizedEnsembleDetector',
        'models': {
            'yolo_world': 'yolov8m-worldv2',
            'grounding_dino': 'IDEA-Research/grounding-dino-tiny',
        },
        'confidence_threshold': defaults.CONFIDENCE_THRESHOLD,
        'merge_threshold': defaults.MERGE_THRESHOLD,
        'two_pass': defaults.TWO_PASS,
        'primary_selection': 'center-weighted scoring',
        'target_width': defaults.TARGET_WIDTH,
        'target_height': defaults.TARGET_HEIGHT,
        'zoom_factor': defaults.ZOOM_FACTOR,
        'use_saliency_fallback': defaults.USE_SALIENCY_FALLBACK,
        'yolo_world_prompts': detector._art_classes,
        'grounding_dino_prompts': detector._dino_prompts,
    }

    print(f"\nProcessing {len(ground_truth)} test images...")

    correct_count = 0
    total_with_gt = 0

    for idx, gt_entry in enumerate(ground_truth, 1):
        filename = gt_entry['filename']
        image_path = input_dir / filename
        is_not_art = gt_entry.get('not_art', False)

        if not image_path.exists():
            print(f"  [{idx}/{len(ground_truth)}] Skipping missing: {filename}")
            continue

        print(f"  [{idx}/{len(ground_truth)}] Processing: {filename}" + (" [NOT ART]" if is_not_art else ""))

        # Get ground truth boxes
        gt_boxes = []
        for box_data in gt_entry.get('manual_boxes', []):
            gt_boxes.append(box_data['bbox'])
        for det_data in gt_entry.get('correct_detections', []):
            gt_boxes.append(det_data['bbox'])

        # Run detection with optimized ensemble (uses caching, verbose for two-pass info)
        detection_result = run_detection(image_path, detector, verbose=True)

        # Check accuracy using smart primary selection
        # Exclude not_art images from accuracy denominator
        is_correct = False
        best_iou = 0.0
        if not is_not_art and gt_boxes and detection_result['primary']:
            is_correct, best_iou = check_accuracy(detection_result['primary'], gt_boxes)
            total_with_gt += 1
            if is_correct:
                correct_count += 1
        elif not is_not_art and gt_boxes:
            # Has ground truth but no primary detection — counts as incorrect
            total_with_gt += 1

        # Generate visualization
        img_with_boxes = draw_boxes_on_image(
            image_path,
            detection_result['all_detections'],
            gt_boxes,
            primary=detection_result['primary']
        )

        # Generate result/cropped image (skip for not-art images)
        result_image, zoom_applied = None, 1.0
        multi_crop_images = []
        if not is_not_art:
            result_image, zoom_applied = generate_result_image(
                image_path,
                detection_result['all_detections'],
                cropper
            )
            # Also generate multi-crop results for display
            multi_crop_images = generate_multi_crop_images(
                image_path,
                detection_result['all_detections'],
                cropper
            )

        results.append({
            'filename': filename,
            'image_with_boxes': img_with_boxes,
            'result_image': result_image,
            'zoom_applied': zoom_applied,
            'multi_crop_images': multi_crop_images,
            'detections': detection_result['all_detections'],
            'primary': detection_result['primary'],
            'primary_by_confidence': detection_result['primary_by_confidence'],
            'selection_changed': detection_result['selection_changed'],
            'detection_count': detection_result['count'],
            'has_ground_truth': len(gt_boxes) > 0,
            'is_correct': is_correct,
            'best_iou': best_iou,
            'ground_truth_boxes': gt_boxes,
            'art_score': detection_result['art_score'],
            'is_not_art': is_not_art,
        })

    accuracy = (correct_count / total_with_gt * 100) if total_with_gt > 0 else 0
    selection_changed_count = sum(1 for r in results if r['selection_changed'])
    not_art_count = sum(1 for r in results if r.get('is_not_art'))

    print(f"\n✓ Processing complete!")
    print(f"  Accuracy: {correct_count}/{total_with_gt} ({accuracy:.1f}%) (excludes {not_art_count} not-art images)")
    print(f"  Primary selection changed: {selection_changed_count}/{len(results)} images")
    print(f"\nGenerating HTML report...")

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Interactive Detection Report - Updated Zoom Logic</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            padding: 20px;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}

        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
        }}

        .header p {{
            opacity: 0.9;
            font-size: 16px;
        }}

        .config-summary {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}

        .config-summary h2 {{
            font-size: 20px;
            color: #1f2937;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}

        .config-section {{
            background: #f9fafb;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}

        .config-section h3 {{
            font-size: 14px;
            color: #667eea;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }}

        .config-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e5e7eb;
        }}

        .config-item:last-child {{
            border-bottom: none;
        }}

        .config-label {{
            color: #6b7280;
            font-size: 13px;
        }}

        .config-value {{
            color: #1f2937;
            font-weight: 600;
            font-size: 13px;
        }}

        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            text-align: center;
        }}

        .stat-value {{
            font-size: 48px;
            font-weight: bold;
            color: #667eea;
            line-height: 1;
        }}

        .stat-label {{
            font-size: 14px;
            color: #6b7280;
            margin-top: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .filters {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}

        .filter-group {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }}

        .filter-btn {{
            padding: 10px 20px;
            border: 2px solid #e5e7eb;
            background: white;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 14px;
            font-weight: 500;
        }}

        .filter-btn:hover {{
            border-color: #667eea;
            color: #667eea;
        }}

        .filter-btn.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}

        .results-grid {{
            display: grid;
            gap: 25px;
        }}

        .result-card {{
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .result-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.12);
        }}

        .result-card.correct {{
            border-left: 5px solid #10b981;
        }}

        .result-card.incorrect {{
            border-left: 5px solid #ef4444;
        }}

        .result-card.no-gt {{
            border-left: 5px solid #6b7280;
        }}

        .result-card.not-art {{
            border-left: 5px solid #f59e0b;
        }}

        .result-header {{
            padding: 20px;
            background: #f9fafb;
            border-bottom: 1px solid #e5e7eb;
        }}

        .result-title {{
            font-size: 16px;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 10px;
        }}

        .result-meta {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            font-size: 13px;
            color: #6b7280;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}

        .badge.correct {{
            background: #d1fae5;
            color: #065f46;
        }}

        .badge.incorrect {{
            background: #fee2e2;
            color: #991b1b;
        }}

        .badge.no-gt {{
            background: #e5e7eb;
            color: #374151;
        }}

        .badge.not-art {{
            background: #fef3c7;
            color: #92400e;
        }}

        .images-container {{
            display: flex;
            gap: 10px;
            background: #1f2937;
            padding: 10px;
        }}

        .image-wrapper {{
            flex: 1;
            position: relative;
        }}

        .image-wrapper.detection {{
            flex: 2;
        }}

        .image-wrapper.result {{
            flex: 1;
            max-width: 300px;
        }}

        .image-label {{
            position: absolute;
            top: 8px;
            left: 8px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .result-image {{
            width: 100%;
            display: block;
            border-radius: 4px;
        }}

        .detection-image {{
            width: 100%;
            display: block;
            border-radius: 4px;
        }}

        .result-details {{
            padding: 20px;
        }}

        .detection-info {{
            background: #f9fafb;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }}

        .detection-label {{
            font-size: 12px;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}

        .detection-value {{
            font-size: 14px;
            color: #1f2937;
            line-height: 1.5;
        }}

        .feedback-section {{
            padding-top: 15px;
            border-top: 1px solid #e5e7eb;
        }}

        .feedback-label {{
            font-size: 13px;
            font-weight: 600;
            color: #374151;
            margin-bottom: 10px;
        }}

        .feedback-buttons {{
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }}

        .feedback-btn {{
            flex: 1;
            padding: 10px;
            border: 2px solid #e5e7eb;
            background: white;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 5px;
        }}

        .feedback-btn:hover {{
            border-color: #667eea;
        }}

        .feedback-btn.selected {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}

        .feedback-textarea {{
            width: 100%;
            padding: 10px;
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            resize: vertical;
            min-height: 60px;
        }}

        .feedback-textarea:focus {{
            outline: none;
            border-color: #667eea;
        }}

        .legend {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}

        .legend h3 {{
            font-size: 16px;
            margin-bottom: 15px;
            color: #1f2937;
        }}

        .legend-items {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .legend-color {{
            width: 30px;
            height: 4px;
            border-radius: 2px;
        }}

        .export-btn {{
            background: #667eea;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .export-btn:hover {{
            background: #5568d3;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 Interactive Detection Report</h1>
        <p>Updated Adaptive Zoom Logic - {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>

    <div class="config-summary">
        <h2>⚙️ Configuration Summary</h2>
        <div class="config-grid">
            <div class="config-section">
                <h3>Detection Strategy</h3>
                <div class="config-item">
                    <span class="config-label">Ensemble</span>
                    <span class="config-value">OptimizedEnsembleDetector</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Models</span>
                    <span class="config-value">YOLO-World + Grounding DINO</span>
                </div>
                <div class="config-item">
                    <span class="config-label">YOLO-World Model</span>
                    <span class="config-value">yolov8m-worldv2</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Grounding DINO</span>
                    <span class="config-value">grounding-dino-tiny</span>
                </div>
            </div>
            <div class="config-section">
                <h3>Detection Parameters</h3>
                <div class="config-item">
                    <span class="config-label">Confidence Threshold</span>
                    <span class="config-value">{config['confidence_threshold']}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Merge Threshold (IoU)</span>
                    <span class="config-value">{config['merge_threshold']}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Two-Pass Detection</span>
                    <span class="config-value">{'Enabled' if config['two_pass'] else 'Disabled'}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Primary Selection</span>
                    <span class="config-value">{config['primary_selection']}</span>
                </div>
            </div>
            <div class="config-section">
                <h3>Cropping Strategy</h3>
                <div class="config-item">
                    <span class="config-label">Strategy</span>
                    <span class="config-value">Smart (detection-anchored)</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Target Dimensions</span>
                    <span class="config-value">{config['target_width']}x{config['target_height']}</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Max Zoom Factor</span>
                    <span class="config-value">{config['zoom_factor']}x</span>
                </div>
                <div class="config-item">
                    <span class="config-label">Saliency Fallback</span>
                    <span class="config-value">{'Enabled' if config['use_saliency_fallback'] else 'Disabled'}</span>
                </div>
            </div>
        </div>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{accuracy:.1f}%</div>
            <div class="stat-label">Accuracy (excl. not-art)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{correct_count}/{total_with_gt}</div>
            <div class="stat-label">Correct Detections</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(results)}</div>
            <div class="stat-label">Total Images</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{not_art_count}</div>
            <div class="stat-label">Not Art</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{selection_changed_count}</div>
            <div class="stat-label">Selection Changed</div>
        </div>
    </div>

    <div class="legend">
        <h3>Legend</h3>
        <div class="legend-items">
            <div class="legend-item">
                <div class="legend-color" style="background: #10b981; width: 40px; height: 4px;"></div>
                <span>Primary Detection (GREEN - thick border)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #0ea5e9; width: 40px; height: 4px;"></div>
                <span>Ground Truth (BLUE)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #6ee7b7; width: 40px; height: 4px;"></div>
                <span>Other Detections (light green)</span>
            </div>
        </div>
    </div>

    <div class="filters">
        <div class="filter-group">
            <button class="filter-btn active" onclick="filterResults('all')">All ({len(results)})</button>
            <button class="filter-btn" onclick="filterResults('correct')">Correct ({sum(1 for r in results if r['is_correct'])})</button>
            <button class="filter-btn" onclick="filterResults('incorrect')">Incorrect ({sum(1 for r in results if r['has_ground_truth'] and not r['is_correct'] and not r.get('is_not_art'))})</button>
            <button class="filter-btn" onclick="filterResults('not-art')">Not Art ({sum(1 for r in results if r.get('is_not_art'))})</button>
            <button class="filter-btn" onclick="filterResults('no-gt')">No Ground Truth ({sum(1 for r in results if not r['has_ground_truth'] and not r.get('is_not_art'))})</button>
            <button class="export-btn" onclick="exportFeedback()">📥 Export Feedback</button>
        </div>
    </div>

    <div class="results-grid" id="results-grid">
"""

    # Add each result
    for idx, result in enumerate(results):
        is_not_art = result.get('is_not_art', False)
        if is_not_art:
            status_class = 'not-art'
            status_label = 'NOT ART'
        elif not result['has_ground_truth']:
            status_class = 'no-gt'
            status_label = 'No Ground Truth'
        elif result['is_correct']:
            status_class = 'correct'
            status_label = '✓ Correct'
        else:
            status_class = 'incorrect'
            status_label = '✗ Incorrect'

        primary_info = "No detections"
        if result['primary']:
            primary_info = f"{result['primary'].class_name} (conf: {result['primary'].confidence:.3f})"
            # Show if selection algorithm chose differently than confidence-based
            if result['selection_changed'] and result['primary_by_confidence']:
                primary_info += f"<br><span style='color: #059669; font-size: 12px;'>✓ Changed from: {result['primary_by_confidence'].class_name} (conf: {result['primary_by_confidence'].confidence:.3f})</span>"

        # Show art score
        art_score = result.get('art_score', 0.0)
        art_score_info = f"<span>Art score: {art_score:.3f}</span>"
        if art_score < MIN_ART_SCORE:
            art_score_info = f"<span style='color: #ef4444;'>Art score: {art_score:.3f} (below {MIN_ART_SCORE})</span>"

        iou_info = ""
        if result['has_ground_truth'] and result['detection_count'] > 0:
            iou_info = f"<span>IoU: {result['best_iou']:.3f}</span>"

        selection_badge = ""
        if result['selection_changed']:
            selection_badge = "<span class='badge' style='background: #d1fae5; color: #065f46;'>Selection Changed</span>"

        not_art_badge = ""
        if is_not_art:
            not_art_badge = "<span class='badge not-art'>NOT ART</span>"

        multi_crop_badge = ""
        if multi_crop_images:
            multi_crop_badge = f"<span class='badge' style='background: #dbeafe; color: #1e40af;'>Multi-crop: {len(multi_crop_images)}</span>"

        # Generate result image HTML if available
        result_img_html = ""
        multi_crop_images = result.get('multi_crop_images', [])
        if multi_crop_images:
            # Show multi-crop results side by side
            for ci, (mc_uri, mc_zoom, mc_class) in enumerate(multi_crop_images, 1):
                result_img_html += f"""
                <div class="image-wrapper result">
                    <span class="image-label">Crop {ci}: {mc_class} ({mc_zoom:.1f}x)</span>
                    <img src="{mc_uri}" class="result-image" alt="Crop {ci}">
                </div>"""
        elif result['result_image']:
            result_img_html = f"""
                <div class="image-wrapper result">
                    <span class="image-label">Result ({result['zoom_applied']:.2f}x)</span>
                    <img src="{result['result_image']}" class="result-image" alt="Result">
                </div>"""

        html += f"""
        <div class="result-card {status_class}" data-status="{status_class}" data-index="{idx}">
            <div class="result-header">
                <div class="result-title">{result['filename']}</div>
                <div class="result-meta">
                    <span class="badge {status_class}">{status_label}</span>
                    {not_art_badge}
                    {selection_badge}
                    {multi_crop_badge}
                    <span>Detections: {result['detection_count']}</span>
                    {art_score_info}
                    <span>Zoom: {result['zoom_applied']:.2f}x</span>
                    {iou_info}
                </div>
            </div>

            <div class="images-container">
                <div class="image-wrapper detection">
                    <span class="image-label">Detection</span>
                    <img src="{result['image_with_boxes']}" class="detection-image" alt="{result['filename']}">
                </div>
                {result_img_html}
            </div>

            <div class="result-details">
                <div class="detection-info">
                    <div class="detection-label">Primary Detection</div>
                    <div class="detection-value">{primary_info}</div>
                </div>

                <div class="feedback-section">
                    <div class="feedback-label">Your Feedback:</div>
                    <div class="feedback-buttons">
                        <button class="feedback-btn" onclick="setFeedback({idx}, 'good')">
                            👍 Good Detection
                        </button>
                        <button class="feedback-btn" onclick="setFeedback({idx}, 'bad')">
                            👎 Poor Detection
                        </button>
                        <button class="feedback-btn" onclick="setFeedback({idx}, 'zoom')">
                            🔍 Zoom Issue
                        </button>
                        <button class="feedback-btn" onclick="setFeedback({idx}, 'other')">
                            💬 Other
                        </button>
                    </div>
                    <textarea
                        class="feedback-textarea"
                        placeholder="Optional: Add comments about detection quality, zoom appropriateness, or subject identification..."
                        onchange="updateComment({idx}, this.value)"
                    ></textarea>
                </div>
            </div>
        </div>
"""

    # Build per-image metadata for JS (detection context for feedback export)
    image_metadata = []
    for idx, result in enumerate(results):
        det_list = []
        for det in (result['detections'] or []):
            det_list.append({
                'class_name': det.class_name,
                'confidence': round(float(det.confidence), 4),
                'bbox': [int(c) for c in det.bbox],
            })
        primary_data = None
        if result['primary']:
            primary_data = {
                'class_name': result['primary'].class_name,
                'confidence': round(float(result['primary'].confidence), 4),
                'bbox': [int(c) for c in result['primary'].bbox],
            }
        image_metadata.append({
            'filename': result['filename'],
            'detection_count': result['detection_count'],
            'detections': det_list,
            'primary': primary_data,
            'is_correct': result['is_correct'],
            'best_iou': round(result['best_iou'], 4),
            'ground_truth_boxes': result['ground_truth_boxes'],
            'art_score': round(result.get('art_score', 0.0), 4),
            'is_not_art': result.get('is_not_art', False),
        })

    html += """
    </div>

    <script>
        const imageMetadata = """ + json.dumps(image_metadata) + """;
        const reportConfig = """ + json.dumps(config) + """;
        const feedbackData = {};

        function filterResults(filter) {
            const cards = document.querySelectorAll('.result-card');
            const buttons = document.querySelectorAll('.filter-btn');

            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            cards.forEach(card => {
                if (filter === 'all') {
                    card.style.display = 'block';
                } else {
                    card.style.display = card.dataset.status === filter ? 'block' : 'none';
                }
            });
        }

        function setFeedback(index, rating) {
            const fname = imageMetadata[index].filename;
            if (!feedbackData[fname]) {
                feedbackData[fname] = {};
            }
            feedbackData[fname].rating = rating;

            // Update button states
            const card = document.querySelector(`[data-index="${index}"]`);
            const buttons = card.querySelectorAll('.feedback-btn');
            buttons.forEach(btn => btn.classList.remove('selected'));

            const ratingMap = {
                'good': 0,
                'bad': 1,
                'zoom': 2,
                'other': 3
            };
            buttons[ratingMap[rating]].classList.add('selected');
        }

        function updateComment(index, comment) {
            const fname = imageMetadata[index].filename;
            if (!feedbackData[fname]) {
                feedbackData[fname] = {};
            }
            feedbackData[fname].comment = comment;
        }

        function exportFeedback() {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const outFilename = `detection_feedback_${timestamp}.json`;

            // Build full export with per-image detection context
            const feedbackWithContext = {};
            for (const [fname, fb] of Object.entries(feedbackData)) {
                const meta = imageMetadata.find(m => m.filename === fname);
                feedbackWithContext[fname] = {
                    ...fb,
                    detections: meta ? meta.detections : [],
                    primary: meta ? meta.primary : null,
                    is_correct: meta ? meta.is_correct : null,
                    best_iou: meta ? meta.best_iou : null,
                    ground_truth_boxes: meta ? meta.ground_truth_boxes : [],
                };
            }

            const exportData = {
                generated_at: new Date().toISOString(),
                config: reportConfig,
                total_images: imageMetadata.length,
                feedback_count: Object.keys(feedbackData).length,
                feedback: feedbackWithContext
            };

            const blob = new Blob([JSON.stringify(exportData, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = outFilename;
            a.click();
            URL.revokeObjectURL(url);

            alert(`Exported feedback for ${Object.keys(feedbackData).length} images`);
        }

        // Pre-mark correct detections as "good"
        document.querySelectorAll('.result-card[data-status="correct"]').forEach(card => {
            setFeedback(parseInt(card.dataset.index), 'good');
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'f' && e.ctrlKey) {
                e.preventDefault();
                exportFeedback();
            }
        });
    </script>
</body>
</html>
"""

    # Save report
    output_path = Path('reports/interactive_detection_report.html')
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\n✓ Report generated: {output_path}")
    print(f"\nOpen in browser to view and provide feedback:")
    print(f"  file://{output_path.absolute()}")
    print(f"\nKeyboard shortcuts:")
    print(f"  Ctrl+F: Export feedback to JSON")


if __name__ == '__main__':
    generate_report()
