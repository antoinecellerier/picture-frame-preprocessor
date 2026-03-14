"""Interactive HTML report generation for detection testing and feedback."""

import json
from pathlib import Path
import base64
from PIL import Image, ImageDraw, ImageFont, ImageOps
import io
from datetime import datetime

from .detector import OptimizedEnsembleDetector, ArtFeatureDetector, calculate_iou
from .cropper import SmartCropper
from .clip_detector import CLIPMosaicDetector, SigLIPClassVerifier
from . import defaults
from .defaults import MIN_ART_SCORE

# Module-level singletons — models load lazily on first use.
_clip_detector = CLIPMosaicDetector(threshold=0.022)
_clip_candidate_detector = OptimizedEnsembleDetector(confidence_threshold=0.10)
_siglip_verifier = SigLIPClassVerifier()


def draw_boxes_on_image(image_path, detections, ground_truth_boxes=None,
                        primary=None, crop_targets=None, focal_detections=None,
                        selected_anchor=None, vlm_detections=None, max_width=800):
    """Draw detected and ground truth bounding boxes on image.

    Args:
        crop_targets: list of Detection objects used as crop anchors.
            These get highlighted with distinct colors (orange) so
            the user can see which detections produced crops.
        focal_detections: list of Detection objects from the focal-point
            second pass.  Drawn in magenta so they're visually distinct.
        selected_anchor: the single Detection chosen as the crop anchor
            from the inner/focal detections.  Drawn in gold with thick
            border so it's easy to spot.
    """
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

        # Build lookup sets for special detection categories
        crop_target_bboxes = set()
        if crop_targets:
            crop_target_bboxes = {tuple(d.bbox) for d in crop_targets}

        focal_bboxes = set()
        if focal_detections:
            focal_bboxes = {tuple(d.bbox) for d in focal_detections}

        vlm_bboxes = set()
        if vlm_detections:
            vlm_bboxes = {tuple(d.bbox) for d in vlm_detections}

        selected_anchor_bbox = tuple(selected_anchor.bbox) if selected_anchor else None

        # Draw ground truth boxes in blue
        if ground_truth_boxes:
            for gt_box in ground_truth_boxes:
                bbox = [int(coord * scale) for coord in gt_box]
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255), width=3)
                draw.text((x1, y2 + 5), "Ground Truth", fill=(0, 0, 255), font=small_font)

        # Draw detected boxes
        if detections:
            for det in detections:
                bbox = [int(coord * scale) for coord in det.bbox]
                x1, y1, x2, y2 = bbox

                det_bbox_t = tuple(det.bbox)
                is_primary = primary is not None and det.bbox == primary.bbox
                is_crop_target = det_bbox_t in crop_target_bboxes
                is_selected_anchor = det_bbox_t == selected_anchor_bbox
                is_focal = det_bbox_t in focal_bboxes

                is_vlm = det_bbox_t in vlm_bboxes

                # Color scheme for primary varies by detection source:
                #   yolo → green, dino → chartreuse, vlm → cyan, mixed → yellow
                # selected anchor=gold, crop target=orange, focal=magenta, other=dim green
                if is_primary:
                    src = getattr(det, 'source', None)
                    if src == 'dino':
                        color = (160, 255, 0)   # chartreuse
                    elif src == 'vlm':
                        color = (0, 220, 220)   # cyan
                    elif src is None:
                        color = (255, 255, 0)   # yellow = merged/mixed
                    else:
                        color = (0, 255, 0)     # green = yolo (default)
                    line_w = 4
                elif is_selected_anchor:
                    color = (255, 215, 0)   # gold
                    line_w = 4
                elif is_crop_target:
                    color = (255, 165, 0)   # orange
                    line_w = 3
                elif is_focal:
                    color = (220, 0, 220)   # magenta
                    line_w = 3
                elif is_vlm:
                    color = (0, 220, 220)   # cyan
                    line_w = 2
                else:
                    color = (0, 200, 0)
                    line_w = 2

                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

                label = f"{det.class_name} {det.confidence:.2f}"
                if is_primary:
                    src = getattr(det, 'source', None)
                    src_tag = f"[{src.upper()}]" if src else "[MIXED]"
                    label = f"PRIMARY {src_tag}: " + label
                elif is_selected_anchor:
                    label = "ANCHOR: " + label
                elif is_crop_target:
                    label = "CROP: " + label
                elif is_focal:
                    label = "FOCAL: " + label
                elif is_vlm:
                    label = "VLM: " + label

                text_bbox = draw.textbbox((x1, y1-20), label, font=font)
                draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2],
                             fill=color)
                text_color = (255, 255, 255) if not is_crop_target else (0, 0, 0)
                draw.text((x1, y1-20), label, fill=text_color, font=font)

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=90)
        img_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_data}"
    except Exception as e:
        print(f"Error drawing boxes on {image_path}: {e}")
        return None


def run_detection(image_path, detector, verbose=False, cropper=None, clip_mosaic=False,
                  siglip_verify=False):
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

        # === SIGLIP CLASS VERIFICATION ===
        # Correct mislabelled detections before primary selection.
        if siglip_verify and detections:
            img_w, img_h = img.size
            detections = _siglip_verifier.verify_all(
                detections, img, img_w, img_h, verbose=verbose
            )
        siglip_corrected_count = sum(
            1 for d in detections if d.original_class is not None
        )

        # Get primary by smart selection algorithm (with score)
        primary = None
        art_score = 0.0
        if detections and hasattr(detector, 'get_primary_subject_with_score'):
            primary, art_score = detector.get_primary_subject_with_score(detections)
        elif detections:
            primary = detector.get_primary_subject(detections)

        # === CLIP MOSAIC DETECTION ===
        # Low-conf candidate pass provides extra bbox zones for CLIP without
        # polluting the main detection list.
        clip_dets = []
        if clip_mosaic:
            try:
                candidate_dets = _clip_candidate_detector.detect(
                    img, verbose=False, image_path=str(image_path)
                )
            except TypeError:
                candidate_dets = _clip_candidate_detector.detect(img, verbose=False)
            clip_dets = _clip_detector.detect(
                img, detections,
                candidate_detections=candidate_dets,
                verbose=verbose,
            )
        clip_count = len(clip_dets)
        clip_max_score = max(d.confidence for d in clip_dets) if clip_dets else None
        if clip_dets:
            if verbose:
                print(f"  CLIP: {len(clip_dets)} mosaic detection(s)")
            detections.extend(clip_dets)
            if hasattr(detector, 'get_primary_subject_with_score'):
                primary, art_score = detector.get_primary_subject_with_score(detections)
            else:
                primary = detector.get_primary_subject(detections)
        clip_primary_selected = (
            primary is not None
            and clip_dets
            and any(primary.bbox == d.bbox for d in clip_dets)
        )

        # === FOCAL POINT DETECTION ===
        # When primary fills the frame, run a focused pass on the primary's
        # zone with face/figure prompts to find a better crop anchor.
        # Skip for 3D objects (sculpture, statue, etc.) — the object itself
        # is the focal point and face detection inside it adds noise.
        focal_detections = []
        if (primary is not None
                and cropper is not None
                and hasattr(detector, 'detect_focal_points')
                and not ArtFeatureDetector.is_3d_art(primary.class_name)
                and cropper.primary_fills_frame(primary.bbox, img.size)):
            focal_detections = detector.detect_focal_points(img, primary.bbox, verbose=verbose)
            if focal_detections:
                if verbose:
                    print(f"  Focal pass: {len(focal_detections)} detections")
                # NOTE: focal_detections are kept separate from main detections so
                # they cannot corrupt primary selection in crop_with_detections.

        # Get primary by confidence (old method) for comparison
        primary_by_confidence = detections[0] if detections else None

        # Check if selection algorithm chose a different primary
        selection_changed = False
        if primary and primary_by_confidence:
            selection_changed = primary.bbox != primary_by_confidence.bbox

        vlm_raw = getattr(detector, '_last_vlm_detections', [])
        vlm_count = len(vlm_raw)
        vlm_primary = (primary is not None and vlm_raw
                       and any(primary.bbox == d.bbox for d in vlm_raw))

        return {
            'all_detections': detections,
            'focal_detections': focal_detections,
            'primary': primary,
            'primary_by_confidence': primary_by_confidence,
            'selection_changed': selection_changed,
            'count': len(detections),
            'art_score': art_score,
            'clip_count': clip_count,
            'clip_max_score': clip_max_score,
            'clip_primary_selected': clip_primary_selected,
            'siglip_corrected_count': siglip_corrected_count,
            'vlm_count': vlm_count,
            'vlm_primary': vlm_primary,
            'vlm_detections': vlm_raw,
        }
    except Exception as e:
        print(f"Error detecting in {image_path}: {e}")
        return {'all_detections': [], 'focal_detections': [], 'primary': None,
                'primary_by_confidence': None, 'selection_changed': False,
                'count': 0, 'art_score': 0.0,
                'clip_count': 0, 'clip_max_score': None, 'clip_primary_selected': False,
                'siglip_corrected_count': 0}


def check_accuracy(primary, ground_truth_boxes, iou_threshold=0.3):
    """Check if primary detection matches ground truth."""
    if not primary or not ground_truth_boxes:
        return False, 0.0

    best_iou = 0.0

    for gt_box in ground_truth_boxes:
        iou = calculate_iou(primary.bbox, gt_box)
        best_iou = max(best_iou, iou)

    return best_iou >= iou_threshold, best_iou


def generate_result_image(image_path, detections, cropper, focal_detections=None, max_width=400):
    """Generate the cropped result image for comparison.

    Returns (data_uri, zoom_applied, primary_fills_frame, selected_inner_det).
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = ImageOps.exif_transpose(img)

        # Run the actual cropping logic (focal_dets passed separately)
        cropped = cropper.crop_image(img, detections, strategy='smart',
                                     focal_detections=focal_detections)

        # Get crop info (including selected inner anchor, if any)
        zoom_applied = cropper.last_zoom_applied
        selected_inner_det = getattr(cropper, 'last_inner_detection', None)

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
        primary_fills_frame = getattr(cropper, 'last_primary_fills_frame', False)
        return f"data:image/jpeg;base64,{img_data}", zoom_applied, primary_fills_frame, selected_inner_det
    except Exception as e:
        print(f"Error generating result for {image_path}: {e}")
        return None, 1.0, False, None


def generate_multi_crop_images(image_path, detections, cropper, focal_detections=None, max_width=250):
    """Generate cropped images for all viable art subjects (multi-crop display).

    Returns (crops, crop_target_detections) where crops is a list of
    (data_uri, zoom_applied, class_name) tuples (empty if < 2 subjects),
    and crop_target_detections is the list of Detection objects used as
    crop anchors (for highlighting in the detection image).
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = ImageOps.exif_transpose(img)

        multi_results = cropper.crop_all_subjects(img, detections, focal_detections=focal_detections)
        if len(multi_results) < 2:
            return [], []

        output = []
        crop_targets = []
        for cropped, det, zoom_applied in multi_results:
            crop_targets.append(det)

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

        return output, crop_targets
    except Exception as e:
        print(f"Error generating multi-crop for {image_path}: {e}")
        return [], []


def generate_report(input_dir=None, ground_truth_path=None, output_file=None,
                    detector=None, cropper=None, clip_mosaic=False,
                    siglip_verify=False, verbose=False):
    """Generate interactive HTML report.

    Args:
        input_dir: Path to input images directory (default: test_real_images/input/)
        ground_truth_path: Path to ground truth JSON (default: test_real_images/ground_truth_annotations.json)
        output_file: Path to output HTML report (default: reports/interactive_detection_report.html)
        detector: Pre-configured detector instance (default: OptimizedEnsembleDetector)
        cropper: Pre-configured cropper instance (default: SmartCropper with defaults)
        verbose: Whether to print verbose output
    """
    if ground_truth_path is None:
        ground_truth_path = 'test_real_images/ground_truth_annotations.json'
    if input_dir is None:
        input_dir = 'test_real_images/input'
    if output_file is None:
        output_file = 'reports/interactive_detection_report.html'

    print("Loading ground truth annotations...")
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    input_dir = Path(input_dir)
    results = []

    # Create detector once (reused for all images, with caching)
    if detector is None:
        detector = OptimizedEnsembleDetector(
            confidence_threshold=defaults.CONFIDENCE_THRESHOLD,
            merge_threshold=defaults.MERGE_THRESHOLD,
            two_pass=defaults.TWO_PASS
        )

    # Create cropper for generating result images
    if cropper is None:
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
        'focal_prompts': getattr(detector, '_focal_prompts', []),
        'use_vlm': getattr(detector, 'use_vlm', False),
        'vlm_confirm': getattr(detector, 'vlm_confirm', False),
        'vlm_model': getattr(detector, 'vlm_model', None),
        'vlm_max_image_size': getattr(detector, 'vlm_max_image_size', None),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
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
        detection_result = run_detection(image_path, detector, verbose=True, cropper=cropper,
                                         clip_mosaic=clip_mosaic, siglip_verify=siglip_verify)

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
            # Has ground truth but no primary detection -- counts as incorrect
            total_with_gt += 1

        # Generate result/cropped image (skip when art score is below threshold,
        # matching the batch pipeline's filter_non_art behavior)
        # Run multi-crop first so we know which detections are crop targets
        result_image, zoom_applied = None, 1.0
        multi_crop_images = []
        crop_targets = []
        selected_inner_det = None
        auto_filtered = detection_result['art_score'] < MIN_ART_SCORE
        primary_fills_frame = False
        focal_dets = detection_result.get('focal_detections') or None
        if not auto_filtered:
            result_image, zoom_applied, primary_fills_frame, selected_inner_det = generate_result_image(
                image_path,
                detection_result['all_detections'],
                cropper,
                focal_detections=focal_dets,
            )
            multi_crop_images, crop_targets = generate_multi_crop_images(
                image_path,
                detection_result['all_detections'],
                cropper,
                focal_detections=focal_dets,
            )

        # all_detections for drawing = main dets + focal dets (for display only)
        display_detections = detection_result['all_detections'] + (focal_dets or [])

        # Generate visualization (after multi-crop so we can highlight targets)
        img_with_boxes = draw_boxes_on_image(
            image_path,
            display_detections,
            gt_boxes,
            primary=detection_result['primary'],
            crop_targets=crop_targets if multi_crop_images else None,
            focal_detections=focal_dets,
            selected_anchor=selected_inner_det,
            vlm_detections=detection_result.get('vlm_detections') or None,
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
            'auto_filtered': auto_filtered,
            'primary_fills_frame': primary_fills_frame,
            'clip_count': detection_result.get('clip_count', 0),
            'clip_max_score': detection_result.get('clip_max_score'),
            'clip_primary_selected': detection_result.get('clip_primary_selected', False),
            'siglip_corrected_count': detection_result.get('siglip_corrected_count', 0),
            'vlm_count': detection_result.get('vlm_count', 0),
            'vlm_primary': detection_result.get('vlm_primary', False),
            'vlm_detections': detection_result.get('vlm_detections', []),
        })

    accuracy = (correct_count / total_with_gt * 100) if total_with_gt > 0 else 0
    selection_changed_count = sum(1 for r in results if r['selection_changed'])
    not_art_count = sum(1 for r in results if r.get('is_not_art'))
    auto_filtered_count = sum(1 for r in results if r.get('auto_filtered'))

    print(f"\nProcessing complete!")
    print(f"  Accuracy: {correct_count}/{total_with_gt} ({accuracy:.1f}%) (excludes {not_art_count} not-art images)")
    print(f"  Primary selection changed: {selection_changed_count}/{len(results)} images")
    print(f"\nGenerating HTML report...")

    # Build JS data array (one entry per result, used for rendering + export)
    results_js_data = []
    for result in results:
        is_not_art = result.get('is_not_art', False)
        if is_not_art:
            status = 'not-art'
        elif not result['has_ground_truth']:
            status = 'no-gt'
        elif result['is_correct']:
            status = 'correct'
        else:
            status = 'incorrect'

        primary_text = 'No detections'
        primary_changed_text = None
        primary_siglip_text = None
        if result['primary']:
            src = getattr(result['primary'], 'source', None)
            src_tag = f" [{src.upper()}]" if src else " [MIXED]"
            primary_text = f"{result['primary'].class_name} (conf: {result['primary'].confidence:.3f}){src_tag}"
            if result['selection_changed'] and result['primary_by_confidence']:
                primary_changed_text = (
                    f"Changed from: {result['primary_by_confidence'].class_name} "
                    f"({result['primary_by_confidence'].confidence:.3f})"
                )
            if result['primary'].original_class:
                primary_siglip_text = f"SigLIP: was {result['primary'].original_class}"

        crops = []
        for mc_uri, mc_zoom, mc_class in result.get('multi_crop_images', []):
            crops.append({'uri': mc_uri, 'zoom': round(float(mc_zoom), 2), 'cls': mc_class})

        det_export = []
        for det in (result['detections'] or []):
            det_export.append({
                'class_name': det.class_name,
                'confidence': round(float(det.confidence), 4),
                'bbox': [int(c) for c in det.bbox],
            })
        primary_export = None
        if result['primary']:
            primary_export = {
                'class_name': result['primary'].class_name,
                'confidence': round(float(result['primary'].confidence), 4),
                'bbox': [int(c) for c in result['primary'].bbox],
            }

        art_score = result.get('art_score', 0.0)
        results_js_data.append({
            'filename': result['filename'],
            'status': status,
            'imgDetection': result['image_with_boxes'] or '',
            'imgResult': result['result_image'] or '',
            'imgCrops': crops,
            'primaryText': primary_text,
            'primaryChangedText': primary_changed_text,
            'primarySiglipText': primary_siglip_text,
            'artScore': round(float(art_score), 4),
            'artScoreLow': art_score < MIN_ART_SCORE,
            'iou': round(float(result['best_iou']), 4),
            'hasGT': result['has_ground_truth'],
            'detCount': result['detection_count'],
            'zoom': round(float(result['zoom_applied']), 3),
            'primaryFills': result.get('primary_fills_frame', False),
            'clipCount': result.get('clip_count', 0),
            'clipMax': round(float(result['clip_max_score']), 4) if result.get('clip_max_score') is not None else None,
            'clipPrimary': result.get('clip_primary_selected', False),
            'siglipCount': result.get('siglip_corrected_count', 0),
            'vlmCount': result.get('vlm_count', 0),
            'vlmPrimary': result.get('vlm_primary', False),
            'isNotArt': is_not_art,
            'autoFiltered': result.get('auto_filtered', False),
            # export fields
            'detections': det_export,
            'primary': primary_export,
            'isCorrect': result['is_correct'],
            'bestIou': round(float(result['best_iou']), 4),
            'gtBoxes': result['ground_truth_boxes'],
        })

    n_correct = sum(1 for r in results if r['is_correct'])
    n_incorrect = sum(1 for r in results if r['has_ground_truth'] and not r['is_correct'] and not r.get('is_not_art'))
    n_not_art = sum(1 for r in results if r.get('is_not_art'))
    n_no_gt = sum(1 for r in results if not r['has_ground_truth'] and not r.get('is_not_art'))
    n_big_primary = sum(1 for r in results if r.get('primary_fills_frame'))
    n_clip = sum(1 for r in results if r.get('clip_count', 0) > 0)
    n_vlm = sum(1 for r in results if r.get('vlm_count', 0) > 0)

    results_json = json.dumps(results_js_data)
    config_json = json.dumps(config)
    min_art_score_val = MIN_ART_SCORE

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Detection Report</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: #1a1a1a; color: #e0e0e0;
  height: 100vh; display: flex; flex-direction: column; overflow: hidden;
}}

/* ── Header ── */
#header {{
  background: #2a2a2a; border-bottom: 2px solid #4a90d9;
  padding: 6px 16px 0; display: flex; flex-direction: column;
  gap: 0; flex-shrink: 0;
}}
#header-top {{
  display: flex; align-items: center; gap: 8px;
  padding-bottom: 6px; flex-wrap: nowrap; overflow: hidden;
}}
#header-filters {{
  display: flex; align-items: center; gap: 6px;
  padding-bottom: 6px; flex-wrap: nowrap; overflow-x: auto;
}}
#header-filters::-webkit-scrollbar {{ height: 3px; }}
#header-filters::-webkit-scrollbar-thumb {{ background: #444; border-radius: 2px; }}
#header h1 {{ color: #4a90d9; font-size: 1.1rem; flex-shrink: 0; }}
#progress-text {{ color: #FFD700; font-size: 0.9rem; font-weight: bold; white-space: nowrap; flex-shrink: 0; }}
#stats-line {{ font-size: 0.8rem; color: #aaa; white-space: nowrap; flex-shrink: 0; }}
#params-line {{ font-size: 0.75rem; color: #666; white-space: nowrap; flex-grow: 1; overflow: hidden; text-overflow: ellipsis; }}
#config-panel {{
  background: #232323; border-bottom: 1px solid #3a3a3a;
  padding: 12px 16px; display: none; flex-shrink: 0;
  overflow-y: auto; max-height: 40vh;
}}
#config-panel.visible {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.cfg-section {{
  min-width: 200px; flex: 1;
}}
.cfg-section h3 {{
  color: #4a90d9; font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.5px; margin-bottom: 6px; border-bottom: 1px solid #333; padding-bottom: 3px;
}}
.cfg-row {{
  display: flex; justify-content: space-between; gap: 8px;
  padding: 2px 0; font-size: 0.72rem; border-bottom: 1px solid #2a2a2a;
}}
.cfg-row:last-child {{ border-bottom: none; }}
.cfg-key {{ color: #888; }}
.cfg-val {{ color: #ddd; font-family: monospace; text-align: right; word-break: break-all; }}
.cfg-prompts {{ font-size: 0.68rem; color: #999; line-height: 1.5; font-family: monospace; }}
.btn {{
  padding: 5px 11px; border: none; border-radius: 4px; cursor: pointer;
  font-size: 0.8rem; font-weight: bold; background: #3a3a3a; color: #e0e0e0;
}}
.btn:hover {{ background: #555; }}
.filter-btn {{
  background: #333; color: #aaa; border: 1px solid #444;
  padding: 4px 10px; border-radius: 12px; cursor: pointer;
  font-size: 0.75rem; font-weight: bold;
}}
.filter-btn:hover {{ background: #444; color: #e0e0e0; }}
.filter-btn.active {{ background: #4a90d9; color: white; border-color: #4a90d9; }}

/* ── 3-panel layout ── */
#main {{ display: flex; flex: 1; overflow: hidden; }}

/* ── Sidebar ── */
#sidebar {{
  width: 175px; flex-shrink: 0; background: #252525;
  border-right: 1px solid #333; overflow-y: auto; padding: 4px 0;
}}
.sidebar-item {{
  padding: 3px 8px; cursor: pointer; font-size: 0.7rem;
  border-left: 3px solid transparent;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.sidebar-item:hover {{ background: #333; }}
.sidebar-item.active {{ background: #333; }}
.sidebar-item.correct  {{ border-left-color: #10b981; }}
.sidebar-item.incorrect {{ border-left-color: #ef4444; }}
.sidebar-item.no-gt    {{ border-left-color: #6b7280; }}
.sidebar-item.not-art  {{ border-left-color: #f59e0b; }}
.sidebar-item.hidden   {{ display: none; }}

/* ── Content area ── */
#content-area {{
  flex: 1; overflow: hidden; background: #111;
  padding: 10px; display: flex; flex-direction: column; gap: 6px; min-width: 0;
}}
#images-row {{
  flex: 1; min-height: 0;
  display: flex; gap: 8px; flex-wrap: nowrap; align-items: stretch; overflow: hidden;
}}
.img-panel {{
  flex: 2; min-width: 0; position: relative;
  display: flex; align-items: center; justify-content: center; overflow: hidden;
}}
.img-panel.result {{ flex: 1; max-width: 320px; }}
.img-panel img {{
  max-width: 100%; max-height: 100%; width: auto; height: auto;
  object-fit: contain; display: block; border-radius: 4px;
}}
.img-label {{
  position: absolute; top: 6px; left: 6px;
  background: rgba(0,0,0,0.75); color: white;
  padding: 2px 8px; border-radius: 3px;
  font-size: 0.68rem; font-weight: bold; text-transform: uppercase;
}}
#legend {{
  display: flex; gap: 16px; flex-wrap: wrap;
  font-size: 0.72rem; color: #888; padding: 2px 0;
}}
.ldot {{ display: inline-block; width: 22px; height: 3px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }}

/* ── Right panel ── */
#right-panel {{
  width: 225px; flex-shrink: 0; background: #252525;
  border-left: 1px solid #333; display: flex; flex-direction: column; overflow: hidden;
}}
#filename-label {{
  padding: 8px 10px 5px; font-size: 0.68rem; color: #777;
  word-break: break-all; border-bottom: 1px solid #333; flex-shrink: 0;
}}
#badges-section {{
  padding: 6px 10px; display: flex; flex-wrap: wrap; gap: 4px;
  border-bottom: 1px solid #333; flex-shrink: 0;
}}
.badge {{
  font-size: 0.7rem; font-weight: bold; padding: 2px 7px; border-radius: 10px;
}}
.badge.correct    {{ background: #064e3b; color: #6ee7b7; }}
.badge.incorrect  {{ background: #7f1d1d; color: #fca5a5; }}
.badge.no-gt      {{ background: #374151; color: #d1d5db; }}
.badge.not-art    {{ background: #78350f; color: #fde68a; }}
.badge.filtered   {{ background: #78350f; color: #fde68a; }}
.badge.selection  {{ background: #1e3a5f; color: #93c5fd; }}
.badge.multicrop  {{ background: #1e3a8a; color: #93c5fd; }}
.badge.clip       {{ background: #4c1d95; color: #c4b5fd; }}
.badge.clip-primary {{ background: #6d28d9; color: white; }}
.badge.siglip     {{ background: #7c2d12; color: #fdba74; }}
.badge.vlm        {{ background: #164e63; color: #67e8f9; }}
.badge.vlm-primary {{ background: #0e7490; color: white; }}
#info-section {{
  padding: 8px 10px; flex-shrink: 0; border-bottom: 1px solid #333; font-size: 0.75rem;
}}
.info-label {{ color: #777; font-size: 0.67rem; text-transform: uppercase; }}
.info-val {{ color: #e0e0e0; }}
.info-val.changed {{ color: #6ee7b7; font-size: 0.68rem; }}
.info-val.siglip-note {{ color: #fbbf24; font-size: 0.68rem; }}
.info-val.low-score {{ color: #ef4444; }}
#feedback-section {{ flex: 1; overflow-y: auto; padding: 8px 10px; }}
#feedback-section h3 {{
  color: #888; font-size: 0.72rem; text-transform: uppercase; margin-bottom: 6px;
}}
.fb-btn {{
  width: 100%; padding: 6px 8px; border: 2px solid transparent;
  border-radius: 5px; cursor: pointer; font-size: 0.78rem; font-weight: bold;
  text-align: left; background: #333; color: #e0e0e0; margin-bottom: 3px;
}}
.fb-btn:hover {{ background: #444; }}
.fb-btn.selected {{ border-color: #4a90d9; background: #1e3a5f; color: #93c5fd; }}
.key-badge {{
  background: rgba(0,0,0,0.4); border-radius: 3px; padding: 0 4px;
  font-size: 0.7rem; color: #ccc; font-family: monospace; margin-right: 4px;
}}
#comment-box {{
  width: 100%; background: #333; border: 1px solid #555; color: #ddd;
  border-radius: 4px; padding: 5px; font-size: 0.75rem;
  resize: vertical; min-height: 55px; margin-top: 6px;
}}
</style>
</head>
<body>
<div id="header">
  <div id="header-top">
    <h1>Detection Report</h1>
    <span id="progress-text">—</span>
    <button class="btn" style="font-size:1rem;padding:3px 10px" onclick="navigate(-1)" title="Previous (←)">&#8592;</button>
    <button class="btn" style="font-size:1rem;padding:3px 10px" onclick="navigate(1)" title="Next (→ or Space)">&#8594; <span style="font-size:0.65rem;opacity:0.6">Space</span></button>
    <span id="stats-line">{accuracy:.1f}% ({correct_count}/{total_with_gt}) &nbsp;|&nbsp; {auto_filtered_count} filtered</span>
    <span id="params-line">{config['models']['yolo_world']} + dino-tiny &nbsp;· conf {config['confidence_threshold']} · {config['target_width']}×{config['target_height']} · {config['generated_at']}</span>
    <button class="btn" onclick="toggleConfig()" id="config-btn" title="Toggle config panel">Config ▾</button>
  </div>
  <div id="header-filters">
    <button class="filter-btn active" onclick="setFilter(this,'all')">All ({len(results)})</button>
    <button class="filter-btn" onclick="setFilter(this,'correct')">Correct ({n_correct})</button>
    <button class="filter-btn" onclick="setFilter(this,'incorrect')">Incorrect ({n_incorrect})</button>
    <button class="filter-btn" onclick="setFilter(this,'not-art')">Not Art ({n_not_art})</button>
    <button class="filter-btn" onclick="setFilter(this,'no-gt')">No GT ({n_no_gt})</button>
    <button class="filter-btn" onclick="setFilter(this,'big-primary')">Big Primary ({n_big_primary})</button>
    <button class="filter-btn" onclick="setFilter(this,'clip')">CLIP ({n_clip})</button>
    <button class="filter-btn" onclick="setFilter(this,'vlm')">VLM ({n_vlm})</button>
    <button class="btn" onclick="exportFeedback()" style="background:#2d6a4f;margin-left:8px">Export Feedback (Ctrl+E)</button>
  </div>
</div>
<div id="config-panel"></div>
<div id="main">
  <div id="sidebar"></div>
  <div id="content-area">
    <div id="images-row"></div>
    <div id="legend">
      <span><span class="ldot" style="background:#00ff00"></span>Primary [YOLO]</span>
      <span><span class="ldot" style="background:#a0ff00"></span>Primary [DINO]</span>
      <span><span class="ldot" style="background:#ffff00"></span>Primary [MIXED]</span>
      <span><span class="ldot" style="background:#00dcdc"></span>Primary/Det [VLM]</span>
      <span><span class="ldot" style="background:#0000ff"></span>Ground Truth</span>
      <span><span class="ldot" style="background:#ffd700"></span>Selected Anchor</span>
      <span><span class="ldot" style="background:#ffa500"></span>Crop Target</span>
      <span><span class="ldot" style="background:#dc00dc"></span>Focal Detection</span>
      <span><span class="ldot" style="background:#00c800"></span>Other Detection</span>
    </div>
  </div>
  <div id="right-panel">
    <div id="filename-label">—</div>
    <div id="badges-section"></div>
    <div id="info-section"></div>
    <div id="feedback-section">
      <h3>Feedback</h3>
      <button class="fb-btn" data-rating="good"><span class="key-badge">G</span>Good</button>
      <button class="fb-btn" data-rating="bad_detection"><span class="key-badge">D</span>Bad Detection</button>
      <button class="fb-btn" data-rating="bad_crop"><span class="key-badge">C</span>Bad Crop</button>
      <button class="fb-btn" data-rating="bad_both"><span class="key-badge">B</span>Both Bad</button>
      <button class="fb-btn" data-rating="other"><span class="key-badge">O</span>Other</button>
      <textarea id="comment-box" placeholder="N to focus · Shift+Enter → next" oninput="updateComment(this.value)"></textarea>
    </div>
  </div>
</div>
<script>
const RESULTS = {results_json};

function buildConfigPanel() {{
  const c = CONFIG;
  const rows = (pairs) => pairs.map(([k,v]) =>
    `<div class="cfg-row"><span class="cfg-key">${{k}}</span><span class="cfg-val">${{v}}</span></div>`
  ).join('');
  document.getElementById('config-panel').innerHTML = `
    <div class="cfg-section">
      <h3>Detection</h3>
      ${{rows([
        ['Ensemble', c.detector],
        ['YOLO-World', c.models.yolo_world],
        ['Grounding DINO', c.models.grounding_dino.split('/').pop()],
        ['Confidence', c.confidence_threshold],
        ['Merge IoU', c.merge_threshold],
        ['Two-pass', c.two_pass ? 'enabled' : 'disabled'],
        ['Primary selection', c.primary_selection],
        ...(c.use_vlm ? [['VLM', c.vlm_confirm ? 'confirm (every image)' : 'fallback (no candidate)']] : []),
        ...(c.use_vlm ? [['VLM model', c.vlm_model ? c.vlm_model.split('/').pop() : '—']] : []),
        ...(c.use_vlm ? [['VLM max px', c.vlm_max_image_size]] : []),
      ])}}
    </div>
    <div class="cfg-section">
      <h3>Cropping</h3>
      ${{rows([
        ['Target size', c.target_width + '×' + c.target_height],
        ['Max zoom', c.zoom_factor + 'x'],
        ['Saliency fallback', c.use_saliency_fallback ? 'enabled' : 'disabled'],
      ])}}
      <h3 style="margin-top:10px">Focal Detection</h3>
      ${{rows([['Prompts', c.focal_prompts.join(', ')]])}}
    </div>
    <div class="cfg-section">
      <h3>YOLO-World Prompts <span style="color:#666;font-weight:normal">(${{c.yolo_world_prompts.length}})</span></h3>
      <div class="cfg-prompts">${{c.yolo_world_prompts.join(' · ')}}</div>
    </div>
    <div class="cfg-section">
      <h3>Grounding DINO Prompts <span style="color:#666;font-weight:normal">(${{c.grounding_dino_prompts.length}})</span></h3>
      <div class="cfg-prompts">${{c.grounding_dino_prompts.join(' · ')}}</div>
    </div>`;
}}

function toggleConfig() {{
  const panel = document.getElementById('config-panel');
  const btn = document.getElementById('config-btn');
  const visible = panel.classList.toggle('visible');
  btn.textContent = visible ? 'Config ▴' : 'Config ▾';
}}
const CONFIG = {config_json};
const MIN_ART_SCORE = {min_art_score_val};
const feedbackData = {{}};

let currentIdx = 0;
let currentFilter = 'all';
let visibleIndices = [];

// Pre-mark correct detections as "good"
RESULTS.forEach((r, i) => {{
  if (r.status === 'correct') feedbackData[i] = {{rating: 'good'}};
}});

function buildSidebar() {{
  const sb = document.getElementById('sidebar');
  RESULTS.forEach((r, i) => {{
    const el = document.createElement('div');
    el.className = 'sidebar-item ' + r.status;
    el.dataset.idx = i;
    el.textContent = (i + 1) + '. ' + r.filename.replace(/[.][^.]+$/, '');
    el.title = r.filename;
    el.onclick = () => setIndex(i);
    sb.appendChild(el);
  }});
  updateFilter();
}}

function updateFilter() {{
  visibleIndices = [];
  document.querySelectorAll('.sidebar-item').forEach(el => {{
    const i = parseInt(el.dataset.idx);
    const r = RESULTS[i];
    let show = false;
    switch (currentFilter) {{
      case 'all':         show = true; break;
      case 'correct':     show = r.status === 'correct'; break;
      case 'incorrect':   show = r.status === 'incorrect'; break;
      case 'not-art':     show = r.status === 'not-art'; break;
      case 'no-gt':       show = r.status === 'no-gt'; break;
      case 'big-primary': show = r.primaryFills; break;
      case 'clip':        show = r.clipCount > 0; break;
      case 'vlm':         show = r.vlmCount > 0; break;
    }}
    el.classList.toggle('hidden', !show);
    if (show) visibleIndices.push(i);
  }});
}}

function setFilter(btn, filter) {{
  currentFilter = filter;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  updateFilter();
  if (!visibleIndices.includes(currentIdx) && visibleIndices.length > 0) {{
    setIndex(visibleIndices[0]);
  }} else {{
    updateProgress();
    highlightSidebar();
  }}
}}

function navigate(delta) {{
  const pos = visibleIndices.indexOf(currentIdx);
  const npos = pos + delta;
  if (npos >= 0 && npos < visibleIndices.length) setIndex(visibleIndices[npos]);
}}

function setIndex(idx) {{
  currentIdx = idx;
  location.hash = encodeURIComponent(RESULTS[idx].filename);
  renderContent();
  highlightSidebar();
  updateProgress();
}}

function updateProgress() {{
  const pos = visibleIndices.indexOf(currentIdx);
  const total = visibleIndices.length;
  const suffix = total < RESULTS.length ? ` (of ${{RESULTS.length}})` : '';
  document.getElementById('progress-text').textContent = `${{pos + 1}}/${{total}}${{suffix}}`;
}}

function highlightSidebar() {{
  document.querySelectorAll('.sidebar-item').forEach(el => {{
    el.classList.toggle('active', parseInt(el.dataset.idx) === currentIdx);
  }});
  const active = document.querySelector('.sidebar-item.active');
  if (active) active.scrollIntoView({{block: 'nearest'}});
}}

function statusLabel(s) {{
  return {{correct:'Correct', incorrect:'Incorrect', 'no-gt':'No GT', 'not-art':'Not Art'}}[s] || s;
}}

function renderContent() {{
  const r = RESULTS[currentIdx];

  document.getElementById('filename-label').textContent = r.filename;

  // Badges
  let bh = `<span class="badge ${{r.status}}">${{statusLabel(r.status)}}</span>`;
  if (r.autoFiltered)       bh += `<span class="badge filtered">FILTERED</span>`;
  else if (r.isNotArt)      bh += `<span class="badge not-art">NOT ART (GT)</span>`;
  if (r.primaryChangedText) bh += `<span class="badge selection">Sel. Changed</span>`;
  if (r.imgCrops.length)    bh += `<span class="badge multicrop">Multi-crop: ${{r.imgCrops.length}}</span>`;
  if (r.clipCount > 0) {{
    const cs = (r.clipMax || 0).toFixed(3);
    bh += r.clipPrimary
      ? `<span class="badge clip-primary">CLIP primary (${{cs}})</span>`
      : `<span class="badge clip">CLIP: ${{r.clipCount}} (${{cs}})</span>`;
  }}
  if (r.siglipCount > 0) bh += `<span class="badge siglip">SigLIP: ${{r.siglipCount}}</span>`;
  if (r.vlmCount > 0) bh += r.vlmPrimary
    ? `<span class="badge vlm-primary">VLM primary (${{r.vlmCount}} det)</span>`
    : `<span class="badge vlm">VLM: ${{r.vlmCount}} det</span>`;
  document.getElementById('badges-section').innerHTML = bh;

  // Info
  let ih = `<div style="margin-bottom:6px">
    <div class="info-label">Primary Detection</div>
    <div class="info-val">${{r.primaryText}}</div>`;
  if (r.primaryChangedText) ih += `<div class="info-val changed">${{r.primaryChangedText}}</div>`;
  if (r.primarySiglipText)  ih += `<div class="info-val siglip-note">${{r.primarySiglipText}}</div>`;
  ih += `</div><div style="display:flex;gap:10px;flex-wrap:wrap">`;
  ih += `<div><div class="info-label">Art score</div><div class="info-val${{r.artScoreLow ? ' low-score' : ''}}">${{r.artScore.toFixed(3)}}</div></div>`;
  if (r.hasGT) ih += `<div><div class="info-label">IoU</div><div class="info-val">${{r.iou.toFixed(3)}}</div></div>`;
  ih += `<div><div class="info-label">Dets</div><div class="info-val">${{r.detCount}}</div></div>`;
  if (r.vlmCount > 0) ih += `<div><div class="info-label">VLM dets</div><div class="info-val" style="color:#67e8f9">${{r.vlmCount}}${{r.vlmPrimary ? ' ✓ primary' : ''}}</div></div>`;
  ih += `<div><div class="info-label">Zoom</div><div class="info-val">${{r.zoom.toFixed(2)}}x</div></div>`;
  ih += `</div>`;
  document.getElementById('info-section').innerHTML = ih;

  // Images
  let imgh = '';
  if (r.imgDetection) {{
    imgh += `<div class="img-panel"><span class="img-label">Detection</span><img src="${{r.imgDetection}}" alt="detection"></div>`;
  }}
  if (r.imgCrops.length > 0) {{
    r.imgCrops.forEach((c, ci) => {{
      imgh += `<div class="img-panel result"><span class="img-label">Crop ${{ci + 1}}: ${{c.cls}} (${{c.zoom.toFixed(1)}}x)</span><img src="${{c.uri}}" alt="crop ${{ci + 1}}"></div>`;
    }});
  }} else if (r.imgResult) {{
    imgh += `<div class="img-panel result"><span class="img-label">Result (${{r.zoom.toFixed(2)}}x)</span><img src="${{r.imgResult}}" alt="result"></div>`;
  }}
  document.getElementById('images-row').innerHTML = imgh;

  // Feedback state
  const rating = (feedbackData[currentIdx] || {{}}).rating;
  document.querySelectorAll('.fb-btn').forEach(btn => {{
    btn.classList.toggle('selected', btn.dataset.rating === rating);
  }});
  document.getElementById('comment-box').value = (feedbackData[currentIdx] || {{}}).comment || '';
}}

// Feedback buttons
document.querySelectorAll('.fb-btn').forEach(btn => {{
  btn.onclick = () => setFeedbackRating(btn.dataset.rating);
}});

function updateComment(val) {{
  if (!feedbackData[currentIdx]) feedbackData[currentIdx] = {{}};
  feedbackData[currentIdx].comment = val;
}}

function exportFeedback() {{
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const outFilename = `detection_feedback_${{timestamp}}.json`;
  const feedbackWithContext = {{}};
  for (const [idxStr, fb] of Object.entries(feedbackData)) {{
    const i = parseInt(idxStr);
    const r = RESULTS[i];
    feedbackWithContext[r.filename] = {{
      ...fb,
      detections: r.detections,
      primary: r.primary,
      is_correct: r.isCorrect,
      best_iou: r.bestIou,
      ground_truth_boxes: r.gtBoxes,
    }};
  }}
  const exportData = {{
    generated_at: new Date().toISOString(),
    config: CONFIG,
    total_images: RESULTS.length,
    feedback_count: Object.keys(feedbackData).length,
    feedback: feedbackWithContext,
  }};
  const blob = new Blob([JSON.stringify(exportData, null, 2)], {{type: 'application/json'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = outFilename; a.click();
  URL.revokeObjectURL(url);
  alert(`Exported feedback for ${{Object.keys(feedbackData).length}} images`);
}}

const FB_KEYS = {{g:'good', d:'bad_detection', c:'bad_crop', b:'bad_both', o:'other'}};
const commentBox = document.getElementById('comment-box');

function setFeedbackRating(rating) {{
  if (!feedbackData[currentIdx]) feedbackData[currentIdx] = {{}};
  feedbackData[currentIdx].rating = rating;
  document.querySelectorAll('.fb-btn').forEach(btn => {{
    btn.classList.toggle('selected', btn.dataset.rating === rating);
  }});
}}

document.addEventListener('keydown', e => {{
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') {{
    if ((e.shiftKey || e.ctrlKey) && e.key === 'Enter') {{
      e.preventDefault();
      e.target.blur();
      navigate(1);
    }}
    return;
  }}
  if (e.key === 'ArrowLeft') navigate(-1);
  else if (e.key === 'ArrowRight' || e.key === ' ') {{ e.preventDefault(); navigate(1); }}
  else if (e.key === 'e' && (e.ctrlKey || e.metaKey)) {{ e.preventDefault(); exportFeedback(); }}
  else if (e.key.toLowerCase() === 'n') {{ e.preventDefault(); commentBox.focus(); commentBox.select(); }}
  else {{
    const rating = FB_KEYS[e.key.toLowerCase()];
    if (rating) setFeedbackRating(rating);
  }}
}});

buildConfigPanel();
buildSidebar();
if (RESULTS.length > 0) {{
  const savedName = decodeURIComponent(location.hash.slice(1));
  const savedIdx = RESULTS.findIndex(r => r.filename === savedName);
  setIndex(savedIdx >= 0 ? savedIdx : 0);
}}
</script>
</body>
</html>
"""
    # Save report
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\nReport generated: {output_path}")
    print(f"\nOpen in browser to view and provide feedback:")
    print(f"  file://{output_path.absolute()}")
