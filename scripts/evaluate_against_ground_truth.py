#!/usr/bin/env python3
"""
Evaluate model detections against ground truth annotations.

Usage:
    python scripts/evaluate_against_ground_truth.py \\
        --ground-truth ground_truth_annotations.json \\
        --model yolov8m \\
        --confidence 0.15
"""

import json
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.frame_prep.detector import ArtFeatureDetector


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def evaluate_model(ground_truth_path, model_name='yolov8m', confidence_threshold=0.15, iou_threshold=0.5):
    """
    Evaluate a model against ground truth annotations.

    Args:
        ground_truth_path: Path to ground truth JSON file
        model_name: YOLO model to evaluate
        confidence_threshold: Detection confidence threshold
        iou_threshold: IoU threshold for considering a detection correct

    Returns:
        dict: Evaluation metrics
    """
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    detector = ArtFeatureDetector(
        model_name=model_name,
        confidence_threshold=confidence_threshold
    )

    metrics = {
        'total_images': len(ground_truth),
        'images_with_ground_truth': 0,
        'images_detected_correctly': 0,
        'images_missed': 0,
        'total_ground_truth_boxes': 0,
        'total_detected_boxes': 0,
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'per_image_results': []
    }

    input_dir = Path('test_real_images/input')

    for gt_entry in ground_truth:
        filename = gt_entry['filename']
        img_path = input_dir / filename

        if not img_path.exists():
            print(f"Warning: Image not found: {filename}")
            continue

        # Load image
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)

        # Get ground truth boxes (combine manual + correct detections)
        gt_boxes = []

        # Manual ground truth boxes
        for box_data in gt_entry.get('manual_boxes', []):
            gt_boxes.append(box_data['bbox'])

        # Marked correct detections from models (these are also ground truth)
        for det_data in gt_entry.get('correct_detections', []):
            gt_boxes.append(det_data['bbox'])

        if len(gt_boxes) == 0:
            # No ground truth for this image, skip
            continue

        metrics['images_with_ground_truth'] += 1
        metrics['total_ground_truth_boxes'] += len(gt_boxes)

        # Run detection
        detections = detector.detect(img, verbose=False)
        detected_boxes = [det.bbox for det in detections]

        metrics['total_detected_boxes'] += len(detected_boxes)

        # Match detections to ground truth using IoU
        matched_gt = set()
        matched_det = set()

        for det_idx, det_box in enumerate(detected_boxes):
            best_iou = 0
            best_gt_idx = None

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue

                iou = calculate_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx is not None:
                matched_gt.add(best_gt_idx)
                matched_det.add(det_idx)
                metrics['true_positives'] += 1

        # False positives: detections that didn't match any ground truth
        metrics['false_positives'] += len(detected_boxes) - len(matched_det)

        # False negatives: ground truth boxes that weren't detected
        metrics['false_negatives'] += len(gt_boxes) - len(matched_gt)

        # Image-level metrics
        image_correct = len(matched_gt) > 0  # At least one GT box was detected
        if image_correct:
            metrics['images_detected_correctly'] += 1
        else:
            metrics['images_missed'] += 1

        metrics['per_image_results'].append({
            'filename': filename,
            'gt_boxes': len(gt_boxes),
            'detected_boxes': len(detected_boxes),
            'matched': len(matched_gt),
            'correct': image_correct
        })

    # Calculate aggregate metrics
    if metrics['images_with_ground_truth'] > 0:
        metrics['image_accuracy'] = metrics['images_detected_correctly'] / metrics['images_with_ground_truth']
    else:
        metrics['image_accuracy'] = 0.0

    if metrics['total_ground_truth_boxes'] > 0:
        metrics['box_recall'] = metrics['true_positives'] / metrics['total_ground_truth_boxes']
    else:
        metrics['box_recall'] = 0.0

    if metrics['total_detected_boxes'] > 0:
        metrics['box_precision'] = metrics['true_positives'] / metrics['total_detected_boxes']
    else:
        metrics['box_precision'] = 0.0

    if metrics['box_precision'] + metrics['box_recall'] > 0:
        metrics['box_f1'] = 2 * (metrics['box_precision'] * metrics['box_recall']) / (metrics['box_precision'] + metrics['box_recall'])
    else:
        metrics['box_f1'] = 0.0

    return metrics


def print_results(metrics, model_name):
    """Print evaluation results in a readable format."""
    print('=' * 80)
    print(f'EVALUATION RESULTS: {model_name.upper()}')
    print('=' * 80)
    print()

    print('OVERALL METRICS:')
    print('-' * 80)
    print(f"Total images in ground truth:        {metrics['total_images']}")
    print(f"Images with annotations:              {metrics['images_with_ground_truth']}")
    print(f"Images detected correctly:            {metrics['images_detected_correctly']}")
    print(f"Images missed:                        {metrics['images_missed']}")
    print(f"Image-level accuracy:                 {metrics['image_accuracy']:.1%}")
    print()

    print('BOX-LEVEL METRICS:')
    print('-' * 80)
    print(f"Total ground truth boxes:             {metrics['total_ground_truth_boxes']}")
    print(f"Total detected boxes:                 {metrics['total_detected_boxes']}")
    print(f"True positives (correct detections):  {metrics['true_positives']}")
    print(f"False positives (wrong detections):   {metrics['false_positives']}")
    print(f"False negatives (missed boxes):       {metrics['false_negatives']}")
    print()
    print(f"Precision (of detected boxes):        {metrics['box_precision']:.1%}")
    print(f"Recall (of ground truth boxes):       {metrics['box_recall']:.1%}")
    print(f"F1 Score:                             {metrics['box_f1']:.3f}")
    print()

    print('FAILED DETECTIONS:')
    print('-' * 80)
    failed = [r for r in metrics['per_image_results'] if not r['correct']]
    if failed:
        for result in failed[:20]:  # Show first 20
            print(f"  {result['filename']}: "
                  f"{result['gt_boxes']} GT boxes, "
                  f"{result['detected_boxes']} detected, "
                  f"{result['matched']} matched")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
    else:
        print("  None! All images detected correctly.")
    print()
    print('=' * 80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate model against ground truth')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth JSON file')
    parser.add_argument('--model', default='yolov8m', help='YOLO model to evaluate (default: yolov8m)')
    parser.add_argument('--confidence', type=float, default=0.15, help='Detection confidence threshold (default: 0.15)')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IoU threshold for matching (default: 0.5)')
    parser.add_argument('--output-json', help='Optional path to save detailed results as JSON')

    args = parser.parse_args()

    print(f"\nEvaluating {args.model} against ground truth...")
    print(f"Ground truth file: {args.ground_truth}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"IoU threshold: {args.iou_threshold}")
    print()

    metrics = evaluate_model(
        args.ground_truth,
        model_name=args.model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou_threshold
    )

    print_results(metrics, args.model)

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nDetailed results saved to: {args.output_json}")


if __name__ == '__main__':
    main()
