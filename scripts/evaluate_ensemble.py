#!/usr/bin/env python3
"""
Evaluate ensemble of multiple models with box merging.

Combines detections from multiple models and merges overlapping boxes.
"""

import json
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.frame_prep.detector import ArtFeatureDetector


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def merge_boxes(boxes, iou_threshold=0.3):
    """
    Merge overlapping boxes using NMS-like approach.

    Args:
        boxes: List of (bbox, confidence) tuples
        iou_threshold: IoU threshold for merging

    Returns:
        List of merged boxes
    """
    if not boxes:
        return []

    # Sort by confidence
    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)

    merged = []
    used = set()

    for i, (box1, conf1) in enumerate(boxes):
        if i in used:
            continue

        # Find all boxes that overlap with this one
        to_merge = [(box1, conf1)]

        for j, (box2, conf2) in enumerate(boxes[i+1:], start=i+1):
            if j in used:
                continue

            iou = calculate_iou(box1, box2)
            if iou > iou_threshold:
                to_merge.append((box2, conf2))
                used.add(j)

        # Merge boxes (weighted average by confidence)
        if len(to_merge) == 1:
            merged.append(box1)
        else:
            total_conf = sum(c for _, c in to_merge)
            x1 = sum(b[0] * c for b, c in to_merge) / total_conf
            y1 = sum(b[1] * c for b, c in to_merge) / total_conf
            x2 = sum(b[2] * c for b, c in to_merge) / total_conf
            y2 = sum(b[3] * c for b, c in to_merge) / total_conf
            merged.append([int(x1), int(y1), int(x2), int(y2)])

    return merged


def evaluate_ensemble(ground_truth_path, models, confidence_threshold=0.15,
                     merge_threshold=0.3, iou_threshold=0.5):
    """
    Evaluate ensemble of models.

    Args:
        ground_truth_path: Path to ground truth JSON
        models: List of model names to ensemble
        confidence_threshold: Detection confidence threshold
        merge_threshold: IoU threshold for merging boxes
        iou_threshold: IoU threshold for matching with ground truth
    """
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    # Initialize all detectors
    detectors = {}
    for model_name in models:
        print(f"Loading {model_name}...")
        detectors[model_name] = ArtFeatureDetector(
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
            continue

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)

        # Get ground truth
        gt_boxes = []
        for box_data in gt_entry.get('manual_boxes', []):
            gt_boxes.append(box_data['bbox'])
        for det_data in gt_entry.get('correct_detections', []):
            gt_boxes.append(det_data['bbox'])

        if len(gt_boxes) == 0:
            continue

        metrics['images_with_ground_truth'] += 1
        metrics['total_ground_truth_boxes'] += len(gt_boxes)

        # Run all detectors and combine
        all_boxes = []
        for model_name, detector in detectors.items():
            detections = detector.detect(img, verbose=False)
            for det in detections:
                all_boxes.append((det.bbox, det.confidence))

        # Merge overlapping boxes
        merged_boxes = merge_boxes(all_boxes, merge_threshold)

        metrics['total_detected_boxes'] += len(merged_boxes)

        # Match to ground truth
        matched_gt = set()
        matched_det = set()

        for det_idx, det_box in enumerate(merged_boxes):
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

        metrics['false_positives'] += len(merged_boxes) - len(matched_det)
        metrics['false_negatives'] += len(gt_boxes) - len(matched_gt)

        image_correct = len(matched_gt) > 0
        if image_correct:
            metrics['images_detected_correctly'] += 1
        else:
            metrics['images_missed'] += 1

        metrics['per_image_results'].append({
            'filename': filename,
            'gt_boxes': len(gt_boxes),
            'detected_boxes': len(merged_boxes),
            'matched': len(matched_gt),
            'correct': image_correct
        })

    # Calculate metrics
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


def print_results(metrics, models):
    """Print results."""
    model_names = ' + '.join(models)

    print('=' * 80)
    print(f'ENSEMBLE EVALUATION: {model_names}')
    print('=' * 80)
    print()

    print('OVERALL METRICS:')
    print('-' * 80)
    print(f"Images with annotations:              {metrics['images_with_ground_truth']}")
    print(f"Images detected correctly:            {metrics['images_detected_correctly']}")
    print(f"Images missed:                        {metrics['images_missed']}")
    print(f"Image-level accuracy:                 {metrics['image_accuracy']:.1%}")
    print()

    print('BOX-LEVEL METRICS:')
    print('-' * 80)
    print(f"Total ground truth boxes:             {metrics['total_ground_truth_boxes']}")
    print(f"Total detected boxes (after merge):   {metrics['total_detected_boxes']}")
    print(f"True positives:                       {metrics['true_positives']}")
    print(f"False positives:                      {metrics['false_positives']}")
    print(f"False negatives:                      {metrics['false_negatives']}")
    print()
    print(f"Precision:                            {metrics['box_precision']:.1%}")
    print(f"Recall:                               {metrics['box_recall']:.1%}")
    print(f"F1 Score:                             {metrics['box_f1']:.3f}")
    print()
    print('=' * 80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate ensemble of models')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth JSON')
    parser.add_argument('--models', nargs='+', required=True, help='Models to ensemble')
    parser.add_argument('--confidence', type=float, default=0.15, help='Confidence threshold')
    parser.add_argument('--merge-threshold', type=float, default=0.3, help='IoU threshold for merging boxes')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IoU threshold for GT matching')

    args = parser.parse_args()

    print(f"\nEvaluating ensemble: {' + '.join(args.models)}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Merge threshold: {args.merge_threshold}")
    print(f"IoU threshold: {args.iou_threshold}\n")

    metrics = evaluate_ensemble(
        args.ground_truth,
        args.models,
        confidence_threshold=args.confidence,
        merge_threshold=args.merge_threshold,
        iou_threshold=args.iou_threshold
    )

    print_results(metrics, args.models)


if __name__ == '__main__':
    main()
