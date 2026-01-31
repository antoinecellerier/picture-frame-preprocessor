#!/usr/bin/env python3
"""
Evaluate models using cached detections.

Allows fast testing of different merge thresholds, IoU thresholds, etc.
without rerunning the models.
"""

import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_iou(box1, box2):
    """Calculate IoU."""
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
    """Merge overlapping boxes."""
    if not boxes:
        return []

    # Sort by confidence
    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)

    merged = []
    used = set()

    for i, (box1, conf1) in enumerate(boxes):
        if i in used:
            continue

        to_merge = [(box1, conf1)]

        for j, (box2, conf2) in enumerate(boxes[i+1:], start=i+1):
            if j in used:
                continue

            iou = calculate_iou(box1, box2)
            if iou > iou_threshold:
                to_merge.append((box2, conf2))
                used.add(j)

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


def load_cache(model_name, confidence, cache_dir='cache'):
    """Load cached detections."""
    cache_path = Path(cache_dir) / f"{model_name}_conf{confidence}.json"

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}\nRun cache_model_detections.py first!")

    with open(cache_path, 'r') as f:
        return json.load(f)


def evaluate_from_cache(ground_truth_path, models, confidence=0.15,
                       merge_threshold=0.3, iou_threshold=0.5, cache_dir='cache'):
    """Evaluate using cached detections."""

    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    # Load all caches
    caches = {}
    for model_name in models:
        print(f"Loading cache for {model_name}...")
        caches[model_name] = load_cache(model_name, confidence, cache_dir)

    metrics = {
        'total_images': len(ground_truth),
        'images_with_ground_truth': 0,
        'images_detected_correctly': 0,
        'images_missed': 0,
        'total_ground_truth_boxes': 0,
        'total_detected_boxes': 0,
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0
    }

    for gt_entry in ground_truth:
        filename = gt_entry['filename']

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

        # Combine detections from all models
        all_boxes = []
        for model_name in models:
            cache_data = caches[model_name]['detections']

            if filename not in cache_data:
                continue

            for det in cache_data[filename]['detections']:
                all_boxes.append((det['bbox'], det['confidence']))

        # Merge if multiple models
        if len(models) > 1:
            merged_boxes = merge_boxes(all_boxes, merge_threshold)
        else:
            merged_boxes = [box for box, _ in all_boxes]

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

        if len(matched_gt) > 0:
            metrics['images_detected_correctly'] += 1
        else:
            metrics['images_missed'] += 1

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


def print_results(metrics, models, merge_threshold, iou_threshold):
    """Print results."""
    model_names = ' + '.join(models) if len(models) > 1 else models[0]

    print()
    print('=' * 80)
    print(f'EVALUATION: {model_names}')
    if len(models) > 1:
        print(f'Merge threshold: {merge_threshold}')
    print(f'IoU threshold: {iou_threshold}')
    print('=' * 80)
    print()

    print('METRICS:')
    print('-' * 80)
    print(f"Images detected correctly:            {metrics['images_detected_correctly']}/{metrics['images_with_ground_truth']}")
    print(f"Image-level accuracy:                 {metrics['image_accuracy']:.1%}")
    print()
    print(f"Total detected boxes:                 {metrics['total_detected_boxes']}")
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
    parser = argparse.ArgumentParser(description='Evaluate from cached detections')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth JSON')
    parser.add_argument('--models', nargs='+', required=True, help='Models to evaluate')
    parser.add_argument('--confidence', type=float, default=0.15, help='Confidence used in cache')
    parser.add_argument('--merge-threshold', type=float, default=0.3, help='IoU threshold for merging')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IoU threshold for GT matching')
    parser.add_argument('--cache-dir', default='cache', help='Cache directory')

    args = parser.parse_args()

    print(f"\nEvaluating from cache: {' + '.join(args.models)}")

    metrics = evaluate_from_cache(
        args.ground_truth,
        args.models,
        confidence=args.confidence,
        merge_threshold=args.merge_threshold,
        iou_threshold=args.iou_threshold,
        cache_dir=args.cache_dir
    )

    print_results(metrics, args.models, args.merge_threshold, args.iou_threshold)


if __name__ == '__main__':
    main()
