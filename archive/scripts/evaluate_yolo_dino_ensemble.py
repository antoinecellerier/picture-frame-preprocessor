#!/usr/bin/env python3
"""Evaluate YOLO-World + Grounding DINO ensemble."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate_from_cache import calculate_iou, merge_boxes

def evaluate_ensemble(ground_truth_path, merge_threshold=0.3, iou_threshold=0.3):
    """Evaluate YOLO-World + Grounding DINO ensemble."""

    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    # Load both caches
    with open('cache/yolo_world_conf0.25.json', 'r') as f:
        yolo_world_cache = json.load(f)

    with open('cache/grounding_dino_conf0.25.json', 'r') as f:
        grounding_dino_cache = json.load(f)

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
        'yolo_only': 0,  # Detections only YOLO-World found
        'dino_only': 0,  # Detections only Grounding DINO found
        'both': 0        # Detections both found
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

        # Combine detections from both models
        all_boxes = []

        # YOLO-World detections
        if filename in yolo_world_cache['detections']:
            for det in yolo_world_cache['detections'][filename]['detections']:
                all_boxes.append((det['bbox'], det['confidence']))

        # Grounding DINO detections
        if filename in grounding_dino_cache['detections']:
            for det in grounding_dino_cache['detections'][filename]['detections']:
                all_boxes.append((det['bbox'], det['confidence']))

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

def print_results(metrics):
    """Print evaluation results."""
    print()
    print('=' * 80)
    print('YOLO-WORLD + GROUNDING DINO ENSEMBLE')
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate YOLO-World + Grounding DINO ensemble')
    parser.add_argument('--ground-truth', default='test_real_images/ground_truth_annotations.json')
    parser.add_argument('--merge-threshold', type=float, default=0.3)
    parser.add_argument('--iou-threshold', type=float, default=0.3)

    args = parser.parse_args()

    metrics = evaluate_ensemble(args.ground_truth, args.merge_threshold, args.iou_threshold)
    print_results(metrics)
