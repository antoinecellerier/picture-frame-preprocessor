#!/usr/bin/env python3
"""Optimize YOLO-World + Grounding DINO ensemble parameters."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.evaluate_from_cache import calculate_iou, merge_boxes

def evaluate_ensemble_config(yolo_cache_name, merge_threshold, iou_threshold):
    """Evaluate ensemble with specific configuration."""

    with open('test_real_images/ground_truth_annotations.json', 'r') as f:
        ground_truth = json.load(f)

    # Load caches
    with open(f'cache/{yolo_cache_name}.json', 'r') as f:
        yolo_cache = json.load(f)

    with open('cache/grounding_dino_conf0.25.json', 'r') as f:
        dino_cache = json.load(f)

    metrics = {
        'images_with_ground_truth': 0,
        'images_detected_correctly': 0,
        'total_ground_truth_boxes': 0,
        'total_detected_boxes': 0,
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0
    }

    for gt_entry in ground_truth:
        filename = gt_entry['filename']

        gt_boxes = []
        for box_data in gt_entry.get('manual_boxes', []):
            gt_boxes.append(box_data['bbox'])
        for det_data in gt_entry.get('correct_detections', []):
            gt_boxes.append(det_data['bbox'])

        if len(gt_boxes) == 0:
            continue

        metrics['images_with_ground_truth'] += 1
        metrics['total_ground_truth_boxes'] += len(gt_boxes)

        # Combine detections
        all_boxes = []

        if filename in yolo_cache['detections']:
            for det in yolo_cache['detections'][filename]['detections']:
                all_boxes.append((det['bbox'], det['confidence']))

        if filename in dino_cache['detections']:
            for det in dino_cache['detections'][filename]['detections']:
                all_boxes.append((det['bbox'], det['confidence']))

        # Merge
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

    # Calculate metrics
    metrics['image_accuracy'] = metrics['images_detected_correctly'] / metrics['images_with_ground_truth']
    metrics['box_recall'] = metrics['true_positives'] / metrics['total_ground_truth_boxes'] if metrics['total_ground_truth_boxes'] > 0 else 0
    metrics['box_precision'] = metrics['true_positives'] / metrics['total_detected_boxes'] if metrics['total_detected_boxes'] > 0 else 0

    if metrics['box_precision'] + metrics['box_recall'] > 0:
        metrics['box_f1'] = 2 * (metrics['box_precision'] * metrics['box_recall']) / (metrics['box_precision'] + metrics['box_recall'])
    else:
        metrics['box_f1'] = 0.0

    return metrics

# Test configurations
print("\nOptimizing YOLO-World + Grounding DINO Ensemble\n")
print("="*80)

configs = []

# Test both YOLO-World versions
for yolo_name in ['yolo_world_conf0.25', 'yolo_world_improved_conf0.25']:
    # Test merge thresholds
    for merge_t in [0.2, 0.3, 0.4, 0.5]:
        # Test IoU thresholds
        for iou_t in [0.3, 0.4, 0.5]:
            metrics = evaluate_ensemble_config(yolo_name, merge_t, iou_t)

            configs.append({
                'yolo_version': 'Improved' if 'improved' in yolo_name else 'Original',
                'merge_threshold': merge_t,
                'iou_threshold': iou_t,
                'accuracy': metrics['image_accuracy'],
                'precision': metrics['box_precision'],
                'recall': metrics['box_recall'],
                'f1': metrics['box_f1'],
                'detected': metrics['images_detected_correctly']
            })

# Sort by accuracy
configs.sort(key=lambda x: x['accuracy'], reverse=True)

# Print top 10
print(f"\n{'Rank':<6} {'YOLO':<10} {'Merge':<7} {'IoU':<6} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Detected'}")
print('-' * 80)

for i, config in enumerate(configs[:10], 1):
    print(f"{i:<6} {config['yolo_version']:<10} {config['merge_threshold']:<7.1f} "
          f"{config['iou_threshold']:<6.1f} {config['accuracy']:<8.1%} "
          f"{config['precision']:<8.1%} {config['recall']:<8.1%} "
          f"{config['f1']:<8.3f} {config['detected']}/63")

best = configs[0]
print(f"\n{'='*80}")
print("🏆 BEST CONFIGURATION:")
print(f"{'='*80}")
print(f"YOLO-World version:  {best['yolo_version']}")
print(f"Merge threshold:     {best['merge_threshold']:.1f}")
print(f"IoU threshold:       {best['iou_threshold']:.1f}")
print(f"Image accuracy:      {best['accuracy']:.1%} ({best['detected']}/63)")
print(f"Precision:           {best['precision']:.1%}")
print(f"Recall:              {best['recall']:.1%}")
print(f"F1 Score:            {best['f1']:.3f}")
print(f"{'='*80}\n")
