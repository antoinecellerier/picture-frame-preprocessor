#!/usr/bin/env python3
"""
Sweep through different parameters to find best accuracy configuration.

Tests different merge and IoU thresholds to maximize image-level accuracy.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate_from_cache import evaluate_from_cache


def optimize_accuracy(ground_truth_path, models, confidence=0.15, cache_dir='cache'):
    """
    Sweep through parameters to find best accuracy.

    Args:
        ground_truth_path: Path to ground truth JSON
        models: List of models to ensemble
        confidence: Confidence threshold used in cache
        cache_dir: Cache directory
    """
    # Parameter ranges to test
    if len(models) > 1:
        merge_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    else:
        merge_thresholds = [0.3]  # Doesn't matter for single model

    iou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    results = []

    print(f"\n{'='*80}")
    print(f"OPTIMIZING ACCURACY FOR: {' + '.join(models)}")
    print(f"{'='*80}\n")

    print(f"Testing {len(merge_thresholds)} merge thresholds × {len(iou_thresholds)} IoU thresholds = {len(merge_thresholds) * len(iou_thresholds)} configurations...\n")

    for merge_thresh in merge_thresholds:
        for iou_thresh in iou_thresholds:
            if len(models) > 1:
                print(f"Testing merge={merge_thresh:.1f}, IoU={iou_thresh:.1f}... ", end='', flush=True)
            else:
                print(f"Testing IoU={iou_thresh:.1f}... ", end='', flush=True)

            metrics = evaluate_from_cache(
                ground_truth_path,
                models,
                confidence=confidence,
                merge_threshold=merge_thresh,
                iou_threshold=iou_thresh,
                cache_dir=cache_dir
            )

            results.append({
                'merge_threshold': merge_thresh,
                'iou_threshold': iou_thresh,
                'image_accuracy': metrics['image_accuracy'],
                'precision': metrics['box_precision'],
                'recall': metrics['box_recall'],
                'f1': metrics['box_f1'],
                'detected_boxes': metrics['total_detected_boxes']
            })

            print(f"Acc={metrics['image_accuracy']:.1%}, F1={metrics['box_f1']:.3f}")

    # Find best configuration
    print(f"\n{'='*80}")
    print("RESULTS SORTED BY IMAGE ACCURACY:")
    print(f"{'='*80}\n")

    results_sorted = sorted(results, key=lambda x: x['image_accuracy'], reverse=True)

    print(f"{'Rank':<6} {'Merge':<8} {'IoU':<8} {'Img Acc':<10} {'Precision':<12} {'Recall':<10} {'F1':<8} {'Boxes':<8}")
    print('-' * 80)

    for i, result in enumerate(results_sorted[:10], 1):
        merge_str = f"{result['merge_threshold']:.1f}" if len(models) > 1 else "N/A"
        print(f"{i:<6} {merge_str:<8} {result['iou_threshold']:.1f}     "
              f"{result['image_accuracy']:.1%}      "
              f"{result['precision']:.1%}        "
              f"{result['recall']:.1%}      "
              f"{result['f1']:.3f}    "
              f"{result['detected_boxes']}")

    # Best configuration
    best = results_sorted[0]

    print(f"\n{'='*80}")
    print("🏆 BEST CONFIGURATION:")
    print(f"{'='*80}")
    if len(models) > 1:
        print(f"Merge threshold:    {best['merge_threshold']:.1f}")
    print(f"IoU threshold:      {best['iou_threshold']:.1f}")
    print(f"Image accuracy:     {best['image_accuracy']:.1%}")
    print(f"Precision:          {best['precision']:.1%}")
    print(f"Recall:             {best['recall']:.1%}")
    print(f"F1 Score:           {best['f1']:.3f}")
    print(f"Detected boxes:     {best['detected_boxes']}")
    print(f"{'='*80}\n")

    return best


def main():
    parser = argparse.ArgumentParser(description='Optimize accuracy by sweeping parameters')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth JSON')
    parser.add_argument('--models', nargs='+', required=True, help='Models to ensemble')
    parser.add_argument('--confidence', type=float, default=0.15, help='Confidence threshold')
    parser.add_argument('--cache-dir', default='cache', help='Cache directory')

    args = parser.parse_args()

    best = optimize_accuracy(
        args.ground_truth,
        args.models,
        confidence=args.confidence,
        cache_dir=args.cache_dir
    )


if __name__ == '__main__':
    main()
