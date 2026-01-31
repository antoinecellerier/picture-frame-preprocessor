#!/usr/bin/env python3
"""
Cache model detections to speed up evaluations.

Runs models once and saves all detections to JSON cache file.
Subsequent evaluations can use cached detections instead of rerunning models.
"""

import json
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import sys
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.frame_prep.detector import ArtFeatureDetector


def get_cache_key(model_name, confidence_threshold):
    """Generate cache key for model+config."""
    key = f"{model_name}_{confidence_threshold}"
    return hashlib.md5(key.encode()).hexdigest()


def cache_detections(ground_truth_path, models, confidence_threshold=0.15, cache_dir='cache'):
    """
    Run models and cache all detections.

    Args:
        ground_truth_path: Path to ground truth JSON
        models: List of model names
        confidence_threshold: Detection confidence threshold
        cache_dir: Directory to save cache files
    """
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    input_dir = Path('test_real_images/input')

    for model_name in models:
        print(f"\n{'='*80}")
        print(f"Caching detections for: {model_name} (confidence={confidence_threshold})")
        print(f"{'='*80}")

        cache_file = cache_path / f"{model_name}_conf{confidence_threshold}.json"

        if cache_file.exists():
            print(f"✓ Cache already exists: {cache_file}")
            response = input("Overwrite? (y/n): ").strip().lower()
            if response != 'y':
                print("Skipping...")
                continue

        detector = ArtFeatureDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold
        )

        all_detections = {}

        for idx, gt_entry in enumerate(ground_truth):
            filename = gt_entry['filename']
            print(f"[{idx+1}/{len(ground_truth)}] {filename}...")

            img_path = input_dir / filename

            if not img_path.exists():
                print(f"  Warning: Image not found")
                continue

            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)

            # Run detection
            detections = detector.detect(img, verbose=False)

            # Save detections
            detection_data = []
            for det in detections:
                detection_data.append({
                    'bbox': list(det.bbox),
                    'confidence': float(det.confidence),
                    'class_name': det.class_name,
                    'area': int(det.area)
                })

            all_detections[filename] = {
                'detections': detection_data,
                'image_size': (img.width, img.height)
            }

        # Save cache
        cache_data = {
            'model_name': model_name,
            'confidence_threshold': confidence_threshold,
            'detections': all_detections
        }

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

        print(f"\n✓ Cached {len(all_detections)} images to: {cache_file}")


def load_cached_detections(model_name, confidence_threshold=0.15, cache_dir='cache'):
    """
    Load cached detections for a model.

    Returns:
        dict: Filename -> detections mapping, or None if cache doesn't exist
    """
    cache_path = Path(cache_dir)
    cache_file = cache_path / f"{model_name}_conf{confidence_threshold}.json"

    if not cache_file.exists():
        return None

    with open(cache_file, 'r') as f:
        cache_data = json.load(f)

    return cache_data['detections']


def main():
    parser = argparse.ArgumentParser(description='Cache model detections')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth JSON')
    parser.add_argument('--models', nargs='+', required=True, help='Models to cache')
    parser.add_argument('--confidence', type=float, default=0.15, help='Confidence threshold')
    parser.add_argument('--cache-dir', default='cache', help='Cache directory')

    args = parser.parse_args()

    print(f"\nCaching detections for: {', '.join(args.models)}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Cache directory: {args.cache_dir}\n")

    cache_detections(
        args.ground_truth,
        args.models,
        confidence_threshold=args.confidence,
        cache_dir=args.cache_dir
    )

    print(f"\n{'='*80}")
    print("DONE! Cached detections can now be used for fast evaluations.")
    print("Use evaluate_from_cache.py to evaluate without rerunning models.")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
