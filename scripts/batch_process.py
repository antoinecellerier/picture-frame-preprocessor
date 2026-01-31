#!/usr/bin/env python3
"""Batch processing script for directories."""

import os
import sys
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from frame_prep.preprocessor import ImagePreprocessor, ProcessingResult
from frame_prep.detector import ArtFeatureDetector, EnsembleDetector, OptimizedEnsembleDetector
from frame_prep.cropper import SmartCropper
from frame_prep.utils import is_image_file, get_output_path, ensure_directory


@dataclass
class BatchStats:
    """Statistics for batch processing."""
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def process_single_image(args):
    """
    Process a single image (for multiprocessing).

    Args:
        args: Tuple of (input_path, output_path, config_dict)

    Returns:
        ProcessingResult
    """
    input_path, output_path, config = args

    # Check if output exists and skip if requested
    if config.get('skip_existing') and os.path.exists(output_path):
        return ProcessingResult(
            success=True,
            input_path=input_path,
            output_path=output_path,
            strategy_used='skipped',
            detections_found=0
        )

    # Create preprocessor (each worker needs its own)
    if config.get('optimized', False):
        # Use optimized ensemble for BEST accuracy (96.8%)
        detector = OptimizedEnsembleDetector(
            confidence_threshold=0.25,
            merge_threshold=0.2
        )
    elif config.get('ensemble', False):
        # Use ensemble detector for good accuracy (63.5%)
        detector = EnsembleDetector(
            models=['yolov8m', 'rtdetr-l'],
            confidence_threshold=config['confidence'],
            merge_threshold=0.4
        )
    else:
        # Use single model detector
        detector = ArtFeatureDetector(
            model_name=config['model'],
            confidence_threshold=config['confidence']
        )
    cropper = SmartCropper(
        target_width=config['width'],
        target_height=config['height'],
        zoom_factor=config.get('zoom', 1.3),
        use_saliency_fallback=True
    )
    preprocessor = ImagePreprocessor(
        target_width=config['width'],
        target_height=config['height'],
        detector=detector,
        cropper=cropper,
        strategy=config['strategy'],
        quality=config['quality']
    )

    result = preprocessor.process_image(input_path, output_path, verbose=False)

    # Save ML analysis data as JSON for quality reports
    if result.success and result.output_path:
        json_path = result.output_path.rsplit('.', 1)[0] + '_analysis.json'
        analysis_data = {
            'filename': os.path.basename(input_path),
            'original_dimensions': result.original_dimensions,
            'detections': result.detections,
            'crop_box': result.crop_box,
            'zoom_applied': result.zoom_applied,
            'strategy_used': result.strategy_used,
            'detections_found': result.detections_found
        }

        try:
            with open(json_path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
        except Exception as e:
            # Don't fail the whole process if JSON save fails
            pass

    return result


def collect_images(input_dir: str) -> List[str]:
    """
    Recursively collect all image files from directory.

    Args:
        input_dir: Directory to scan

    Returns:
        List of image file paths
    """
    images = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if is_image_file(file_path):
                images.append(file_path)
    return sorted(images)


def main():
    parser = argparse.ArgumentParser(
        description='Batch process images for e-ink display'
    )
    parser.add_argument(
        '--input-dir', '-i',
        required=True,
        help='Input directory containing images'
    )
    parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='Output directory for processed images'
    )
    parser.add_argument(
        '--width', '-w',
        type=int,
        default=480,
        help='Target width in pixels (default: 480)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=800,
        help='Target height in pixels (default: 800)'
    )
    parser.add_argument(
        '--strategy', '-s',
        choices=['smart', 'saliency', 'center'],
        default='smart',
        help='Cropping strategy (default: smart)'
    )
    parser.add_argument(
        '--model', '-m',
        default='yolov8m',
        help='YOLO model variant (default: yolov8m for better detection)'
    )
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.15,
        help='Detection confidence threshold (default: 0.15 for more detections)'
    )
    parser.add_argument(
        '--ensemble',
        action='store_true',
        help='Use ensemble detector (YOLOv8m + RT-DETR-L) for good accuracy (63.5%%)'
    )
    parser.add_argument(
        '--optimized',
        action='store_true',
        help='Use optimized ensemble (YOLO-World + Grounding DINO) for BEST accuracy (96.8%%)'
    )
    parser.add_argument(
        '--zoom', '-z',
        type=float,
        default=1.3,
        help='Zoom factor to focus on subjects (default: 1.3)'
    )
    parser.add_argument(
        '--quality', '-q',
        type=int,
        default=95,
        help='JPEG quality 1-100 (default: 95)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel workers (default: 8, optimized for 16-thread CPUs)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip images that already exist in output directory'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Process subdirectories recursively'
    )

    args = parser.parse_args()

    # Validate directories
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1

    ensure_directory(args.output_dir)

    # Collect images
    print("Scanning for images...")
    if args.recursive:
        images = collect_images(args.input_dir)
    else:
        images = [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if is_image_file(os.path.join(args.input_dir, f))
        ]

    if not images:
        print("No images found in input directory")
        return 0

    print(f"Found {len(images)} images")

    # Prepare processing tasks
    config = {
        'width': args.width,
        'height': args.height,
        'strategy': args.strategy,
        'model': args.model,
        'confidence': args.confidence,
        'ensemble': args.ensemble,
        'optimized': args.optimized,
        'zoom': args.zoom,
        'quality': args.quality,
        'skip_existing': args.skip_existing
    }

    tasks = [
        (img, get_output_path(img, args.output_dir), config)
        for img in images
    ]

    # Process images
    stats = BatchStats(total=len(images))

    print(f"\nProcessing images (strategy: {args.strategy}, workers: {args.workers})...")
    print(f"Target: {args.width}x{args.height}\n")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_image, task): task for task in tasks}

        with tqdm(total=len(tasks), unit='img') as pbar:
            for future in as_completed(futures):
                task = futures[future]
                input_path = task[0]

                try:
                    result = future.result()
                    if result.success:
                        if result.strategy_used == 'skipped':
                            stats.skipped += 1
                        else:
                            stats.success += 1
                    else:
                        stats.failed += 1
                        stats.errors.append(f"{input_path}: {result.error_message}")
                except Exception as e:
                    stats.failed += 1
                    stats.errors.append(f"{input_path}: {str(e)}")

                pbar.update(1)

    # Print summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total images:     {stats.total}")
    print(f"✓ Successful:     {stats.success}")
    if stats.skipped > 0:
        print(f"⊘ Skipped:        {stats.skipped}")
    print(f"✗ Failed:         {stats.failed}")

    if stats.errors:
        print("\nErrors:")
        for error in stats.errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more")

    print(f"\nOutput directory: {args.output_dir}")
    print("="*60)

    return 0 if stats.failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
