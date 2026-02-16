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
from frame_prep import defaults


@dataclass
class BatchStats:
    """Statistics for batch processing."""
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    filtered: int = 0
    output_files: int = 0  # Total output files (may exceed success when multi-crop)
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


# Global detector instance (one per worker process)
_detector = None
_cropper = None
_preprocessor = None


def init_worker(config):
    """
    Initialize worker process with pre-loaded models.

    This function runs once per worker process, loading models into memory
    and reusing them for all images processed by that worker.

    Args:
        config: Configuration dictionary with detector settings
    """
    global _detector, _cropper, _preprocessor

    # Optimize threading for multi-process batch processing
    # Each worker gets fewer threads to avoid over-subscription
    import os
    import torch

    # Calculate optimal threads per worker
    # With 8 workers on 16-core CPU: 16/8 = 2 threads per worker is ideal
    # But allow some overlap for I/O: use 3-4 threads per worker
    threads_per_worker = config.get('threads_per_worker', 4)

    os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
    os.environ['MKL_NUM_THREADS'] = str(threads_per_worker)
    os.environ['OPENVINO_INFERENCE_NUM_THREADS'] = str(threads_per_worker)
    torch.set_num_threads(threads_per_worker)

    # Create detector once per worker
    # use_openvino defaults to True for best CPU performance
    use_openvino = config.get('use_openvino', True)

    if config.get('single_model', False):
        _detector = ArtFeatureDetector(
            model_name=config['model'],
            confidence_threshold=config['confidence'],
            use_openvino=use_openvino
        )
    elif config.get('ensemble', False):
        _detector = EnsembleDetector(
            models=['yolov8m', 'rtdetr-l'],
            confidence_threshold=config['confidence'],
            merge_threshold=0.4,
            use_openvino=use_openvino
        )
    else:
        # Default: optimized ensemble (YOLO-World + Grounding DINO)
        _detector = OptimizedEnsembleDetector(
            confidence_threshold=defaults.CONFIDENCE_THRESHOLD,
            merge_threshold=defaults.MERGE_THRESHOLD,
            two_pass=config.get('two_pass', defaults.TWO_PASS)
        )

    # Create cropper once per worker
    _cropper = SmartCropper(
        target_width=config['width'],
        target_height=config['height'],
        zoom_factor=config.get('zoom', defaults.ZOOM_FACTOR),
        use_saliency_fallback=True
    )

    # Create preprocessor once per worker
    _preprocessor = ImagePreprocessor(
        target_width=config['width'],
        target_height=config['height'],
        detector=_detector,
        cropper=_cropper,
        strategy=config['strategy'],
        quality=config['quality'],
        multi_crop=config.get('multi_crop', False)
    )


def process_single_image(args):
    """
    Process a single image using pre-loaded models.

    Uses the global _preprocessor instance that was initialized once per worker.
    This avoids reloading models for every image (6.7x speedup).

    Args:
        args: Tuple of (input_path, output_path, config_dict)

    Returns:
        ProcessingResult
    """
    global _preprocessor

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

    # Use pre-loaded preprocessor (models already in memory)
    result = _preprocessor.process_image(input_path, output_path, verbose=False)

    # Save ML analysis data as JSON for quality reports
    if result.success and result.output_path:
        # Use the base output_path (first crop for multi-crop) for analysis JSON
        json_path = result.output_path.rsplit('.', 1)[0] + '_analysis.json'
        analysis_data = {
            'filename': os.path.basename(input_path),
            'original_dimensions': result.original_dimensions,
            'detections': result.detections,
            'crop_box': result.crop_box,
            'zoom_applied': result.zoom_applied,
            'strategy_used': result.strategy_used,
            'detections_found': result.detections_found,
        }
        if result.output_paths:
            analysis_data['output_paths'] = result.output_paths

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
        default=defaults.TARGET_WIDTH,
        help=f'Target width in pixels (default: {defaults.TARGET_WIDTH})'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=defaults.TARGET_HEIGHT,
        help=f'Target height in pixels (default: {defaults.TARGET_HEIGHT})'
    )
    parser.add_argument(
        '--strategy', '-s',
        choices=['smart', 'saliency', 'center'],
        default=defaults.STRATEGY,
        help=f'Cropping strategy (default: {defaults.STRATEGY})'
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
        '--single-model',
        action='store_true',
        help='Use single YOLOv8 model instead of default optimized ensemble (faster, lower accuracy)'
    )
    parser.add_argument(
        '--ensemble',
        action='store_true',
        help='Use ensemble detector (YOLOv8m + RT-DETR-L) instead of default optimized ensemble'
    )
    parser.add_argument(
        '--zoom', '-z',
        type=float,
        default=defaults.ZOOM_FACTOR,
        help=f'Zoom factor to focus on subjects (default: {defaults.ZOOM_FACTOR})'
    )
    parser.add_argument(
        '--quality', '-q',
        type=int,
        default=defaults.JPEG_QUALITY,
        help=f'JPEG quality 1-100 (default: {defaults.JPEG_QUALITY})'
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
    parser.add_argument(
        '--no-openvino',
        action='store_true',
        help='Disable OpenVINO acceleration (use PyTorch instead)'
    )
    parser.add_argument(
        '--no-two-pass',
        action='store_true',
        help='Disable two-pass center-crop detection (faster, may miss small centered subjects)'
    )
    parser.add_argument(
        '--threads-per-worker',
        type=int,
        default=4,
        help='Number of threads per worker process (default: 4, optimal for multi-process)'
    )
    parser.add_argument(
        '--multi-crop',
        action='store_true',
        help='Generate one crop per viable art subject (e.g., multiple statues or mural panels)'
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
        'single_model': args.single_model,
        'ensemble': args.ensemble,
        'zoom': args.zoom,
        'quality': args.quality,
        'skip_existing': args.skip_existing,
        'use_openvino': not args.no_openvino,
        'two_pass': not args.no_two_pass,
        'threads_per_worker': args.threads_per_worker,
        'multi_crop': args.multi_crop
    }

    tasks = [
        (img, get_output_path(img, args.output_dir), config)
        for img in images
    ]

    # Process images
    stats = BatchStats(total=len(images))

    print(f"\nProcessing images (strategy: {args.strategy}, workers: {args.workers})...")
    print(f"Target: {args.width}x{args.height}")

    # Show optimization info
    optimizations = []
    if config['use_openvino']:
        optimizations.append("OpenVINO")
    optimizations.append(f"{args.threads_per_worker} threads/worker")

    if not args.single_model and not args.ensemble:
        print("🚀 Optimized ensemble: YOLO-World + Grounding DINO (models cached per worker)")
    print(f"⚡ Optimizations: {', '.join(optimizations)}\n")

    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(config,)) as executor:
        futures = {executor.submit(process_single_image, task): task for task in tasks}

        with tqdm(total=len(tasks), unit='img') as pbar:
            for future in as_completed(futures):
                task = futures[future]
                input_path = task[0]

                try:
                    result = future.result()
                    if result.filtered:
                        stats.filtered += 1
                    elif result.success:
                        if result.strategy_used == 'skipped':
                            stats.skipped += 1
                        else:
                            stats.success += 1
                            if result.output_paths:
                                stats.output_files += len(result.output_paths)
                            else:
                                stats.output_files += 1
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
    if stats.output_files > stats.success:
        print(f"  Output files:   {stats.output_files} (multi-crop)")
    if stats.filtered > 0:
        print(f"⊘ Filtered:       {stats.filtered} (non-art)")
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
