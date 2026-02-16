"""Batch processing logic for directories of images."""

import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List

from tqdm import tqdm

from .preprocessor import ImagePreprocessor, ProcessingResult
from .detector import ArtFeatureDetector, EnsembleDetector, OptimizedEnsembleDetector
from .cropper import SmartCropper
from .utils import is_image_file, get_output_path, ensure_directory
from . import defaults


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
    import os
    import torch

    threads_per_worker = config.get('threads_per_worker', 4)

    os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
    os.environ['MKL_NUM_THREADS'] = str(threads_per_worker)
    os.environ['OPENVINO_INFERENCE_NUM_THREADS'] = str(threads_per_worker)
    torch.set_num_threads(threads_per_worker)

    # Create detector once per worker
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
            merge_threshold=defaults.MERGE_THRESHOLD,
            use_openvino=use_openvino
        )
    else:
        # Default: optimized ensemble (YOLO-World + Grounding DINO)
        _detector = OptimizedEnsembleDetector(
            confidence_threshold=defaults.CONFIDENCE_THRESHOLD,
            merge_threshold=defaults.MERGE_THRESHOLD,
            two_pass=config.get('two_pass', defaults.TWO_PASS)
        )

    _cropper = SmartCropper(
        target_width=config['width'],
        target_height=config['height'],
        zoom_factor=config.get('zoom', defaults.ZOOM_FACTOR),
        use_saliency_fallback=defaults.USE_SALIENCY_FALLBACK
    )

    _preprocessor = ImagePreprocessor(
        target_width=config['width'],
        target_height=config['height'],
        detector=_detector,
        cropper=_cropper,
        strategy=config['strategy'],
        quality=config['quality'],
        filter_non_art=config.get('filter_non_art', defaults.FILTER_NON_ART),
        multi_crop=config.get('multi_crop', False)
    )


def process_single_image(args):
    """
    Process a single image using pre-loaded models.

    Uses the global _preprocessor instance that was initialized once per worker.

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

    result = _preprocessor.process_image(input_path, output_path, verbose=False)

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
            'detections_found': result.detections_found,
        }
        if result.output_paths:
            analysis_data['output_paths'] = result.output_paths

        try:
            with open(json_path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
        except Exception:
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


def run_batch(input_dir, output_dir, config, workers=8):
    """
    Run batch processing on a directory of images.

    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for processed images
        config: Configuration dictionary for worker processes
        workers: Number of parallel workers

    Returns:
        0 on success, 1 if any failures
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1

    ensure_directory(output_dir)

    # Collect images
    print("Scanning for images...")
    if config.get('recursive', False):
        images = collect_images(input_dir)
    else:
        images = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if is_image_file(os.path.join(input_dir, f))
        ]

    if not images:
        print("No images found in input directory")
        return 0

    print(f"Found {len(images)} images")

    tasks = [
        (img, get_output_path(img, output_dir), config)
        for img in images
    ]

    stats = BatchStats(total=len(images))

    print(f"\nProcessing images (strategy: {config['strategy']}, workers: {workers})...")
    print(f"Target: {config['width']}x{config['height']}")

    # Show optimization info
    optimizations = []
    if config.get('use_openvino', True):
        optimizations.append("OpenVINO")
    optimizations.append(f"{config.get('threads_per_worker', 4)} threads/worker")

    if not config.get('single_model') and not config.get('ensemble'):
        print("Optimized ensemble: YOLO-World + Grounding DINO (models cached per worker)")
    print(f"Optimizations: {', '.join(optimizations)}\n")

    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(config,)) as executor:
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
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total images:     {stats.total}")
    print(f"Successful:       {stats.success}")
    if stats.output_files > stats.success:
        print(f"  Output files:   {stats.output_files} (multi-crop)")
    if stats.filtered > 0:
        print(f"Filtered:         {stats.filtered} (non-art)")
    if stats.skipped > 0:
        print(f"Skipped:          {stats.skipped}")
    print(f"Failed:           {stats.failed}")

    if stats.errors:
        print("\nErrors:")
        for error in stats.errors[:10]:
            print(f"  - {error}")
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more")

    print(f"\nOutput directory: {output_dir}")
    print("=" * 60)

    return 0 if stats.failed == 0 else 1
