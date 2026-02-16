"""Click-based CLI entry point with unified subcommands."""

import os
import sys
import functools
import click

from .preprocessor import ImagePreprocessor
from .detector import ArtFeatureDetector, EnsembleDetector, OptimizedEnsembleDetector
from .cropper import SmartCropper
from .utils import get_output_path, ensure_directory
from . import defaults


def common_options(f):
    """Decorator that adds shared detection/cropping options to a Click command."""
    @click.option('--width', '-w', default=defaults.TARGET_WIDTH, type=int,
                  help=f'Target width in pixels (default: {defaults.TARGET_WIDTH})')
    @click.option('--height', '-h', default=defaults.TARGET_HEIGHT, type=int,
                  help=f'Target height in pixels (default: {defaults.TARGET_HEIGHT})')
    @click.option('--strategy', '-s', type=click.Choice(['smart', 'saliency', 'center'], case_sensitive=False),
                  default=defaults.STRATEGY, help=f'Cropping strategy (default: {defaults.STRATEGY})')
    @click.option('--model', '-m', default='yolov8m',
                  help='YOLO model variant (default: yolov8m for better art detection)')
    @click.option('--confidence', '-c', default=defaults.CONFIDENCE_THRESHOLD, type=float,
                  help=f'Detection confidence threshold (default: {defaults.CONFIDENCE_THRESHOLD})')
    @click.option('--single-model', is_flag=True,
                  help='Use single YOLOv8 model instead of default optimized ensemble (faster, lower accuracy)')
    @click.option('--ensemble', is_flag=True,
                  help='Use ensemble detector (YOLOv8m + RT-DETR-L) instead of default optimized ensemble')
    @click.option('--zoom', '-z', default=defaults.ZOOM_FACTOR, type=float,
                  help=f'Zoom factor to focus on subjects (default: {defaults.ZOOM_FACTOR})')
    @click.option('--quality', '-q', default=defaults.JPEG_QUALITY, type=int,
                  help=f'JPEG quality 1-100 (default: {defaults.JPEG_QUALITY})')
    @click.option('--no-two-pass', is_flag=True,
                  help='Disable two-pass center-crop detection (faster, may miss small centered subjects)')
    @click.option('--no-filter', is_flag=True,
                  help='Disable non-art image filtering (process all images regardless of art score)')
    @click.option('--multi-crop', is_flag=True,
                  help='Generate one crop per viable art subject (e.g., multiple statues or mural panels)')
    @click.option('--verbose', '-v', is_flag=True, help='Verbose output')
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


def create_detector(single_model, ensemble, model, confidence, no_two_pass, verbose,
                    use_openvino=False):
    """Create a detector instance from CLI flags."""
    if single_model:
        detector = ArtFeatureDetector(
            model_name=model,
            confidence_threshold=confidence,
            **(dict(use_openvino=use_openvino) if use_openvino else {})
        )
        if verbose:
            click.echo(f"Using single model: {model}")
    elif ensemble:
        detector = EnsembleDetector(
            models=['yolov8m', 'rtdetr-l'],
            confidence_threshold=confidence,
            merge_threshold=defaults.MERGE_THRESHOLD,
            **(dict(use_openvino=use_openvino) if use_openvino else {})
        )
        if verbose:
            click.echo("Using ensemble detector: YOLOv8m + RT-DETR-L")
    else:
        # Default: optimized ensemble (YOLO-World + Grounding DINO)
        detector = OptimizedEnsembleDetector(
            confidence_threshold=defaults.CONFIDENCE_THRESHOLD,
            merge_threshold=defaults.MERGE_THRESHOLD,
            two_pass=defaults.TWO_PASS and not no_two_pass
        )
        if verbose:
            click.echo("Using optimized ensemble: YOLO-World + Grounding DINO")
    return detector


def create_cropper(width, height, zoom):
    """Create a SmartCropper instance from CLI options."""
    return SmartCropper(
        target_width=width,
        target_height=height,
        zoom_factor=zoom,
        use_saliency_fallback=defaults.USE_SALIENCY_FALLBACK
    )


@click.group()
@click.version_option(version="0.2.1")
def cli():
    """Picture Frame Preprocessor - Intelligent image preprocessing for e-ink frames with contextual zoom."""
    pass


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input image file path')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output directory or file path')
@common_options
def process(input, output, width, height, strategy, model, confidence, single_model,
            ensemble, zoom, quality, no_two_pass, no_filter, multi_crop, verbose):
    """Process a single image for e-ink display."""

    try:
        detector = create_detector(single_model, ensemble, model, confidence,
                                   no_two_pass, verbose)
        cropper = create_cropper(width, height, zoom)
        preprocessor = ImagePreprocessor(
            target_width=width,
            target_height=height,
            detector=detector,
            cropper=cropper,
            strategy=strategy,
            quality=quality,
            filter_non_art=defaults.FILTER_NON_ART and not no_filter,
            multi_crop=multi_crop
        )

        # Determine output path
        if os.path.isdir(output):
            output_path = get_output_path(input, output)
        else:
            output_path = output
            ensure_directory(os.path.dirname(output_path))

        if verbose:
            click.echo(f"Processing: {input}")
            click.echo(f"Target: {width}x{height}")
            click.echo(f"Strategy: {strategy}")
            click.echo()

        # Process image
        result = preprocessor.process_image(input, output_path, verbose=verbose)

        if result.filtered:
            click.secho(f"Filtered (not art, score: {result.art_score:.3f})", fg='yellow', bold=True)
            if verbose:
                click.echo(f"  Detections: {result.detections_found}")
                click.echo(f"  Score threshold: {defaults.MIN_ART_SCORE}")
        elif result.success:
            click.secho("Success!", fg='green', bold=True)
            if result.output_paths:
                click.echo(f"  Multi-crop: {len(result.output_paths)} outputs")
                for path in result.output_paths:
                    click.echo(f"    {path}")
            if verbose:
                if result.detections_found > 0:
                    click.echo(f"  Detections: {result.detections_found}")
                click.echo(f"  Strategy: {result.strategy_used}")
                if not result.output_paths:
                    click.echo(f"  Output: {result.output_path}")
        else:
            click.secho("Failed!", fg='red', bold=True)
            click.echo(f"  Error: {result.error_message}")
            raise click.Abort()

    except RuntimeError as e:
        if "Failed to load YOLO model" in str(e):
            click.secho("Model Error:", fg='red', bold=True)
            click.echo(str(e))
            click.echo()
            click.echo("To download models, run:")
            click.echo("  python scripts/download_models.py")
            raise click.Abort()
        raise

    except Exception as e:
        click.secho(f"Error: {e}", fg='red')
        if verbose:
            raise
        raise click.Abort()


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input directory containing images')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output directory for processed images')
@click.option('--workers', default=8, type=int,
              help='Number of parallel workers (default: 8)')
@click.option('--skip-existing', is_flag=True,
              help='Skip images that already exist in output directory')
@click.option('--recursive', '-r', is_flag=True,
              help='Process subdirectories recursively')
@click.option('--no-openvino', is_flag=True,
              help='Disable OpenVINO acceleration (use PyTorch instead)')
@click.option('--threads-per-worker', default=4, type=int,
              help='Number of threads per worker process (default: 4)')
@common_options
def batch(input, output, workers, skip_existing, recursive, no_openvino,
          threads_per_worker, width, height, strategy, model, confidence,
          single_model, ensemble, zoom, quality, no_two_pass, no_filter,
          multi_crop, verbose):
    """Batch process a directory of images for e-ink display."""
    from .batch import run_batch

    config = {
        'width': width,
        'height': height,
        'strategy': strategy,
        'model': model,
        'confidence': confidence,
        'single_model': single_model,
        'ensemble': ensemble,
        'zoom': zoom,
        'quality': quality,
        'skip_existing': skip_existing,
        'recursive': recursive,
        'use_openvino': not no_openvino,
        'two_pass': not no_two_pass,
        'threads_per_worker': threads_per_worker,
        'filter_non_art': defaults.FILTER_NON_ART and not no_filter,
        'multi_crop': multi_crop,
    }

    sys.exit(run_batch(input, output, config, workers=workers))


@cli.command()
@click.option('--input-dir', default='test_real_images/input/',
              type=click.Path(exists=True),
              help='Input directory with test images (default: test_real_images/input/)')
@click.option('--ground-truth', default='test_real_images/ground_truth_annotations.json',
              type=click.Path(exists=True),
              help='Ground truth annotations JSON (default: test_real_images/ground_truth_annotations.json)')
@click.option('--output-file', default='reports/interactive_detection_report.html',
              type=click.Path(),
              help='Output HTML report path (default: reports/interactive_detection_report.html)')
@common_options
def report(input_dir, ground_truth, output_file, width, height, strategy, model,
           confidence, single_model, ensemble, zoom, quality, no_two_pass,
           no_filter, multi_crop, verbose):
    """Generate an interactive HTML detection report."""
    from .report import generate_report as _generate_report

    detector = create_detector(single_model, ensemble, model, confidence,
                               no_two_pass, verbose)
    cropper = create_cropper(width, height, zoom)

    _generate_report(
        input_dir=input_dir,
        ground_truth_path=ground_truth,
        output_file=output_file,
        detector=detector,
        cropper=cropper,
        verbose=verbose,
    )


if __name__ == '__main__':
    cli()
