"""Click-based CLI entry point."""

import os
import click
from pathlib import Path

from .preprocessor import ImagePreprocessor
from .detector import ArtFeatureDetector, EnsembleDetector, OptimizedEnsembleDetector
from .cropper import SmartCropper
from .utils import get_output_path, ensure_directory
from . import defaults


@click.group()
@click.version_option(version="0.2.1")
def cli():
    """Picture Frame Preprocessor - Intelligent image preprocessing for e-ink frames with contextual zoom."""
    pass


@cli.command()
@click.option(
    '--input', '-i',
    required=True,
    type=click.Path(exists=True),
    help='Input image file path'
)
@click.option(
    '--output', '-o',
    required=True,
    type=click.Path(),
    help='Output directory or file path'
)
@click.option(
    '--width', '-w',
    default=defaults.TARGET_WIDTH,
    type=int,
    help=f'Target width in pixels (default: {defaults.TARGET_WIDTH})'
)
@click.option(
    '--height', '-h',
    default=defaults.TARGET_HEIGHT,
    type=int,
    help=f'Target height in pixels (default: {defaults.TARGET_HEIGHT})'
)
@click.option(
    '--strategy', '-s',
    type=click.Choice(['smart', 'saliency', 'center'], case_sensitive=False),
    default=defaults.STRATEGY,
    help=f'Cropping strategy (default: {defaults.STRATEGY})'
)
@click.option(
    '--model', '-m',
    default='yolov8m',
    help='YOLO model variant (default: yolov8m for better art detection)'
)
@click.option(
    '--confidence', '-c',
    default=defaults.CONFIDENCE_THRESHOLD,
    type=float,
    help=f'Detection confidence threshold (default: {defaults.CONFIDENCE_THRESHOLD})'
)
@click.option(
    '--single-model',
    is_flag=True,
    help='Use single YOLOv8 model instead of default optimized ensemble (faster, lower accuracy)'
)
@click.option(
    '--ensemble',
    is_flag=True,
    help='Use ensemble detector (YOLOv8m + RT-DETR-L) instead of default optimized ensemble'
)
@click.option(
    '--zoom', '-z',
    default=defaults.ZOOM_FACTOR,
    type=float,
    help=f'Zoom factor to focus on subjects (default: {defaults.ZOOM_FACTOR})'
)
@click.option(
    '--quality', '-q',
    default=defaults.JPEG_QUALITY,
    type=int,
    help=f'JPEG quality 1-100 (default: {defaults.JPEG_QUALITY})'
)
@click.option(
    '--no-two-pass',
    is_flag=True,
    help='Disable two-pass center-crop detection (faster, may miss small centered subjects)'
)
@click.option(
    '--no-filter',
    is_flag=True,
    help='Disable non-art image filtering (process all images regardless of art score)'
)
@click.option(
    '--multi-crop',
    is_flag=True,
    help='Generate one crop per viable art subject (e.g., multiple statues or mural panels)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Verbose output'
)
def process(input, output, width, height, strategy, model, confidence, single_model, ensemble, zoom, quality, no_two_pass, no_filter, multi_crop, verbose):
    """Process a single image for e-ink display."""

    try:
        # Initialize components
        if single_model:
            # Use single YOLO model (faster, lower accuracy)
            detector = ArtFeatureDetector(
                model_name=model,
                confidence_threshold=confidence
            )
            if verbose:
                click.echo(f"Using single model: {model}")
        elif ensemble:
            # Use ensemble detector (YOLOv8m + RT-DETR-L)
            detector = EnsembleDetector(
                models=['yolov8m', 'rtdetr-l'],
                confidence_threshold=confidence,
                merge_threshold=0.4
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
        cropper = SmartCropper(
            target_width=width,
            target_height=height,
            zoom_factor=zoom,
            use_saliency_fallback=True
        )
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
            click.secho(f"⊘ Filtered (not art, score: {result.art_score:.3f})", fg='yellow', bold=True)
            if verbose:
                click.echo(f"  Detections: {result.detections_found}")
                click.echo(f"  Score threshold: {defaults.MIN_ART_SCORE}")
        elif result.success:
            click.secho("✓ Success!", fg='green', bold=True)
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
            click.secho("✗ Failed!", fg='red', bold=True)
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


if __name__ == '__main__':
    cli()
