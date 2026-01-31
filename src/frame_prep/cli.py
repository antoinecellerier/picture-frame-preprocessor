"""Click-based CLI entry point."""

import os
import click
from pathlib import Path

from .preprocessor import ImagePreprocessor
from .detector import ArtFeatureDetector, EnsembleDetector, OptimizedEnsembleDetector
from .cropper import SmartCropper
from .utils import get_output_path, ensure_directory


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
    default=480,
    type=int,
    help='Target width in pixels (default: 480)'
)
@click.option(
    '--height', '-h',
    default=800,
    type=int,
    help='Target height in pixels (default: 800)'
)
@click.option(
    '--strategy', '-s',
    type=click.Choice(['smart', 'saliency', 'center'], case_sensitive=False),
    default='smart',
    help='Cropping strategy (default: smart)'
)
@click.option(
    '--model', '-m',
    default='yolov8m',
    help='YOLO model variant (default: yolov8m for better art detection)'
)
@click.option(
    '--confidence', '-c',
    default=0.15,
    type=float,
    help='Detection confidence threshold (default: 0.15 for more detections)'
)
@click.option(
    '--ensemble',
    is_flag=True,
    help='Use ensemble detector (YOLOv8m + RT-DETR-L) for good accuracy (63.5%)'
)
@click.option(
    '--optimized',
    is_flag=True,
    help='Use optimized ensemble (YOLO-World + Grounding DINO) for BEST accuracy (96.8%)'
)
@click.option(
    '--zoom', '-z',
    default=1.3,
    type=float,
    help='Zoom factor to focus on subjects (default: 1.3)'
)
@click.option(
    '--quality', '-q',
    default=95,
    type=int,
    help='JPEG quality 1-100 (default: 95)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Verbose output'
)
def process(input, output, width, height, strategy, model, confidence, ensemble, optimized, zoom, quality, verbose):
    """Process a single image for e-ink display."""

    try:
        # Initialize components
        if optimized:
            # Use optimized ensemble for BEST accuracy (96.8%)
            detector = OptimizedEnsembleDetector(
                confidence_threshold=0.25,  # Optimized threshold
                merge_threshold=0.2  # Optimized merge threshold
            )
            if verbose:
                click.echo("Using OPTIMIZED ensemble: YOLO-World + Grounding DINO (96.8% accuracy)")
        elif ensemble:
            # Use ensemble detector for good accuracy (63.5%)
            detector = EnsembleDetector(
                models=['yolov8m', 'rtdetr-l'],
                confidence_threshold=confidence,
                merge_threshold=0.4
            )
            if verbose:
                click.echo("Using ensemble detector: YOLOv8m + RT-DETR-L (63.5% accuracy)")
        else:
            # Use single model detector
            detector = ArtFeatureDetector(
                model_name=model,
                confidence_threshold=confidence
            )
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
            quality=quality
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

        if result.success:
            click.secho("✓ Success!", fg='green', bold=True)
            if verbose:
                if result.detections_found > 0:
                    click.echo(f"  Detections: {result.detections_found}")
                click.echo(f"  Strategy: {result.strategy_used}")
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
