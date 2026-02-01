#!/usr/bin/env python3
"""Export YOLO models to OpenVINO format for optimized CPU inference."""

import argparse
import sys
from pathlib import Path


def export_model(model_path: Path, half: bool = False):
    """
    Export a YOLO model to OpenVINO format.

    Args:
        model_path: Path to the .pt model file
        half: Use FP16 precision (smaller, may be faster on some hardware)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed")
        print("Run: pip install -e .")
        return False

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return False

    print(f"\nExporting {model_path.name} to OpenVINO...")
    print(f"  Source: {model_path}")

    try:
        model = YOLO(str(model_path))

        # Export to OpenVINO
        export_path = model.export(
            format='openvino',
            half=half,
            dynamic=False,  # Fixed input size for better optimization
            simplify=True   # Simplify the model graph
        )

        print(f"  ✓ Exported successfully")
        print(f"  Output: {export_path}")

        # Check size
        output_dir = Path(export_path).parent if Path(export_path).is_file() else Path(export_path)
        if output_dir.exists():
            size_mb = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) / 1024 / 1024
            print(f"  Size: {size_mb:.1f} MB")

        return True

    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Export YOLO models to OpenVINO format for CPU optimization'
    )
    parser.add_argument(
        'models',
        nargs='*',
        help='Model names to export (e.g., yolov8m rtdetr-l). If not specified, exports common models.'
    )
    parser.add_argument(
        '--half',
        action='store_true',
        help='Use FP16 precision (smaller, may be faster)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Export all available .pt models in models/ directory'
    )

    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).parent.parent
    models_dir = project_root / 'models'

    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        return 1

    # Determine which models to export
    if args.all:
        # Export all .pt models
        model_files = sorted(models_dir.glob('*.pt'))
        # Skip YOLO-World models (they don't export well to OpenVINO)
        model_files = [m for m in model_files if 'world' not in m.name.lower()]
    elif args.models:
        # Export specified models
        model_files = []
        for model_name in args.models:
            if not model_name.endswith('.pt'):
                model_name += '.pt'
            model_path = models_dir / model_name
            if model_path.exists():
                model_files.append(model_path)
            else:
                print(f"Warning: Model not found: {model_path}")
    else:
        # Export commonly used models for this project
        common_models = ['yolov8m', 'rtdetr-l']
        model_files = []
        for model_name in common_models:
            model_path = models_dir / f'{model_name}.pt'
            if model_path.exists():
                model_files.append(model_path)
            else:
                print(f"Note: {model_name}.pt not found, skipping")

    if not model_files:
        print("\nNo models to export.")
        print(f"Available models in {models_dir}:")
        for pt_file in sorted(models_dir.glob('*.pt')):
            print(f"  - {pt_file.name}")
        return 1

    # Export models
    print("=" * 60)
    print("EXPORTING MODELS TO OPENVINO")
    print("=" * 60)
    print(f"Models directory: {models_dir}")
    print(f"Models to export: {len(model_files)}")
    print(f"Precision: {'FP16' if args.half else 'FP32'}")

    success_count = 0
    for model_path in model_files:
        if export_model(model_path, half=args.half):
            success_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"Total models: {len(model_files)}")
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {len(model_files) - success_count}")

    if success_count > 0:
        print("\nOpenVINO models are automatically used when available.")
        print("Run batch processing to see improved CPU performance!")

    return 0 if success_count == len(model_files) else 1


if __name__ == '__main__':
    sys.exit(main())
