#!/usr/bin/env python3
"""Download and initialize YOLOv8 models."""

import os
import sys


def download_models():
    """Download YOLOv8 models using Ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not installed")
        print("Run: pip install -e .")
        return 1

    models = ['yolov8n', 'yolov8s', 'yolov8m']

    print("Downloading YOLOv8 models...")
    print("This will download models to the Ultralytics cache directory.")
    print("yolov8m (52 MB) is the default for best art detection quality.\n")

    for model_name in models:
        print(f"Downloading {model_name}.pt...")
        try:
            model = YOLO(f'{model_name}.pt')
            print(f"✓ {model_name}.pt downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download {model_name}.pt: {e}")
            return 1

    print("\n" + "="*60)
    print("Model download complete!")
    print("="*60)
    print("\nModels are cached by Ultralytics and ready to use.")
    print("\nYou can now run:")
    print("  frame-prep process --input image.jpg --output out/ -v")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(download_models())
