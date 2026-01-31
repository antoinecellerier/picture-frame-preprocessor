"""Shared utility functions."""

import os
from pathlib import Path
from typing import Optional
from PIL import Image


SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp'}


def is_image_file(path: str) -> bool:
    """Check if file is a supported image format."""
    return Path(path).suffix.lower() in SUPPORTED_FORMATS


def ensure_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def validate_image(image_path: str) -> bool:
    """Validate that image file can be opened and processed."""
    if not os.path.exists(image_path):
        return False

    if not is_image_file(image_path):
        return False

    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_output_path(input_path: str, output_dir: str, suffix: str = "") -> str:
    """Generate output path for processed image."""
    input_name = Path(input_path).stem
    if suffix:
        output_name = f"{input_name}{suffix}.jpg"
    else:
        output_name = f"{input_name}.jpg"
    return os.path.join(output_dir, output_name)
