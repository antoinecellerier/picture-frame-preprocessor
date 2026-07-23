"""Shared utility functions."""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from PIL import Image


SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp'}


def is_image_file(path: str) -> bool:
    """Check if file is a supported image format."""
    return Path(path).suffix.lower() in SUPPORTED_FORMATS


def filter_by_mtime(paths: List[str],
                    since: Optional[datetime] = None,
                    until: Optional[datetime] = None) -> List[str]:
    """
    Filter paths to those with mtime in [since, until).

    Args:
        paths: File paths to filter
        since: Keep files modified at or after this time (inclusive)
        until: Keep files modified before this time (exclusive)

    Returns:
        Filtered list, preserving input order. Files that can't be
        stat'd (e.g. deleted between listing and filtering) are excluded.
    """
    since_ts = since.timestamp() if since else None
    until_ts = until.timestamp() if until else None

    result = []
    for path in paths:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if since_ts is not None and mtime < since_ts:
            continue
        if until_ts is not None and mtime >= until_ts:
            continue
        result.append(path)
    return result


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
