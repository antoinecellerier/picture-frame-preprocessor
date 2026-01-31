#!/usr/bin/env python3
"""Generate random test set from source directory."""

import os
import sys
import argparse
import random
import shutil
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from frame_prep.utils import is_image_file


def generate_test_set(source_dir, output_dir, count, seed=None):
    """
    Generate random test set from source directory.

    Args:
        source_dir: Directory containing source images
        output_dir: Directory to copy test images to
        count: Number of random images to select
        seed: Random seed for reproducibility

    Returns:
        List of selected image filenames
    """
    if seed is not None:
        random.seed(seed)

    # Find all images in source directory
    all_images = [
        f for f in os.listdir(source_dir)
        if is_image_file(os.path.join(source_dir, f))
    ]

    if len(all_images) < count:
        print(f"Warning: Only {len(all_images)} images available, requested {count}")
        count = len(all_images)

    # Select random sample
    selected = random.sample(all_images, count)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Copy images
    print(f"Copying {count} images to {output_dir}...")
    for filename in selected:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(output_dir, filename)
        shutil.copy2(src, dst)

    print(f"✓ Test set created: {count} images")
    return selected


def main():
    parser = argparse.ArgumentParser(
        description='Generate random test set from source directory'
    )
    parser.add_argument(
        '--source', '-s',
        required=True,
        help='Source directory containing images'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for test set'
    )
    parser.add_argument(
        '--count', '-n',
        type=int,
        default=64,
        help='Number of images to select (default: 64)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    if not os.path.isdir(args.source):
        print(f"Error: Source directory does not exist: {args.source}")
        return 1

    selected = generate_test_set(
        args.source,
        args.output,
        args.count,
        args.seed
    )

    # Print summary
    print(f"\nSelected images saved to: {args.output}")
    print(f"Total images: {len(selected)}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
