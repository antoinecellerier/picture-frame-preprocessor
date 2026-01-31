"""Tests for utils module."""

import os
import tempfile
import pytest
from pathlib import Path
from PIL import Image
from frame_prep.utils import (
    is_image_file,
    ensure_directory,
    validate_image,
    get_output_path
)


def test_is_image_file():
    """Test image file detection."""
    assert is_image_file('test.jpg') is True
    assert is_image_file('test.jpeg') is True
    assert is_image_file('test.png') is True
    assert is_image_file('test.webp') is True
    assert is_image_file('test.JPG') is True  # Case insensitive
    assert is_image_file('test.txt') is False
    assert is_image_file('test.pdf') is False


def test_ensure_directory():
    """Test directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, 'test', 'nested', 'dir')
        ensure_directory(test_dir)
        assert os.path.isdir(test_dir)


def test_validate_image_nonexistent():
    """Test validation of nonexistent file."""
    assert validate_image('/nonexistent/file.jpg') is False


def test_validate_image_valid():
    """Test validation of valid image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, 'test.jpg')
        img = Image.new('RGB', (100, 100), color='red')
        img.save(img_path)

        assert validate_image(img_path) is True


def test_validate_image_invalid():
    """Test validation of invalid image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, 'test.jpg')
        with open(img_path, 'w') as f:
            f.write('not an image')

        assert validate_image(img_path) is False


def test_get_output_path():
    """Test output path generation."""
    input_path = '/path/to/input/image.jpg'
    output_dir = '/path/to/output'

    result = get_output_path(input_path, output_dir)
    assert result == '/path/to/output/image.jpg'


def test_get_output_path_with_suffix():
    """Test output path with suffix."""
    input_path = '/path/to/input/image.png'
    output_dir = '/path/to/output'

    result = get_output_path(input_path, output_dir, suffix='_processed')
    assert result == '/path/to/output/image_processed.jpg'
