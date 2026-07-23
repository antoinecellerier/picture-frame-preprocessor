"""Tests for utils module."""

import os
import tempfile
import pytest
from datetime import datetime
from pathlib import Path
from PIL import Image
from frame_prep.utils import (
    is_image_file,
    ensure_directory,
    validate_image,
    get_output_path,
    filter_by_mtime
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


@pytest.fixture
def dated_files():
    """Three files with mtimes on Jan 1, Jun 15, and Dec 31 of 2025."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for name, dt in [
            ('old.jpg', datetime(2025, 1, 1, 12, 0)),
            ('mid.jpg', datetime(2025, 6, 15, 12, 0)),
            ('new.jpg', datetime(2025, 12, 31, 12, 0)),
        ]:
            path = os.path.join(tmpdir, name)
            with open(path, 'w') as f:
                f.write('x')
            ts = dt.timestamp()
            os.utime(path, (ts, ts))
            paths.append(path)
        yield paths


def test_filter_by_mtime_no_bounds(dated_files):
    """No bounds returns all files."""
    assert filter_by_mtime(dated_files) == dated_files


def test_filter_by_mtime_since(dated_files):
    """Since bound keeps files modified at or after it."""
    result = filter_by_mtime(dated_files, since=datetime(2025, 6, 1))
    assert result == dated_files[1:]


def test_filter_by_mtime_until(dated_files):
    """Until bound keeps files modified strictly before it."""
    result = filter_by_mtime(dated_files, until=datetime(2025, 6, 1))
    assert result == dated_files[:1]


def test_filter_by_mtime_range(dated_files):
    """Range keeps only files inside [since, until)."""
    result = filter_by_mtime(
        dated_files,
        since=datetime(2025, 6, 1),
        until=datetime(2025, 7, 1),
    )
    assert result == [dated_files[1]]


def test_filter_by_mtime_since_inclusive(dated_files):
    """A file modified exactly at the since bound is kept."""
    result = filter_by_mtime(dated_files, since=datetime(2025, 6, 15, 12, 0))
    assert dated_files[1] in result


def test_filter_by_mtime_missing_file(dated_files):
    """Files that can't be stat'd are excluded."""
    paths = dated_files + [os.path.join(os.path.dirname(dated_files[0]), 'gone.jpg')]
    assert filter_by_mtime(paths) == dated_files
