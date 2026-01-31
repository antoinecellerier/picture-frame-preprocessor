"""Tests for preprocessor module."""

import os
import tempfile
import pytest
from PIL import Image
from frame_prep.preprocessor import ImagePreprocessor, ProcessingResult
from frame_prep.detector import ArtFeatureDetector
from frame_prep.cropper import SmartCropper


def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = ImagePreprocessor(480, 800)

    assert preprocessor.target_width == 480
    assert preprocessor.target_height == 800
    assert preprocessor.strategy == 'smart'
    assert preprocessor.quality == 95
    assert isinstance(preprocessor.detector, ArtFeatureDetector)
    assert isinstance(preprocessor.cropper, SmartCropper)


def test_preprocessor_custom_components():
    """Test preprocessor with custom components."""
    detector = ArtFeatureDetector(model_name='yolov8s')
    cropper = SmartCropper(640, 480)

    preprocessor = ImagePreprocessor(
        640, 480,
        detector=detector,
        cropper=cropper,
        strategy='center',
        quality=90
    )

    assert preprocessor.detector is detector
    assert preprocessor.cropper is cropper
    assert preprocessor.strategy == 'center'
    assert preprocessor.quality == 90


def test_process_image_portrait():
    """Test processing portrait image (no crop needed)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create portrait test image
        input_path = os.path.join(tmpdir, 'input.jpg')
        output_path = os.path.join(tmpdir, 'output.jpg')

        img = Image.new('RGB', (480, 800), color='blue')
        img.save(input_path)

        preprocessor = ImagePreprocessor(480, 800, strategy='center')
        result = preprocessor.process_image(input_path, output_path)

        assert result.success is True
        assert result.output_path == output_path
        assert result.strategy_used == 'none'  # No crop needed
        assert os.path.exists(output_path)

        # Verify output dimensions
        with Image.open(output_path) as output_img:
            assert output_img.size == (480, 800)


def test_process_image_landscape():
    """Test processing landscape image (crop needed)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create landscape test image
        input_path = os.path.join(tmpdir, 'input.jpg')
        output_path = os.path.join(tmpdir, 'output.jpg')

        img = Image.new('RGB', (1920, 1080), color='green')
        img.save(input_path)

        preprocessor = ImagePreprocessor(480, 800, strategy='center')
        result = preprocessor.process_image(input_path, output_path)

        assert result.success is True
        assert result.output_path == output_path
        assert result.strategy_used == 'center'
        assert os.path.exists(output_path)

        # Verify output dimensions
        with Image.open(output_path) as output_img:
            assert output_img.size == (480, 800)


def test_process_image_invalid():
    """Test processing invalid image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, 'invalid.jpg')
        output_path = os.path.join(tmpdir, 'output.jpg')

        # Create invalid image file
        with open(input_path, 'w') as f:
            f.write('not an image')

        preprocessor = ImagePreprocessor(480, 800)
        result = preprocessor.process_image(input_path, output_path)

        assert result.success is False
        assert result.error_message is not None
        assert not os.path.exists(output_path)


def test_process_image_nonexistent():
    """Test processing nonexistent image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, 'nonexistent.jpg')
        output_path = os.path.join(tmpdir, 'output.jpg')

        preprocessor = ImagePreprocessor(480, 800)
        result = preprocessor.process_image(input_path, output_path)

        assert result.success is False
        assert result.error_message is not None


def test_save_output():
    """Test save_output method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, 'test.jpg')

        img = Image.new('RGB', (480, 800), color='red')
        preprocessor = ImagePreprocessor(480, 800, quality=85)

        preprocessor.save_output(img, output_path)

        assert os.path.exists(output_path)

        # Verify saved image
        with Image.open(output_path) as saved_img:
            assert saved_img.size == (480, 800)
            assert saved_img.format == 'JPEG'


def test_save_output_with_exif():
    """Test save_output preserves EXIF data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create image with EXIF
        input_path = os.path.join(tmpdir, 'input.jpg')
        img = Image.new('RGB', (480, 800), color='red')
        exif_data = b'fake_exif_data'
        img.save(input_path, exif=exif_data)

        # Load and get EXIF
        with Image.open(input_path) as loaded_img:
            exif = loaded_img.info.get('exif')

        # Save with EXIF
        output_path = os.path.join(tmpdir, 'output.jpg')
        preprocessor = ImagePreprocessor(480, 800)
        preprocessor.save_output(img, output_path, exif_data=exif)

        assert os.path.exists(output_path)
