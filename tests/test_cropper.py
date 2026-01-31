"""Tests for cropper module."""

import pytest
from PIL import Image
from frame_prep.cropper import SmartCropper
from frame_prep.detector import Detection


def test_cropper_initialization():
    """Test cropper initialization."""
    cropper = SmartCropper(480, 800)
    assert cropper.target_width == 480
    assert cropper.target_height == 800
    assert cropper.target_aspect == 800 / 480


def test_needs_cropping_landscape():
    """Test detection of landscape images needing crop."""
    cropper = SmartCropper(480, 800)
    landscape_img = Image.new('RGB', (1920, 1080))
    assert cropper.needs_cropping(landscape_img) is True


def test_needs_cropping_portrait():
    """Test detection of portrait images not needing crop."""
    cropper = SmartCropper(480, 800)
    portrait_img = Image.new('RGB', (1080, 1920))
    assert cropper.needs_cropping(portrait_img) is False


def test_crop_center():
    """Test center cropping."""
    cropper = SmartCropper(480, 800)
    img = Image.new('RGB', (1920, 1080))

    cropped = cropper.crop_center(img)

    # Check that aspect ratio is correct
    width, height = cropped.size
    aspect = height / width
    target_aspect = 800 / 480
    assert abs(aspect - target_aspect) < 0.01


def test_crop_with_detections():
    """Test cropping with ML detections."""
    cropper = SmartCropper(480, 800)
    img = Image.new('RGB', (1920, 1080))

    # Create mock detection in center
    detection = Detection(
        bbox=(900, 500, 1020, 580),
        confidence=0.8,
        class_name='person',
        area=120 * 80
    )

    cropped = cropper.crop_with_detections(img, [detection])

    # Check aspect ratio
    width, height = cropped.size
    aspect = height / width
    target_aspect = 800 / 480
    assert abs(aspect - target_aspect) < 0.01


def test_calculate_crop_window():
    """Test crop window calculation."""
    cropper = SmartCropper(480, 800)

    # Test with centered anchor
    image_size = (1920, 1080)
    anchor_point = (960, 540)

    left, top, right, bottom = cropper._calculate_crop_window(image_size, anchor_point)

    # Check bounds
    assert left >= 0
    assert right <= 1920
    assert top == 0
    assert bottom == 1080

    # Check dimensions maintain aspect ratio
    width = right - left
    height = bottom - top
    aspect = height / width
    target_aspect = 800 / 480
    assert abs(aspect - target_aspect) < 0.01


def test_calculate_crop_window_edge_left():
    """Test crop window at left edge."""
    cropper = SmartCropper(480, 800)

    image_size = (1920, 1080)
    anchor_point = (100, 540)  # Near left edge

    left, top, right, bottom = cropper._calculate_crop_window(image_size, anchor_point)

    assert left >= 0
    assert right <= 1920


def test_calculate_crop_window_edge_right():
    """Test crop window at right edge."""
    cropper = SmartCropper(480, 800)

    image_size = (1920, 1080)
    anchor_point = (1820, 540)  # Near right edge

    left, top, right, bottom = cropper._calculate_crop_window(image_size, anchor_point)

    assert left >= 0
    assert right <= 1920
