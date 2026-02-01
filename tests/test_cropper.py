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
    """Test detection of portrait images not needing crop when aspect matches."""
    cropper = SmartCropper(480, 800)
    # Use exact target aspect ratio (800/480 = 1.667)
    # 600 * 1.667 = 1000
    portrait_img = Image.new('RGB', (600, 1000))
    assert cropper.needs_cropping(portrait_img) is False


def test_needs_cropping_portrait_different_aspect():
    """Test detection of portrait images needing crop when aspect differs."""
    cropper = SmartCropper(480, 800)
    # 1080x1920 has aspect 1.778, target is 1.667 - needs cropping
    portrait_img = Image.new('RGB', (1080, 1920))
    assert cropper.needs_cropping(portrait_img) is True


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


def test_contextual_zoom_large_subject():
    """Test that large subjects get minimal zoom."""
    cropper = SmartCropper(480, 800, zoom_factor=2.0)

    # Subject fills 70% of frame - no zoom needed
    zoom = cropper._calculate_contextual_zoom(
        subject_bbox=(100, 50, 540, 810),  # 440x760
        crop_width=600,
        crop_height=1000
    )
    assert zoom == 1.0


def test_contextual_zoom_tall_subject():
    """Test that tall thin subjects (filling height) get minimal zoom."""
    cropper = SmartCropper(480, 800, zoom_factor=2.0)

    # Subject is tall but narrow - fills 80% of height
    # Should NOT zoom much despite small area
    zoom = cropper._calculate_contextual_zoom(
        subject_bbox=(250, 100, 350, 900),  # 100x800 (tall thin)
        crop_width=600,
        crop_height=1000
    )
    # height_ratio = 800/1000 = 0.8 >= 0.65, so no zoom
    assert zoom == 1.0


def test_contextual_zoom_small_subject():
    """Test that small subjects get moderate zoom."""
    cropper = SmartCropper(480, 800, zoom_factor=2.0)

    # Subject fills 30% of frame - needs zoom
    zoom = cropper._calculate_contextual_zoom(
        subject_bbox=(200, 350, 400, 650),  # 200x300
        crop_width=600,
        crop_height=1000
    )
    # max_ratio = 300/1000 = 0.3, target 0.7
    # zoom_needed = 0.7/0.3 = 2.33, capped at zoom_factor 2.0
    assert zoom > 1.0
    assert zoom <= 2.0


def test_contextual_zoom_tiny_subject():
    """Test that tiny subjects get aggressive zoom (up to max)."""
    cropper = SmartCropper(480, 800, zoom_factor=1.5)

    # Subject is tiny - only 10% of frame
    zoom = cropper._calculate_contextual_zoom(
        subject_bbox=(270, 450, 330, 550),  # 60x100
        crop_width=600,
        crop_height=1000
    )
    # max_ratio = 100/1000 = 0.1, target 0.7
    # zoom_needed = 0.7/0.1 = 7.0, capped at zoom_factor 1.5
    assert zoom == 1.5  # Capped at max zoom factor
