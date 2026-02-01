"""Tests for detector module."""

import pytest
from frame_prep.detector import Detection, ArtFeatureDetector


def test_detection_dataclass():
    """Test Detection dataclass."""
    det = Detection(
        bbox=(100, 100, 200, 200),
        confidence=0.85,
        class_name='person',
        area=10000
    )

    assert det.bbox == (100, 100, 200, 200)
    assert det.confidence == 0.85
    assert det.class_name == 'person'
    assert det.area == 10000


def test_detection_center():
    """Test detection center calculation."""
    det = Detection(
        bbox=(100, 100, 200, 200),
        confidence=0.85,
        class_name='person',
        area=10000
    )

    center = det.center
    assert center == (150, 150)


def test_detector_initialization():
    """Test detector initialization."""
    detector = ArtFeatureDetector(
        model_name='yolov8n',
        confidence_threshold=0.3
    )

    assert detector.model_name == 'yolov8n'
    assert detector.confidence_threshold == 0.3
    assert detector._model is None  # Lazy loading


def test_get_primary_subject_empty():
    """Test get_primary_subject with no detections."""
    detector = ArtFeatureDetector()
    result = detector.get_primary_subject([])
    assert result is None


def test_get_primary_subject_with_person():
    """Test get_primary_subject deprioritizes people for art detection."""
    detector = ArtFeatureDetector()

    # Art detection: people are viewers, not subjects
    # 'dog' is in art_related_classes (animal sculptures) with 2.5x multiplier
    # 'person' has 0.5x multiplier (deprioritized)
    # 'car' has 1.5x multiplier (other objects)
    detections = [
        Detection(bbox=(0, 0, 100, 100), confidence=0.9, class_name='car', area=10000),
        Detection(bbox=(100, 100, 200, 200), confidence=0.7, class_name='person', area=10000),
        Detection(bbox=(200, 200, 300, 300), confidence=0.6, class_name='dog', area=10000),
    ]

    primary = detector.get_primary_subject(detections)
    # 'dog' wins: 0.6 * 2.5 = 1.5 > 'car': 0.9 * 1.5 = 1.35 > 'person': 0.7 * 0.5 = 0.35
    assert primary.class_name == 'dog'


def test_get_primary_subject_no_person():
    """Test get_primary_subject prioritizes art-related classes."""
    detector = ArtFeatureDetector()

    # 'dog' is art-related (animal sculptures) with 2.5x multiplier
    # 'car' has 1.5x multiplier (other objects)
    detections = [
        Detection(bbox=(0, 0, 100, 100), confidence=0.9, class_name='car', area=10000),
        Detection(bbox=(100, 100, 200, 200), confidence=0.7, class_name='dog', area=10000),
    ]

    primary = detector.get_primary_subject(detections)
    # 'dog' wins: 0.7 * 2.5 = 1.75 > 'car': 0.9 * 1.5 = 1.35
    assert primary.class_name == 'dog'
    assert primary.confidence == 0.7


def test_get_primary_subject_highest_confidence():
    """Test get_primary_subject with same class multipliers."""
    detector = ArtFeatureDetector()

    # Both are 'other' objects with 1.5x multiplier
    # Higher confidence wins
    detections = [
        Detection(bbox=(0, 0, 100, 100), confidence=0.9, class_name='car', area=10000),
        Detection(bbox=(100, 100, 200, 200), confidence=0.7, class_name='truck', area=10000),
    ]

    primary = detector.get_primary_subject(detections)
    assert primary.class_name == 'car'
    assert primary.confidence == 0.9
