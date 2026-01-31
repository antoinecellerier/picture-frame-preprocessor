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
    """Test get_primary_subject prioritizes people."""
    detector = ArtFeatureDetector()

    detections = [
        Detection(bbox=(0, 0, 100, 100), confidence=0.9, class_name='car', area=10000),
        Detection(bbox=(100, 100, 200, 200), confidence=0.7, class_name='person', area=10000),
        Detection(bbox=(200, 200, 300, 300), confidence=0.6, class_name='dog', area=10000),
    ]

    primary = detector.get_primary_subject(detections)
    assert primary.class_name == 'person'


def test_get_primary_subject_no_person():
    """Test get_primary_subject without people."""
    detector = ArtFeatureDetector()

    detections = [
        Detection(bbox=(0, 0, 100, 100), confidence=0.9, class_name='car', area=10000),
        Detection(bbox=(100, 100, 200, 200), confidence=0.7, class_name='dog', area=10000),
    ]

    primary = detector.get_primary_subject(detections)
    assert primary.class_name == 'car'  # Highest confidence
    assert primary.confidence == 0.9
