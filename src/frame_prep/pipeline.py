"""Shared detection pipeline used by both preprocessor and report."""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from pathlib import Path
from PIL import Image

from .detector import Detection, ArtFeatureDetector
from .analyzer import TEXT_RATIO_THRESHOLD


@dataclass
class DetectionResult:
    """Result of running the detection pipeline on a single image."""
    all_detections: List[Detection] = field(default_factory=list)
    filtered_detections: List[Detection] = field(default_factory=list)
    text_filtered: List[Detection] = field(default_factory=list)  # Detections removed by text filter
    primary: Optional[Detection] = None
    art_score: float = 0.0
    focal_detections: List[Detection] = field(default_factory=list)
    vlm_detections: List[Detection] = field(default_factory=list)
    vlm_primary: bool = False


def run_detection_pipeline(
    img: Image.Image,
    detector,
    cropper=None,
    *,
    image_path: Union[str, Path, None] = None,
    verbose: bool = False,
) -> DetectionResult:
    """Run the full detection pipeline: detect → primary selection → text
    filter → focal point detection.

    This is the single source of truth for detection logic, used by both
    the preprocessor (batch processing) and the report (analysis/display).

    Args:
        img: PIL Image (already EXIF-transposed)
        detector: OptimizedEnsembleDetector instance
        cropper: SmartCropper instance (needed for text filtering and focal
                 point detection; if None, those steps are skipped)
        image_path: Optional path for cache lookups
        verbose: Print progress info

    Returns:
        DetectionResult with all pipeline outputs
    """
    # === Step 1: Run detection ===
    try:
        detections = detector.detect(img, verbose=verbose, image_path=image_path)
    except TypeError:
        detections = detector.detect(img, verbose=verbose)

    # === Step 2: Primary selection ===
    primary = None
    art_score = 0.0
    if detections and hasattr(detector, 'get_primary_subject_with_score'):
        primary, art_score = detector.get_primary_subject_with_score(detections)
    elif detections:
        primary = detector.get_primary_subject(detections)

    # === Step 3: Text-heavy primary filter ===
    # If primary's region is >10% text, remove it and re-select from
    # remaining detections. Filters signs, labels, exhibit info panels.
    remaining = list(detections)
    text_filtered = []
    if primary is not None and cropper is not None:
        while primary is not None:
            text_ratio = cropper._text_detector.text_ratio(img, primary.bbox)
            if text_ratio <= TEXT_RATIO_THRESHOLD:
                break
            if verbose:
                print(f"  Skipping text-heavy primary: {primary.class_name} "
                      f"({text_ratio:.0%} text)")
            text_filtered.append(primary)
            remaining = [d for d in remaining if d is not primary]
            if remaining and hasattr(detector, 'get_primary_subject_with_score'):
                primary, art_score = detector.get_primary_subject_with_score(remaining)
            else:
                primary, art_score = None, 0.0

    # === Step 4: Focal point detection ===
    focal_detections = []
    if (primary is not None
            and cropper is not None
            and hasattr(detector, 'detect_focal_points')
            and not ArtFeatureDetector.is_3d_art(primary.class_name)
            and cropper.primary_fills_frame(primary.bbox, img.size)):
        focal_detections = detector.detect_focal_points(
            img, primary.bbox, verbose=verbose)
        if focal_detections and verbose:
            print(f"  Focal pass: {len(focal_detections)} detections")

    # === VLM metadata ===
    vlm_raw = getattr(detector, '_last_vlm_detections', [])
    vlm_primary = (primary is not None and vlm_raw
                   and any(primary.bbox == d.bbox for d in vlm_raw))

    return DetectionResult(
        all_detections=detections,
        filtered_detections=remaining,
        text_filtered=text_filtered,
        primary=primary,
        art_score=art_score,
        focal_detections=focal_detections,
        vlm_detections=vlm_raw,
        vlm_primary=vlm_primary,
    )
