"""Core pipeline orchestration for image preprocessing."""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS

from .detector import ArtFeatureDetector
from .cropper import SmartCropper
from .utils import validate_image, ensure_directory
from . import defaults


@dataclass
class ProcessingResult:
    """Result of image processing."""
    success: bool
    input_path: str
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    strategy_used: Optional[str] = None
    detections_found: int = 0
    # ML analysis data for quality reports
    original_dimensions: Optional[Tuple[int, int]] = None  # (width, height)
    detections: List[Dict[str, Any]] = field(default_factory=list)  # Detection details
    crop_box: Optional[Tuple[int, int, int, int]] = None  # (left, top, right, bottom)
    zoom_applied: Optional[float] = None  # Contextual zoom factor used
    filtered: bool = False  # True if image was filtered as non-art
    art_score: Optional[float] = None  # Raw art score (confidence * class_multiplier)
    output_paths: List[str] = field(default_factory=list)  # Multi-crop output paths


class ImagePreprocessor:
    """Main image preprocessing pipeline."""

    def __init__(
        self,
        target_width: int,
        target_height: int,
        detector: Optional[ArtFeatureDetector] = None,
        cropper: Optional[SmartCropper] = None,
        strategy: str = 'smart',
        quality: int = 95,
        filter_non_art: bool = defaults.FILTER_NON_ART,
        multi_crop: bool = False
    ):
        """
        Initialize preprocessor.

        Args:
            target_width: Target output width in pixels
            target_height: Target output height in pixels
            detector: Object detector (created if None)
            cropper: Image cropper (created if None)
            strategy: Default cropping strategy ('smart', 'saliency', 'center')
            quality: JPEG quality (1-100)
            filter_non_art: Filter out non-art images by score threshold
            multi_crop: Generate one crop per viable art subject
        """
        self.target_width = target_width
        self.target_height = target_height
        self.strategy = strategy
        self.quality = quality
        self.filter_non_art = filter_non_art
        self.multi_crop = multi_crop

        self.detector = detector or ArtFeatureDetector()
        self.cropper = cropper or SmartCropper(target_width, target_height)

    def process_image(
        self,
        input_path: str,
        output_path: str,
        verbose: bool = False
    ) -> ProcessingResult:
        """
        Process single image through the pipeline.

        Args:
            input_path: Path to input image
            output_path: Path to save processed image
            verbose: Print processing details

        Returns:
            ProcessingResult with outcome details
        """
        # Validate input
        if not validate_image(input_path):
            return ProcessingResult(
                success=False,
                input_path=input_path,
                error_message="Invalid or corrupted image file"
            )

        try:
            # Load image
            with Image.open(input_path) as img:
                # Apply EXIF orientation (fixes rotated images)
                img = ImageOps.exif_transpose(img)

                # Convert to RGB if necessary
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')

                # Store EXIF data
                exif_data = img.info.get('exif', None)

                width, height = img.size
                original_dimensions = (width, height)

                if verbose:
                    print(f"Input image: {width}x{height}")

                # Initialize ML analysis data
                detections_list = []
                crop_box = None
                zoom_applied = 1.0

                # Run detection for smart strategy (needed for both
                # cropping and non-art filtering)
                detections = []
                art_score = None
                needs_crop = self.cropper.needs_cropping(img)

                if self.strategy == 'smart' and (needs_crop or self.filter_non_art):
                    # Pass image_path for cache lookups (OptimizedEnsembleDetector)
                    try:
                        detections = self.detector.detect(img, verbose=verbose, image_path=input_path)
                    except TypeError:
                        # Fallback for detectors that don't support image_path
                        detections = self.detector.detect(img, verbose=verbose)

                    # Get primary subject and art score
                    primary = None
                    if detections and hasattr(self.detector, 'get_primary_subject_with_score'):
                        primary, art_score = self.detector.get_primary_subject_with_score(detections)
                    elif detections:
                        primary = self.detector.get_primary_subject(detections)

                    # Filter non-art images by score threshold
                    if self.filter_non_art and art_score is not None and art_score < defaults.MIN_ART_SCORE:
                        if verbose:
                            print(f"Filtered as non-art (score: {art_score:.3f} < {defaults.MIN_ART_SCORE})")
                        return ProcessingResult(
                            success=True,
                            input_path=input_path,
                            filtered=True,
                            art_score=art_score,
                            detections_found=len(detections),
                            original_dimensions=original_dimensions,
                        )

                    # Capture detection details for ML analysis
                    for det in detections:
                        detections_list.append({
                            'bbox': det.bbox,
                            'confidence': det.confidence,
                            'class_name': det.class_name,
                            'is_primary': det == primary
                        })

                # Check if cropping is needed
                if needs_crop:
                    if verbose:
                        print(f"Aspect ratio mismatch, applying {self.strategy} cropping...")

                    # Multi-crop path: one output per viable art subject
                    if self.multi_crop and self.strategy == 'smart' and detections:
                        multi_results = self.cropper.crop_all_subjects(img, detections)

                        if len(multi_results) >= 2:
                            output_paths = []
                            base, ext = os.path.splitext(output_path)

                            for i, (cropped, det, zoom) in enumerate(multi_results, 1):
                                suffixed_path = f"{base}_{i}{ext}"
                                resized = cropped
                                if cropped.size != (self.target_width, self.target_height):
                                    resized = cropped.resize(
                                        (self.target_width, self.target_height),
                                        Image.LANCZOS
                                    )
                                self.save_output(resized, suffixed_path, exif_data, verbose=verbose)
                                output_paths.append(suffixed_path)
                                if verbose:
                                    print(f"  Multi-crop {i}: {det.class_name} ({det.confidence:.2f}), zoom {zoom:.2f}x -> {suffixed_path}")

                            return ProcessingResult(
                                success=True,
                                input_path=input_path,
                                output_path=output_paths[0],
                                output_paths=output_paths,
                                strategy_used='multi_crop',
                                detections_found=len(detections),
                                original_dimensions=original_dimensions,
                                detections=detections_list,
                                zoom_applied=multi_results[0][2]
                            )

                    # Crop image
                    img = self.cropper.crop_image(img, detections, self.strategy)

                    # Capture crop box and zoom from cropper
                    crop_box = self.cropper.last_crop_box
                    zoom_applied = self.cropper.last_zoom_applied

                    strategy_used = self.strategy
                    detections_found = len(detections)

                    if verbose:
                        print(f"Cropped to: {img.size[0]}x{img.size[1]}")
                else:
                    if verbose:
                        print("Image is already portrait/square, skipping crop")
                    strategy_used = 'none'
                    detections_found = 0

                # Resize to exact target dimensions
                if img.size != (self.target_width, self.target_height):
                    img = img.resize(
                        (self.target_width, self.target_height),
                        Image.LANCZOS
                    )
                    if verbose:
                        print(f"Resized to target: {self.target_width}x{self.target_height}")

                # Save output
                self.save_output(img, output_path, exif_data, verbose=verbose)

            return ProcessingResult(
                success=True,
                input_path=input_path,
                output_path=output_path,
                strategy_used=strategy_used,
                detections_found=detections_found,
                original_dimensions=original_dimensions,
                detections=detections_list,
                crop_box=crop_box,
                zoom_applied=zoom_applied
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                input_path=input_path,
                error_message=str(e)
            )

    def save_output(
        self,
        image: Image.Image,
        output_path: str,
        exif_data: Optional[bytes] = None,
        verbose: bool = False
    ) -> None:
        """
        Save processed image with EXIF preservation.

        Args:
            image: PIL Image to save
            output_path: Path to save to
            exif_data: EXIF data to preserve
            verbose: Print save details
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            ensure_directory(output_dir)

        # Save with EXIF if available
        save_kwargs = {
            'format': 'JPEG',
            'quality': self.quality,
            'optimize': True
        }

        if exif_data:
            save_kwargs['exif'] = exif_data

        image.save(output_path, **save_kwargs)

        if verbose:
            file_size = os.path.getsize(output_path) / 1024
            print(f"Saved to: {output_path} ({file_size:.1f} KB)")
