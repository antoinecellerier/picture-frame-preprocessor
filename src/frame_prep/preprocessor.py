"""Core pipeline orchestration for image preprocessing."""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS

from .detector import ArtFeatureDetector
from .cropper import SmartCropper
from .utils import validate_image, ensure_directory


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


class ImagePreprocessor:
    """Main image preprocessing pipeline."""

    def __init__(
        self,
        target_width: int,
        target_height: int,
        detector: Optional[ArtFeatureDetector] = None,
        cropper: Optional[SmartCropper] = None,
        strategy: str = 'smart',
        quality: int = 95
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
        """
        self.target_width = target_width
        self.target_height = target_height
        self.strategy = strategy
        self.quality = quality

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

                # Check if cropping is needed
                if self.cropper.needs_cropping(img):
                    if verbose:
                        print(f"Image is landscape, applying {self.strategy} cropping...")

                    # Run detection if using smart strategy
                    detections = []
                    if self.strategy == 'smart':
                        detections = self.detector.detect(img, verbose=verbose)

                        # Capture detection details for ML analysis
                        primary = self.detector.get_primary_subject(detections) if detections else None
                        for det in detections:
                            detections_list.append({
                                'bbox': det.bbox,
                                'confidence': det.confidence,
                                'class_name': det.class_name,
                                'is_primary': det == primary
                            })

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
