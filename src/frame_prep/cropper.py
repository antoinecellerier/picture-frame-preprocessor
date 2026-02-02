"""Intelligent cropping strategies for portrait conversion."""

from typing import List, Tuple, Optional
from PIL import Image
from .detector import Detection, ArtFeatureDetector
from .analyzer import CompositionAnalyzer


class SmartCropper:
    """Implements intelligent cropping strategies."""

    def __init__(
        self,
        target_width: int,
        target_height: int,
        zoom_factor: float = 1.3,
        use_saliency_fallback: bool = True
    ):
        """
        Initialize cropper with target dimensions.

        Args:
            target_width: Target width in pixels
            target_height: Target height in pixels
            zoom_factor: Zoom multiplier for tighter crops (default: 1.3)
            use_saliency_fallback: Use saliency when no detections (default: True)
        """
        self.target_width = target_width
        self.target_height = target_height
        self.target_aspect = target_height / target_width
        self.zoom_factor = zoom_factor
        self.use_saliency_fallback = use_saliency_fallback
        self.analyzer = CompositionAnalyzer()

        # Primary subject selector (uses center-weighting and class priorities)
        self._subject_selector = ArtFeatureDetector()

        # Store last crop info for reporting
        self.last_crop_box: Optional[Tuple[int, int, int, int]] = None
        self.last_zoom_applied: float = 1.0
        self.last_primary_detection: Optional[Detection] = None

    def crop_image(
        self,
        image: Image.Image,
        detections: Optional[List[Detection]] = None,
        strategy: str = 'smart'
    ) -> Image.Image:
        """
        Crop image using specified strategy.

        Args:
            image: PIL Image to crop
            detections: List of object detections (for smart strategy)
            strategy: Cropping strategy ('smart', 'saliency', or 'center')

        Returns:
            Cropped PIL Image
        """
        if strategy == 'smart':
            if detections:
                return self.crop_with_detections(image, detections)
            elif self.use_saliency_fallback:
                # No detections - use saliency instead of center crop
                return self.crop_saliency_based(image)
            else:
                return self.crop_center(image)
        elif strategy == 'saliency':
            return self.crop_saliency_based(image)
        else:
            return self.crop_center(image)

    def crop_with_detections(
        self,
        image: Image.Image,
        detections: List[Detection]
    ) -> Image.Image:
        """
        Crop using ML detections as anchor points with contextual smart zoom.

        Args:
            image: PIL Image
            detections: List of Detection objects

        Returns:
            Cropped image with contextual zoom applied
        """
        if not detections:
            return self.crop_center(image)

        width, height = image.size

        # Use primary subject selection (center-weighted, class-prioritized)
        # instead of just taking highest confidence detection
        self._subject_selector._last_image_size = (width, height)
        primary = self._subject_selector.get_primary_subject(detections)
        if primary is None:
            primary = detections[0]  # Fallback to highest confidence

        # Store for reporting
        self.last_primary_detection = primary
        anchor_x, anchor_y = primary.center

        # Calculate crop window
        crop_window = self._calculate_crop_window(
            image_size=(width, height),
            anchor_point=(anchor_x, anchor_y)
        )

        # Store crop box for reporting
        self.last_crop_box = crop_window

        # Crop to window
        cropped = image.crop(crop_window)

        # Calculate contextual zoom based on subject size relative to crop window
        crop_width = crop_window[2] - crop_window[0]
        crop_height = crop_window[3] - crop_window[1]
        subject_bbox = primary.bbox

        contextual_zoom = self._calculate_contextual_zoom(
            subject_bbox=subject_bbox,
            crop_width=crop_width,
            crop_height=crop_height
        )

        # Store zoom for reporting
        self.last_zoom_applied = contextual_zoom

        if contextual_zoom > 1.0:
            cropped = self._apply_smart_zoom(cropped, contextual_zoom)

        return cropped

    def crop_saliency_based(self, image: Image.Image) -> Image.Image:
        """
        Crop using saliency analysis with moderate zoom.

        Args:
            image: PIL Image

        Returns:
            Cropped image with moderate zoom for art focus
        """
        width, height = image.size
        anchor = self.analyzer.get_best_anchor(image)

        if anchor is None:
            return self.crop_center(image)

        anchor_x, anchor_y = anchor
        crop_window = self._calculate_crop_window(
            image_size=(width, height),
            anchor_point=(anchor_x, anchor_y)
        )

        # Store crop box for reporting
        self.last_crop_box = crop_window

        # Crop to window
        cropped = image.crop(crop_window)

        # Apply moderate zoom for art (can't calculate exact size with saliency)
        # Use a conservative zoom since we don't know subject boundaries
        moderate_zoom = min(1.2, self.zoom_factor)  # Cap at 1.2x for safety

        # Store zoom for reporting
        self.last_zoom_applied = moderate_zoom

        if moderate_zoom > 1.0:
            cropped = self._apply_smart_zoom(cropped, moderate_zoom)

        return cropped

    def crop_center(self, image: Image.Image) -> Image.Image:
        """
        Simple center crop.

        Args:
            image: PIL Image

        Returns:
            Cropped image
        """
        width, height = image.size
        anchor_x = width // 2
        anchor_y = height // 2

        crop_window = self._calculate_crop_window(
            image_size=(width, height),
            anchor_point=(anchor_x, anchor_y)
        )

        # Store crop box for reporting
        self.last_crop_box = crop_window
        self.last_zoom_applied = 1.0  # No zoom for center crop

        return image.crop(crop_window)

    def _calculate_crop_window(
        self,
        image_size: Tuple[int, int],
        anchor_point: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Calculate crop window centered on anchor point.
        Handles both landscape (crop width) and portrait (crop height) images.

        Args:
            image_size: (width, height) of original image
            anchor_point: (x, y) point to center crop on

        Returns:
            (left, top, right, bottom) crop box
        """
        width, height = image_size
        anchor_x, anchor_y = anchor_point

        current_aspect = height / width

        if current_aspect < self.target_aspect:
            # Image is too wide (landscape) - crop width, keep height
            crop_width = height / self.target_aspect
            crop_height = height

            # Center on anchor point horizontally
            left = anchor_x - crop_width / 2
            right = left + crop_width
            top = 0
            bottom = height

            # Clamp to image bounds horizontally
            if left < 0:
                left = 0
                right = crop_width
            if right > width:
                right = width
                left = width - crop_width

            left = max(0, left)
            right = min(width, right)

        else:
            # Image is too tall (portrait) - crop height, keep width
            crop_width = width
            crop_height = width * self.target_aspect

            # Center on anchor point vertically
            top = anchor_y - crop_height / 2
            bottom = top + crop_height
            left = 0
            right = width

            # Clamp to image bounds vertically
            if top < 0:
                top = 0
                bottom = crop_height
            if bottom > height:
                bottom = height
                top = height - crop_height

            top = max(0, top)
            bottom = min(height, bottom)

        return (int(left), int(top), int(right), int(bottom))

    def _calculate_contextual_zoom(
        self,
        subject_bbox: Tuple[int, int, int, int],
        crop_width: int,
        crop_height: int
    ) -> float:
        """
        Calculate zoom factor based on subject dimensions relative to crop window.

        Uses the subject's width and height ratios to determine zoom:
        - If subject fills most of the frame in either dimension, minimize zoom
        - If subject is small in both dimensions, zoom in to fill ~70% of frame
        - Uses the LARGER dimension ratio to avoid over-zooming tall/wide subjects

        Args:
            subject_bbox: (x1, y1, x2, y2) bounding box of the subject
            crop_width: Width of the crop window
            crop_height: Height of the crop window

        Returns:
            Contextual zoom factor (1.0 = no zoom)
        """
        x1, y1, x2, y2 = subject_bbox
        subject_width = x2 - x1
        subject_height = y2 - y1

        # Calculate how much of each dimension the subject fills
        width_ratio = subject_width / crop_width if crop_width > 0 else 0
        height_ratio = subject_height / crop_height if crop_height > 0 else 0

        # Use the larger ratio - if subject fills height, don't zoom even if narrow
        max_ratio = max(width_ratio, height_ratio)

        # Target: subject should fill ~70% of the frame's larger dimension
        target_ratio = 0.70

        if max_ratio >= 0.65:
            # Subject already fills most of the frame, no zoom needed
            return 1.0
        elif max_ratio >= 0.45:
            # Subject is medium-sized, slight zoom
            zoom_needed = target_ratio / max_ratio
            return min(zoom_needed, 1.2, self.zoom_factor)
        elif max_ratio >= 0.25:
            # Subject is small, moderate zoom
            zoom_needed = target_ratio / max_ratio
            return min(zoom_needed, self.zoom_factor)
        else:
            # Subject is tiny, zoom more aggressively (but cap at max)
            zoom_needed = target_ratio / max(max_ratio, 0.05)
            return min(zoom_needed, self.zoom_factor)

    def _apply_smart_zoom(
        self,
        image: Image.Image,
        zoom_factor: float
    ) -> Image.Image:
        """
        Apply centered zoom to focus tighter on subject.

        Args:
            image: PIL Image to zoom
            zoom_factor: Zoom multiplier (e.g., 1.3 = 30% zoom in)

        Returns:
            Zoomed and cropped image
        """
        width, height = image.size

        # Calculate zoomed dimensions (smaller crop area)
        zoom_width = int(width / zoom_factor)
        zoom_height = int(height / zoom_factor)

        # Center the zoom
        left = (width - zoom_width) // 2
        top = (height - zoom_height) // 2
        right = left + zoom_width
        bottom = top + zoom_height

        # Crop to zoomed area
        zoomed = image.crop((left, top, right, bottom))

        # Resize back to original dimensions for consistent output
        return zoomed.resize((width, height), Image.LANCZOS)

    def needs_cropping(self, image: Image.Image) -> bool:
        """
        Check if image needs cropping to reach target aspect ratio.

        Args:
            image: PIL Image

        Returns:
            True if image aspect ratio doesn't match target (within tolerance)
        """
        width, height = image.size
        current_aspect = height / width
        # Crop if aspect ratio differs from target (tolerance: 1%)
        return abs(current_aspect - self.target_aspect) > 0.01
