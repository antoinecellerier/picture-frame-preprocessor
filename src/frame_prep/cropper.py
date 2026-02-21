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
        zoom_factor: float = 8.0,
        use_saliency_fallback: bool = True
    ):
        """
        Initialize cropper with target dimensions.

        Args:
            target_width: Target width in pixels
            target_height: Target height in pixels
            zoom_factor: Max zoom multiplier for tighter crops (default: 8.0, very aggressive for tiny subjects)
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
        self.last_primary_fills_frame: bool = False

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
        self.last_primary_fills_frame = False
        anchor_x, anchor_y = primary.center
        zoom_subject = primary

        # When the primary fills the frame (contextual zoom would be 1.0),
        # it already occupies the full target screen and can't be meaningfully
        # zoomed into. In that case, use the best inner detection as a focal
        # point to zoom into a specific sub-region of the large subject.
        test_crop = self._calculate_crop_window(
            image_size=(width, height),
            anchor_point=(anchor_x, anchor_y)
        )
        test_cw = test_crop[2] - test_crop[0]
        test_ch = test_crop[3] - test_crop[1]
        if self._calculate_contextual_zoom(primary.bbox, test_cw, test_ch) <= 1.0:
            self.last_primary_fills_frame = True
            inner_dets = self._get_quality_inner_detections(primary, detections, (width, height))
            if inner_dets:
                # Shift anchor to the focal point but keep zoom against the primary —
                # we want to frame as much of the piece as possible, just better centred.
                anchor_x, anchor_y = inner_dets[0].center
                # Ensure the crop window still fully contains the primary bbox
                anchor_x, anchor_y = self._clamp_anchor_to_primary(
                    anchor_x, anchor_y, primary.bbox, (width, height)
                )

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
        subject_bbox = zoom_subject.bbox

        contextual_zoom = self._calculate_contextual_zoom(
            subject_bbox=subject_bbox,
            crop_width=crop_width,
            crop_height=crop_height
        )

        # Store zoom for reporting
        self.last_zoom_applied = contextual_zoom

        if contextual_zoom > 1.0:
            # Subject center relative to crop window
            subject_cx = anchor_x - crop_window[0]
            subject_cy = anchor_y - crop_window[1]
            cropped = self._apply_smart_zoom(
                cropped, contextual_zoom,
                center=(subject_cx, subject_cy)
            )

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

    def _crop_width_for(self, img_width: int, img_height: int) -> float:
        """Return the crop window width for a given image size."""
        current_aspect = img_height / img_width
        if current_aspect < self.target_aspect:
            return img_height / self.target_aspect
        return float(img_width)

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
        zoom_factor: float,
        center: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        Apply zoom centered on subject position.

        Args:
            image: PIL Image to zoom
            zoom_factor: Zoom multiplier (e.g., 1.3 = 30% zoom in)
            center: (x, y) point to center zoom on. Defaults to image center.

        Returns:
            Zoomed and cropped image
        """
        width, height = image.size

        # Calculate zoomed dimensions (smaller crop area)
        zoom_width = int(width / zoom_factor)
        zoom_height = int(height / zoom_factor)

        # Center zoom on subject position (or image center as fallback)
        cx = center[0] if center else width // 2
        cy = center[1] if center else height // 2

        left = cx - zoom_width // 2
        top = cy - zoom_height // 2
        right = left + zoom_width
        bottom = top + zoom_height

        # Clamp to image bounds
        if left < 0:
            left = 0
            right = zoom_width
        if top < 0:
            top = 0
            bottom = zoom_height
        if right > width:
            right = width
            left = width - zoom_width
        if bottom > height:
            bottom = height
            top = height - zoom_height

        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)

        # Crop to zoomed area
        zoomed = image.crop((left, top, right, bottom))

        # Resize back to original dimensions for consistent output
        return zoomed.resize((width, height), Image.LANCZOS)

    # Secondary crops must clear this confidence bar to avoid false positives
    MULTI_CROP_SECONDARY_CONFIDENCE = 0.30

    def crop_all_subjects(
        self,
        image: Image.Image,
        detections: List[Detection]
    ) -> List[Tuple[Image.Image, Detection, float]]:
        """
        Crop each viable art subject independently for multi-crop output.

        The primary subject (center-weighted scoring) is always first.  If it
        is wider than a single crop window, it is split into multiple crops
        along its width.  Additional art-class detections that don't overlap
        existing crops are appended, provided they clear the secondary
        confidence threshold.

        Args:
            image: PIL Image
            detections: List of Detection objects

        Returns:
            List of (cropped_image, detection, zoom_applied) — primary first,
            then remaining subjects sorted left-to-right
        """
        if not detections:
            return []

        width, height = image.size

        # Identify primary subject using center-weighted scoring
        self._subject_selector._last_image_size = (width, height)
        primary = self._subject_selector.get_primary_subject(detections)

        # Filter to viable art detections (class_multiplier >= 1.5)
        viable = [
            d for d in detections
            if ArtFeatureDetector._get_class_multiplier(d.class_name) >= 1.5
        ]

        if not viable and primary is None:
            return []

        # Ensure primary is in viable list even if its class_multiplier < 1.5
        if primary is not None and primary not in viable:
            viable.insert(0, primary)

        # Build candidate list: primary first, then others by confidence
        candidates: List[Tuple[Detection, Tuple[int, int, int, int]]] = []

        # --- Primary subject (possibly split if wider than crop window) ---
        if primary is not None:
            crop_w = self._crop_width_for(width, height)
            bx1, by1, bx2, by2 = primary.bbox
            subject_width = bx2 - bx1

            # Trigger focal-point logic when:
            # (a) primary is physically wider than a single crop window, OR
            # (b) primary fills the frame (zoom would be 1.0 — it doesn't fit
            #     on the target screen without showing it at reduced scale)
            test_crop = self._calculate_crop_window((width, height), primary.center)
            test_cw = test_crop[2] - test_crop[0]
            test_ch = test_crop[3] - test_crop[1]
            primary_fills_frame = (
                self._calculate_contextual_zoom(primary.bbox, test_cw, test_ch) <= 1.0
            )

            if subject_width > crop_w * 1.3 or primary_fills_frame:
                # Primary is too large for a single crop to zoom into.
                # Use inner detections as natural focal points.
                inner_dets = self._get_quality_inner_detections(
                    primary, detections, (width, height)
                )

                if inner_dets:
                    # Use inner detections as anchor points, clamped so the
                    # crop window still fully contains the primary bbox
                    for det in inner_dets:
                        ax, ay = self._clamp_anchor_to_primary(
                            det.center[0], det.center[1], primary.bbox, (width, height)
                        )
                        cw = self._calculate_crop_window(
                            image_size=(width, height),
                            anchor_point=(ax, ay)
                        )
                        overlaps = any(
                            self._calculate_iou(cw, ecw) > 0.3
                            for _, ecw in candidates
                        )
                        if not overlaps:
                            candidates.append((det, cw))

                # Always include a primary-centered crop (first position)
                # if no inner detections produced candidates
                if not candidates:
                    cw = self._calculate_crop_window(
                        image_size=(width, height),
                        anchor_point=primary.center
                    )
                    candidates.append((primary, cw))
            else:
                cw = self._calculate_crop_window(
                    image_size=(width, height),
                    anchor_point=primary.center
                )
                candidates.append((primary, cw))

        # --- Remaining viable detections outside primary ---
        # Secondary crops are filtered more aggressively to avoid junk:
        #  - Higher class multiplier bar (>= 2.0 vs 1.5 for primary)
        #  - Must clear secondary confidence threshold
        #  - Skip detections touching image edges (partially out of frame)
        #  - Skip very small detections (< 1.5% of image area)
        img_area = width * height
        edge_margin = 0.01  # 1% of dimension

        remaining = sorted(
            [d for d in viable if d is not primary],
            key=lambda d: d.confidence, reverse=True
        )
        for det in remaining:
            if det.confidence < self.MULTI_CROP_SECONDARY_CONFIDENCE:
                continue

            # Require stronger art-class signal for secondaries
            if ArtFeatureDetector._get_class_multiplier(det.class_name) < 2.0:
                continue

            # Skip detections that touch image edges (likely partial/cut-off)
            bx1, by1, bx2, by2 = det.bbox
            if (bx1 < width * edge_margin or
                by1 < height * edge_margin or
                bx2 > width * (1 - edge_margin) or
                by2 > height * (1 - edge_margin)):
                continue

            # Skip tiny detections (likely noise)
            det_area = (bx2 - bx1) * (by2 - by1)
            if det_area < img_area * 0.015:
                continue

            cw = self._calculate_crop_window(
                image_size=(width, height),
                anchor_point=det.center
            )
            overlaps = any(
                self._calculate_iou(cw, ecw) > 0.3
                for _, ecw in candidates
            )
            if not overlaps:
                candidates.append((det, cw))

        # Primary-anchored crops stay first; remaining sorted left-to-right
        primary_bbox = primary.bbox if primary else None
        def _is_primary_crop(pair):
            det, _ = pair
            if det is primary:
                return True
            if primary_bbox and self._bbox_overlap_ratio(det.bbox, primary_bbox) > 0.5:
                return True
            return False

        primary_candidates = [c for c in candidates if _is_primary_crop(c)]
        other_candidates = [c for c in candidates if not _is_primary_crop(c)]
        # Sort primary sub-crops left-to-right, others left-to-right
        primary_candidates.sort(key=lambda pair: pair[0].center[0])
        other_candidates.sort(key=lambda pair: pair[0].center[0])
        candidates = primary_candidates + other_candidates

        # Crop each detection independently
        results: List[Tuple[Image.Image, Detection, float]] = []
        for det, crop_window in candidates:
            cropped = image.crop(crop_window)

            crop_width = crop_window[2] - crop_window[0]
            crop_height = crop_window[3] - crop_window[1]

            contextual_zoom = self._calculate_contextual_zoom(
                subject_bbox=det.bbox,
                crop_width=crop_width,
                crop_height=crop_height
            )

            if contextual_zoom > 1.0:
                anchor_x, anchor_y = det.center
                subject_cx = anchor_x - crop_window[0]
                subject_cy = anchor_y - crop_window[1]
                cropped = self._apply_smart_zoom(
                    cropped, contextual_zoom,
                    center=(subject_cx, subject_cy)
                )

            results.append((cropped, det, contextual_zoom))

        return results

    def _clamp_anchor_to_primary(
        self,
        anchor_x: int,
        anchor_y: int,
        primary_bbox: Tuple[int, int, int, int],
        image_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Clamp anchor so the crop window fully contains the primary bbox.

        Uses a reference crop at the image centre to determine crop dimensions
        (crop width/height are aspect-ratio driven and don't depend on anchor).
        The primary bbox is clipped to image bounds before clamping so that
        OOB detections don't artificially expand the constraint range.
        """
        width, height = image_size
        px1, py1, px2, py2 = primary_bbox
        # Clip primary to image bounds (handles OOB model extrapolations)
        px1 = max(0, px1); py1 = max(0, py1)
        px2 = min(width, px2); py2 = min(height, py2)

        # Get crop dimensions using a neutral centre anchor
        ref = self._calculate_crop_window((width, height), (width // 2, height // 2))
        cw = ref[2] - ref[0]
        ch = ref[3] - ref[1]

        # Unified clamp for both axes:
        #   When primary fits in crop  → keeps crop containing primary
        #   When primary wider/taller → keeps crop within primary
        # Formula: clamp to [min(px1+cw/2, px2-cw/2), max(px1+cw/2, px2-cw/2)]
        ax_lo = min(px1 + cw / 2, px2 - cw / 2)
        ax_hi = max(px1 + cw / 2, px2 - cw / 2)
        anchor_x = int(max(ax_lo, min(ax_hi, anchor_x)))

        ay_lo = min(py1 + ch / 2, py2 - ch / 2)
        ay_hi = max(py1 + ch / 2, py2 - ch / 2)
        anchor_y = int(max(ay_lo, min(ay_hi, anchor_y)))

        return anchor_x, anchor_y

    def _get_quality_inner_detections(
        self,
        primary: Detection,
        detections: List[Detection],
        image_size: tuple
    ) -> List[Detection]:
        """
        Find quality detections that fall inside the primary bbox.

        Applies edge and size filters identical to secondary crop filters.
        Returns list sorted by class_multiplier * confidence (best first).
        """
        width, height = image_size
        img_area = width * height
        edge_margin = 0.01
        inner_dets = []
        for d in detections:
            if d is primary:
                continue
            if self._bbox_overlap_ratio(d.bbox, primary.bbox) <= 0.5:
                continue
            if d.confidence < self.MULTI_CROP_SECONDARY_CONFIDENCE:
                continue
            dx1, dy1, dx2, dy2 = d.bbox
            if (dx1 < width * edge_margin or
                dy1 < height * edge_margin or
                dx2 > width * (1 - edge_margin) or
                dy2 > height * (1 - edge_margin)):
                continue
            det_area = (dx2 - dx1) * (dy2 - dy1)
            if det_area < img_area * 0.015:
                continue
            inner_dets.append(d)
        inner_dets.sort(
            key=lambda d: ArtFeatureDetector._get_class_multiplier(d.class_name) * d.confidence,
            reverse=True
        )
        return inner_dets

    @staticmethod
    def _calculate_iou(
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate IoU between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _bbox_overlap_ratio(
        inner: Tuple[int, int, int, int],
        outer: Tuple[int, int, int, int]
    ) -> float:
        """Fraction of inner's area that overlaps with outer."""
        x1 = max(inner[0], outer[0])
        y1 = max(inner[1], outer[1])
        x2 = min(inner[2], outer[2])
        y2 = min(inner[3], outer[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
        return intersection / inner_area if inner_area > 0 else 0.0

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
