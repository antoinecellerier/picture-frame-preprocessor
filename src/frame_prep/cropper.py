"""Intelligent cropping strategies for portrait conversion."""

from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
from .detector import Detection, ArtFeatureDetector, calculate_iou
from .analyzer import CompositionAnalyzer, TextDetector, TEXT_RATIO_THRESHOLD


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
        self._text_detector = TextDetector()

        # Primary subject selector (uses center-weighting and class priorities)
        self._subject_selector = ArtFeatureDetector()

        # Store last crop info for reporting
        self.last_crop_box: Optional[Tuple[int, int, int, int]] = None
        self.last_zoom_applied: float = 1.0
        self.last_primary_detection: Optional[Detection] = None
        self.last_primary_fills_frame: bool = False
        self.last_inner_detection: Optional[Detection] = None  # Selected focal anchor

    def crop_image(
        self,
        image: Image.Image,
        detections: Optional[List[Detection]] = None,
        strategy: str = 'smart',
        focal_detections: Optional[List[Detection]] = None
    ) -> Image.Image:
        """
        Crop image using specified strategy.

        Args:
            image: PIL Image to crop
            detections: List of object detections (for smart strategy)
            strategy: Cropping strategy ('smart', 'saliency', or 'center')
            focal_detections: Focal-pass detections (used only for inner anchor
                selection, never for primary subject selection)

        Returns:
            Cropped PIL Image
        """
        if strategy == 'smart':
            if detections:
                return self.crop_with_detections(image, detections,
                                                 focal_detections=focal_detections)
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
        detections: List[Detection],
        focal_detections: Optional[List[Detection]] = None
    ) -> Image.Image:
        """
        Crop using ML detections as anchor points with contextual smart zoom.

        Args:
            image: PIL Image
            detections: List of Detection objects (used for primary selection)
            focal_detections: Focal-pass detections (used only for inner anchor
                selection, never for primary subject selection)

        Returns:
            Cropped image with contextual zoom applied
        """
        if not detections:
            return self.crop_center(image)

        width, height = image.size

        # Use primary subject selection (center-weighted, class-prioritized)
        # instead of just taking highest confidence detection.
        # NOTE: focal_detections are NOT included here — focal class names
        # (e.g. "human figure") can inherit wrong art-class multipliers and
        # would corrupt primary selection.
        self._subject_selector.set_image_size(width, height)
        primary = self._subject_selector.get_primary_subject(detections)
        if primary is None:
            primary = detections[0]  # Fallback to highest confidence

        # Store for reporting
        self.last_primary_detection = primary
        self.last_primary_fills_frame = False
        self.last_inner_detection = None
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
            # 3D art (sculpture, statue, etc.) is its own focal point — don't
            # shift the anchor to background elements like framed artwork.
            if not ArtFeatureDetector.is_3d_art(primary.class_name):
                # Combine regular detections with focal detections for inner anchor
                # selection — focal dets are safe here since we're already past
                # primary selection and only using them as candidate crop anchors.
                inner_candidates = detections + (focal_detections or [])
                inner_dets = self._get_quality_inner_detections(primary, inner_candidates, (width, height), focal_set=focal_detections)
            else:
                inner_dets = []
            if inner_dets:
                # Shift anchor to the focal point but keep zoom against the primary —
                # we want to frame as much of the piece as possible, just better centred.
                self.last_inner_detection = inner_dets[0]
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

    def _effective_visible_region(
        self,
        crop_window: Tuple[int, int, int, int],
        subject_bbox: Tuple[int, int, int, int],
        subject_center: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Compute the image region actually visible after contextual zoom.

        When the crop window is much larger than the subject, the zoom step
        narrows the visible area.  Using this for overlap checks (instead of
        the raw crop window) allows multiple crops from different parts of the
        same base crop window.
        """
        cw_left, cw_top, cw_right, cw_bottom = crop_window
        cw_width = cw_right - cw_left
        cw_height = cw_bottom - cw_top

        zoom = self._calculate_contextual_zoom(subject_bbox, cw_width, cw_height)

        if zoom <= 1.0:
            return crop_window

        # Viewport size after zoom
        vw = cw_width / zoom
        vh = cw_height / zoom

        # Center on subject within crop window
        cx = subject_center[0] - cw_left
        cy = subject_center[1] - cw_top

        # Viewport bounds in crop coordinates
        vx1 = cx - vw / 2
        vy1 = cy - vh / 2
        vx2 = vx1 + vw
        vy2 = vy1 + vh

        # Clamp to crop window bounds
        if vx1 < 0:
            vx1, vx2 = 0, vw
        if vy1 < 0:
            vy1, vy2 = 0, vh
        if vx2 > cw_width:
            vx1, vx2 = cw_width - vw, cw_width
        if vy2 > cw_height:
            vy1, vy2 = cw_height - vh, cw_height

        # Convert back to image coordinates
        return (
            int(cw_left + max(0, vx1)),
            int(cw_top + max(0, vy1)),
            int(cw_left + min(cw_width, vx2)),
            int(cw_top + min(cw_height, vy2))
        )

    # Secondary crops must clear this confidence bar to avoid false positives
    MULTI_CROP_SECONDARY_CONFIDENCE = 0.30
    # Inner focal anchors can be less confident — they're within a detected primary
    FOCAL_INNER_CONFIDENCE = 0.25
    # Ensemble-pass detections reused as inner focal anchors need higher
    # confidence — their class labels (mosaic, mural, etc.) can match non-art
    # regions like placards or peripheral decorations within a large primary.
    ENSEMBLE_INNER_CONFIDENCE = 0.35

    def crop_all_subjects(
        self,
        image: Image.Image,
        detections: List[Detection],
        focal_detections: Optional[List[Detection]] = None
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
        self._subject_selector.set_image_size(width, height)
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

        # Build candidate list: primary first, then others by confidence.
        # Track the effective visible region (post-zoom viewport) for each
        # candidate so the overlap check compares what the viewer actually
        # sees, not the raw (often identical) pre-zoom crop windows.
        candidates: List[Tuple[Detection, Tuple[int, int, int, int]]] = []
        effective_regions: List[Tuple[int, int, int, int]] = []

        # --- Primary subject (possibly split if wider than crop window) ---
        if primary is not None:
            crop_w = self._crop_width_for(width, height)
            bx1, by1, bx2, by2 = primary.bbox
            subject_width = bx2 - bx1

            cw = self._calculate_crop_window(
                image_size=(width, height),
                anchor_point=primary.center
            )
            candidates.append((primary, cw))
            effective_regions.append(
                self._effective_visible_region(
                    cw, primary.bbox, primary.center))

            # --- Inner focal crops for wide primaries ---
            # When the primary fills the frame AND extends beyond the crop
            # window, produce panning crops centred on inner detections.
            # The extends-beyond check (>= 1.25× in one dimension) avoids
            # producing redundant zoom-detail crops for contained artwork
            # that already fits entirely within the crop.
            cw_w = cw[2] - cw[0]
            cw_h = cw[3] - cw[1]
            primary_w = bx2 - bx1
            primary_h = by2 - by1
            extends_beyond = (primary_w >= cw_w * 1.25 or
                              primary_h >= cw_h * 1.25)
            if (self._calculate_contextual_zoom(primary.bbox, cw_w, cw_h) <= 1.0
                    and extends_beyond
                    and not ArtFeatureDetector.is_3d_art(primary.class_name)):
                inner_candidates_all = detections + (focal_detections or [])
                inner_dets = self._get_quality_inner_detections(
                    primary, inner_candidates_all, (width, height),
                    focal_set=focal_detections)

                primary_diag = (primary_w ** 2 + primary_h ** 2) ** 0.5
                min_center_dist = primary_diag * 0.08

                accepted_inner: List[Tuple[Detection, Tuple[int, int, int, int]]] = []
                for d in inner_dets:
                    if len(accepted_inner) >= 4:
                        break
                    inner_cw = self._calculate_crop_window(
                        image_size=(width, height), anchor_point=d.center)
                    icw_w = inner_cw[2] - inner_cw[0]
                    icw_h = inner_cw[3] - inner_cw[1]
                    zoom = self._calculate_contextual_zoom(d.bbox, icw_w, icw_h)
                    if zoom <= 1.3:
                        continue
                    # Viewport must be large enough to be worth showing
                    if min(icw_w / zoom, icw_h / zoom) < 300:
                        continue
                    # Must be far enough from already-accepted inner crops
                    cx, cy = d.center
                    too_close = any(
                        ((cx - ad.center[0]) ** 2 + (cy - ad.center[1]) ** 2) ** 0.5
                        < min_center_dist
                        for ad, _ in accepted_inner
                    )
                    if too_close:
                        continue
                    # Skip text-heavy focal targets (signs, labels, placards)
                    if self._text_detector.center_weighted_text_ratio(image, d.bbox) > TEXT_RATIO_THRESHOLD:
                        continue
                    accepted_inner.append((d, inner_cw))

                # Spread check: a single inner crop always provides value
                # (zoomed view vs full-frame primary). For 2+ inner crops,
                # require they span >= 40% of the primary to avoid clustered
                # zoom duplicates of the same section.
                keep_inner = False
                if len(accepted_inner) == 1:
                    keep_inner = True
                elif len(accepted_inner) >= 2:
                    xs = [d.center[0] for d, _ in accepted_inner]
                    ys = [d.center[1] for d, _ in accepted_inner]
                    x_spread = (max(xs) - min(xs)) / max(primary_w, 1)
                    y_spread = (max(ys) - min(ys)) / max(primary_h, 1)
                    keep_inner = x_spread >= 0.4 or y_spread >= 0.4

                if keep_inner:
                    for d, icw in accepted_inner:
                        eff = self._effective_visible_region(
                            icw, d.bbox, d.center)
                        candidates.append((d, icw))
                        effective_regions.append(eff)

        # --- Remaining viable detections outside primary ---
        # Secondary crops are filtered more aggressively to avoid junk:
        #  - Higher class multiplier bar (>= 2.0 vs 1.5 for primary)
        #  - Must clear secondary confidence threshold
        #  - Skip detections touching image edges (partially out of frame)
        #  - Skip very small detections (< 1.5% of image area)
        img_area = width * height
        edge_margin = 0.01  # 1% of dimension
        img_brightness = np.array(image.convert('L')).mean()

        remaining = sorted(
            [d for d in viable if d is not primary],
            key=lambda d: d.confidence, reverse=True
        )
        for det in remaining:
            if det.confidence < self.MULTI_CROP_SECONDARY_CONFIDENCE:
                continue

            bx1, by1, bx2, by2 = det.bbox
            det_area = (bx2 - bx1) * (by2 - by1)

            # Filter weak small secondaries: low confidence on a small region
            # is almost always noise (signs, stickers, architectural details)
            if det.confidence < 0.35 and det_area < img_area * 0.05:
                continue

            # Require stronger art-class signal for secondaries.
            # Large detections (>5% of image) get a relaxed bar — they're
            # substantial enough that even default-class names (e.g. "figurine")
            # are likely real art subjects, not noise.
            mult = ArtFeatureDetector._get_class_multiplier(det.class_name)
            min_mult = 1.0 if det_area >= img_area * 0.05 else 2.0
            if mult < min_mult:
                continue

            # Skip detections whose content is already visible in the
            # primary crop — either the detection bbox is inside the
            # primary bbox (depicted element within an artwork), or the
            # zoomed viewport is inside the primary's viewport.
            if primary is not None and effective_regions:
                if self._bbox_overlap_ratio(det.bbox, primary.bbox) > 0.7:
                    continue
                det_cw = self._calculate_crop_window(
                    image_size=(width, height), anchor_point=det.center)
                det_eff = self._effective_visible_region(
                    det_cw, det.bbox, det.center)
                if self._bbox_overlap_ratio(det_eff, effective_regions[0]) > 0.85:
                    continue

            # Skip detections that touch image edges (likely partial/cut-off)
            if (bx1 < width * edge_margin or
                by1 < height * edge_margin or
                bx2 > width * (1 - edge_margin) or
                by2 > height * (1 - edge_margin)):
                continue

            # Skip tiny detections (likely noise)
            if det_area < img_area * 0.015:
                continue

            # Skip regions much darker than the image (shadow/underexposure,
            # not visible art). Uses ratio to handle night/dim photos where
            # the whole image is dark but the art is still valid.
            region_brightness = np.array(image.crop((bx1, by1, bx2, by2)).convert('L')).mean()
            if region_brightness < img_brightness * 0.35:
                continue

            # Skip text-heavy regions (signs, labels, exhibit info panels)
            if self._text_detector.text_ratio(image, det.bbox) > TEXT_RATIO_THRESHOLD:
                continue

            cw = self._calculate_crop_window(
                image_size=(width, height),
                anchor_point=det.center
            )
            eff = self._effective_visible_region(cw, det.bbox, det.center)
            # Two-tier overlap check:
            # 1. Unconditional: block if effective regions are nearly
            #    identical (same content regardless of bbox position)
            # 2. Conditional: block zoom variants of the same subject
            #    (one eff region mostly inside the other, AND detection
            #    bboxes are spatially related — one contains the other
            #    or they overlap)
            overlaps = any(
                calculate_iou(eff, er) > 0.6
                or (max(self._bbox_overlap_ratio(eff, er),
                        self._bbox_overlap_ratio(er, eff)) > 0.7
                    and max(self._bbox_overlap_ratio(det.bbox, edet.bbox),
                            self._bbox_overlap_ratio(edet.bbox, det.bbox)) > 0.3)
                for (edet, _), er in zip(candidates, effective_regions)
            )
            if not overlaps:
                candidates.append((det, cw))
                effective_regions.append(eff)

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
        # Primary-centered crop first, then focal sub-crops left-to-right
        primary_candidates.sort(
            key=lambda pair: (0 if pair[0] is primary else 1, pair[0].center[0]))
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
        image_size: tuple,
        focal_set: Optional[List[Detection]] = None
    ) -> List[Detection]:
        """
        Find quality detections that fall inside the primary bbox.

        Filters:
        - Must overlap > 50% with primary bbox
        - Confidence >= FOCAL_INNER_CONFIDENCE for focal-pass detections,
          >= ENSEMBLE_INNER_CONFIDENCE for ensemble-pass detections
        - Must not touch image edges (partial / cut-off detections)
        - Area >= 1% of full image (avoids tiny noise / extreme zoom)

        Sorted by: confidence * parabolic_area_factor
        where parabolic_area_factor = 4 * area_ratio * (1 - area_ratio).
        This peaks at 50% of the primary area and naturally penalises both
        tiny noise detections AND near-full-primary re-detections — no hard
        cap needed.  Class multiplier is intentionally excluded: focal prompt
        labels ("human figure", "portrait", etc.) should not inherit the
        art-class scoring system designed for primary subject selection.
        """
        width, height = image_size
        img_area = width * height
        edge_margin = 0.01
        focal_ids = set(id(d) for d in (focal_set or []))

        px1, py1, px2, py2 = primary.bbox
        primary_area = max(1, (px2 - px1) * (py2 - py1))

        inner_dets = []
        for d in detections:
            if d is primary:
                continue
            if self._bbox_overlap_ratio(d.bbox, primary.bbox) <= 0.5:
                continue
            # Focal-pass detections (face/figure prompts) get a lower bar;
            # ensemble-pass detections (mosaic/mural/etc.) need higher
            # confidence to avoid non-art inner regions (signs, placards).
            min_conf = (self.FOCAL_INNER_CONFIDENCE if id(d) in focal_ids
                        else self.ENSEMBLE_INNER_CONFIDENCE)
            if d.confidence < min_conf:
                continue
            dx1, dy1, dx2, dy2 = d.bbox
            if (dx1 < width * edge_margin or
                dy1 < height * edge_margin or
                dx2 > width * (1 - edge_margin) or
                dy2 > height * (1 - edge_margin)):
                continue
            det_area = (dx2 - dx1) * (dy2 - dy1)
            if det_area < img_area * 0.01:  # 1%: filter tiny noise
                continue
            inner_dets.append(d)

        def _inner_score(d):
            dx1, dy1, dx2, dy2 = d.bbox
            area = (dx2 - dx1) * (dy2 - dy1)
            area_ratio = area / primary_area
            # Parabolic factor peaks at area_ratio=0.5, penalises both tiny and
            # full-primary-covering detections without a hard cutoff.
            area_factor = 4.0 * area_ratio * (1.0 - area_ratio)
            return d.confidence * area_factor

        inner_dets.sort(key=_inner_score, reverse=True)
        return inner_dets

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

    def primary_fills_frame(
        self,
        primary_bbox: Tuple[int, int, int, int],
        image_size: Tuple[int, int]
    ) -> bool:
        """
        Return True when the primary detection already fills the crop frame.

        Mirrors the condition used in crop_with_detections: primary fills the
        frame when _calculate_contextual_zoom would return <= 1.0, meaning the
        subject already occupies the full target screen width or height.
        """
        width, height = image_size
        cx = (primary_bbox[0] + primary_bbox[2]) // 2
        cy = (primary_bbox[1] + primary_bbox[3]) // 2
        test_crop = self._calculate_crop_window((width, height), (cx, cy))
        cw = test_crop[2] - test_crop[0]
        ch = test_crop[3] - test_crop[1]
        return self._calculate_contextual_zoom(primary_bbox, cw, ch) <= 1.0

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
