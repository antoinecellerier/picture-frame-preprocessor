"""OpenCV-based tile/mosaic pattern detector.

Uses Hough Line detection + pairwise spacing voting to find mosaic/tile grids.
Purely additive — only emits a synthetic Detection when a tile grid is found;
never removes or displaces existing detections from the main pipeline.
"""

from typing import Optional, Tuple
import numpy as np
from PIL import Image

from frame_prep.detector import Detection


class TilePatternDetector:
    """
    Detect mosaic/tile grid patterns using Hough line detection.

    Algorithm:
    1. Canny edge detection
    2. Hough probabilistic line transform to find long straight lines
    3. Separate into horizontal and vertical line groups
    4. Score regularity via pairwise spacing voting:
       - Compute all pairwise spacings between detected line positions
       - Build a histogram; a mosaic produces a sharp peak at the tile spacing T
       - Score = (votes near histogram mode) / (total valid votes) × coverage
    5. If both axes score above threshold, emit a synthetic 'mosaic' Detection

    Using Hough Lines (rather than autocorrelation) avoids false positives from
    JPEG DCT compression artifacts, which produce pervasive 8-pixel periodicity
    but do NOT produce long straight parallel lines.

    Using pairwise voting (rather than CV of consecutive spacings) handles:
    - Multiple Hough detections per grout line: intra-cluster pairs have small
      spacings (< min_tile_px) and are filtered out before voting
    - Missed grout lines: pairs at 2T, 3T etc. still support the same period T
      but don't corrupt the mode — they vote in separate histogram bins
    """

    def __init__(
        self,
        min_tile_px: int = 20,
        max_tile_px: int = 400,
        threshold: float = 1.0,  # Effectively disabled — Hough Lines approach has too many false positives
        min_line_count: int = 3,
        min_coverage: float = 0.03,
    ):
        """
        Args:
            min_tile_px: Minimum expected tile spacing in pixels.
            max_tile_px: Maximum expected tile spacing in pixels.
            threshold: Minimum combined H×V score to fire (pairwise vote ratio²).
            min_line_count: Minimum number of parallel lines in each direction.
            min_coverage: Minimum fraction of image the detected region must cover.
        """
        self.min_tile_px = min_tile_px
        self.max_tile_px = max_tile_px
        self.threshold = threshold
        self.min_line_count = min_line_count
        self.min_coverage = min_coverage

    def detect(self, image: Image.Image, verbose: bool = False) -> Optional[Detection]:
        """
        Run tile pattern detection on a PIL image.

        Returns a synthetic Detection(class_name='mosaic') if a regular
        grid of parallel lines is found, else None.
        """
        try:
            import cv2
        except ImportError:
            if verbose:
                print("  Tile detector: cv2 not available, skipping")
            return None

        img_np = np.array(image.convert('RGB'))
        h, w = img_np.shape[:2]

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Canny edge detection
        edges = cv2.Canny(gray, 40, 120)

        # Hough probabilistic line transform.
        # minLineLength: lines must span at least 12% of the shorter image dimension
        # (grout lines in a mosaic extend across the tile region, not just one tile).
        min_line_len = max(50, min(w, h) // 8)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=60,
            minLineLength=min_line_len,
            maxLineGap=25,
        )

        if lines is None:
            if verbose:
                print("  Tile detector: no lines found")
            return None

        # Classify each line as horizontal or vertical
        h_positions = []  # y-midpoints of horizontal lines
        v_positions = []  # x-midpoints of vertical lines

        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            length = (dx * dx + dy * dy) ** 0.5
            if length == 0:
                continue
            angle_deg = abs(np.degrees(np.arctan2(dy, dx)))

            if angle_deg < 12 or angle_deg > 168:  # Horizontal (±12°)
                h_positions.append((y1 + y2) / 2.0)
            elif 78 < angle_deg < 102:              # Vertical (90° ±12°)
                v_positions.append((x1 + x2) / 2.0)

        h_score, h_spacing, h_bbox = self._regularity_score(
            h_positions, axis_len=h, other_len=w, label='H', verbose=verbose
        )
        v_score, v_spacing, v_bbox = self._regularity_score(
            v_positions, axis_len=w, other_len=h, label='V', verbose=verbose
        )

        combined = h_score * v_score

        if verbose:
            print(
                f"  Tile detector: H_lines={len(h_positions)} score={h_score:.3f} "
                f"spacing={h_spacing:.0f}px | "
                f"V_lines={len(v_positions)} score={v_score:.3f} "
                f"spacing={v_spacing:.0f}px | "
                f"combined={combined:.4f} (threshold={self.threshold})"
            )

        if combined < self.threshold:
            return None

        # Build bbox from the detected line regions
        bbox = self._build_bbox(h_bbox, v_bbox, w, h)
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)

        if area / (w * h) < self.min_coverage:
            if verbose:
                print(f"  Tile detector: region too small ({area / (w * h) * 100:.1f}%)")
            return None

        confidence = min(0.50, 0.30 + combined * 8.0)

        if verbose:
            print(
                f"  Tile detector: mosaic at {bbox}, "
                f"conf={confidence:.2f}, coverage={area / (w * h) * 100:.1f}%"
            )

        return Detection(
            bbox=bbox,
            confidence=confidence,
            class_name='mosaic',
            area=area,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _regularity_score(
        self,
        positions: list,
        axis_len: int,
        other_len: int,
        label: str,
        verbose: bool,
    ) -> Tuple[float, float, Optional[Tuple[int, int]]]:
        """
        Score regularity using pairwise spacing voting.

        For each pair of detected lines, vote for their spacing in a histogram.
        A mosaic produces a sharp peak at the tile spacing T (and smaller peaks
        at harmonics 2T, 3T... for missed grout lines).

        Score = (votes near histogram mode) / (total valid votes) × coverage.

        Returns (score, mode_spacing, (min_pos, max_pos)).
        """
        if len(positions) < self.min_line_count:
            return 0.0, 0.0, None

        pos_arr = np.array(sorted(positions), dtype=np.float32)
        n = len(pos_arr)

        # Cap to avoid O(n²) blowup on pathological images
        if n > 250:
            step = n / 250
            pos_arr = pos_arr[[int(i * step) for i in range(250)]]
            n = 250

        # Compute all pairwise spacings in [min_tile_px, max_tile_px].
        # Since positions are sorted, pos_arr[j] - pos_arr[i] > 0 for j > i,
        # and we can break early once spacing exceeds max_tile_px.
        valid_spacings = []
        for i in range(n):
            for j in range(i + 1, n):
                sp = float(pos_arr[j] - pos_arr[i])
                if sp > self.max_tile_px:
                    break  # positions are sorted; further j only grow
                if sp >= self.min_tile_px:
                    valid_spacings.append(sp)

        # Require enough pairs to make the histogram statistically meaningful
        min_pairs = max(self.min_line_count * 2, 30)
        if len(valid_spacings) < min_pairs:
            return 0.0, 0.0, None

        valid_spacings = np.array(valid_spacings, dtype=np.float32)

        # Histogram of spacings — a mosaic produces a clear mode at tile size T
        n_bins = 40
        hist, bin_edges = np.histogram(
            valid_spacings,
            bins=n_bins,
            range=(self.min_tile_px, self.max_tile_px),
        )

        # Mode bin: spacing with the most supporting pairs
        mode_bin = int(np.argmax(hist))
        bin_width = (self.max_tile_px - self.min_tile_px) / n_bins
        mode_sp = float(bin_edges[mode_bin] + bin_width / 2)

        # Widen to ±1 bin to allow slight misalignment / boundary effects
        lo = max(0, mode_bin - 1)
        hi = min(n_bins - 1, mode_bin + 1)
        windowed_votes = int(np.sum(hist[lo:hi + 1]))
        total_votes = len(valid_spacings)

        score = windowed_votes / total_votes

        # Coverage: lines should span a meaningful portion of the axis
        pos_range = float(pos_arr[-1] - pos_arr[0])
        coverage = min(1.0, pos_range / max(1.0, axis_len * 0.5))

        final_score = score * coverage

        bbox_range = (int(pos_arr[0]), int(pos_arr[-1]))

        if verbose:
            print(
                f"    {label}: n={n}, pairs={total_votes}, "
                f"mode={mode_sp:.0f}px, votes={windowed_votes}/{total_votes}"
                f"={score:.3f}, cov={coverage:.3f}, final={final_score:.3f}"
            )

        return final_score, mode_sp, bbox_range

    def _build_bbox(
        self,
        h_bbox: Optional[Tuple[int, int]],
        v_bbox: Optional[Tuple[int, int]],
        w: int,
        h: int,
    ) -> Tuple[int, int, int, int]:
        """Combine horizontal/vertical line ranges into an image bbox."""
        # y range from horizontal lines
        if h_bbox:
            y1 = max(0, h_bbox[0] - self.min_tile_px)
            y2 = min(h, h_bbox[1] + self.min_tile_px)
        else:
            y1, y2 = 0, h

        # x range from vertical lines
        if v_bbox:
            x1 = max(0, v_bbox[0] - self.min_tile_px)
            x2 = min(w, v_bbox[1] + self.min_tile_px)
        else:
            x1, x2 = 0, w

        return (x1, y1, x2, y2)
