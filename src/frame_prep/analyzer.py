"""Saliency, composition, and text analysis for detection filtering."""

import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image

_TEXT_MAX_DIM = 320  # Rescale crops for speed (~160ms per crop with EasyOCR)
TEXT_RATIO_THRESHOLD = 0.15  # Regions with >15% text coverage are filtered
_TEXT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache" / "text_detect"


class TextDetector:
    """Detects text regions using EasyOCR (CRAFT) to filter signs/labels.

    Fewer false positives on art than EAST, catches handwritten/graffiti text.
    Results are cached to disk keyed on crop content hash.
    """

    def __init__(self):
        self._reader = None

    def _load(self):
        if self._reader is not None:
            return True
        try:
            import easyocr
            import torch
            self._reader = easyocr.Reader(['en', 'fr'], gpu=False, verbose=False)
            # torch.compile gives ~2x speedup on CPU
            if hasattr(self._reader, 'detector'):
                self._reader.detector = torch.compile(
                    self._reader.detector, mode='reduce-overhead')
            if hasattr(self._reader, 'recognizer'):
                self._reader.recognizer = torch.compile(
                    self._reader.recognizer, mode='reduce-overhead')
            return True
        except ImportError:
            return False

    @staticmethod
    def _cache_key(image: Image.Image, bbox: Tuple[int, int, int, int]) -> str:
        """Compute cache key from crop pixel content + bbox."""
        bx1, by1, bx2, by2 = bbox
        crop = image.crop((bx1, by1, bx2, by2))
        # Hash a downsampled version for speed (cache key doesn't need full res)
        thumb = crop.resize((64, 64))
        data = np.array(thumb).tobytes()
        # Version bumped when detection method or filtering changes
        _cache_version = 4  # v4: readtext + conf filter + Shoelace polygon area
        key_str = f"v{_cache_version}:{hashlib.md5(data).hexdigest()}:{bx1},{by1},{bx2},{by2}:{_TEXT_MAX_DIM}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:20]

    def text_ratio(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> float:
        """Fraction of a detection region covered by recognized text (0.0-1.0).

        Crops the bbox from the image, rescales to _TEXT_MAX_DIM, and runs
        full OCR (detect + recognize). Using readtext() instead of detect()
        eliminates false positives on geometric art patterns — if the
        recognizer can't read actual text, the detection is ignored.

        Results are cached to disk.
        """
        # Check cache
        _TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_key = self._cache_key(image, bbox)
        cache_file = _TEXT_CACHE_DIR / f"{cache_key}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text())["ratio"]
            except (json.JSONDecodeError, KeyError, OSError):
                pass

        if not self._load():
            return 0.0

        bx1, by1, bx2, by2 = bbox
        crop = image.crop((bx1, by1, bx2, by2))
        w, h = crop.size
        if w < 32 or h < 32:
            return 0.0

        # Rescale for speed
        if max(w, h) > _TEXT_MAX_DIM:
            if w >= h:
                crop = crop.resize((_TEXT_MAX_DIM, max(1, int(h * _TEXT_MAX_DIM / w))), Image.LANCZOS)
            else:
                crop = crop.resize((max(1, int(w * _TEXT_MAX_DIM / h)), _TEXT_MAX_DIM), Image.LANCZOS)

        img_np = np.array(crop)
        cw, ch = crop.size

        # readtext() returns (bbox_points, text, confidence) tuples.
        # Only recognized text counts — geometric patterns that CRAFT
        # detects but can't read are excluded.
        results = self._reader.readtext(img_np)

        text_area = 0
        for bbox_pts, text, conf in results:
            # Skip likely false-positive OCR results:
            # - Very low confidence (<0.1): art textures read as gibberish
            # - Short text with low confidence: geometric patterns as 1-3 chars
            if conf < 0.1 or (len(text.strip()) <= 3 and conf < 0.5):
                continue
            # Shoelace formula for oriented polygon area.
            n = len(bbox_pts)
            if n >= 3:
                area = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    area += bbox_pts[i][0] * bbox_pts[j][1]
                    area -= bbox_pts[j][0] * bbox_pts[i][1]
                text_area += abs(area) / 2

        ratio = text_area / (cw * ch)

        # Save to cache
        try:
            cache_file.write_text(json.dumps({"ratio": ratio}))
        except OSError:
            pass

        return ratio


class CompositionAnalyzer:
    """Analyzes image composition to find points of interest."""

    def __init__(self):
        """Initialize composition analyzer."""
        self._saliency = None

    def _get_saliency_detector(self):
        """Lazy-load OpenCV saliency detector."""
        if self._saliency is None:
            try:
                self._saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            except Exception:
                # Fallback if saliency module not available
                return None
        return self._saliency

    def analyze_saliency(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Compute saliency map for image.

        Args:
            image: PIL Image

        Returns:
            Saliency map as numpy array, or None if detection fails
        """
        detector = self._get_saliency_detector()
        if detector is None:
            return None

        # Convert to numpy array
        img_array = np.array(image)

        # Convert to BGR for OpenCV
        if len(img_array.shape) == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        try:
            success, saliency_map = detector.computeSaliency(img_bgr)
            if success:
                return saliency_map
        except Exception:
            pass

        return None

    def find_interest_points(self, saliency_map: np.ndarray) -> Tuple[int, int]:
        """
        Find the most interesting point in saliency map.

        Args:
            saliency_map: Saliency map from analyze_saliency

        Returns:
            (x, y) coordinates of most interesting point
        """
        # Find maximum saliency point
        max_loc = np.unravel_index(saliency_map.argmax(), saliency_map.shape)
        y, x = max_loc  # OpenCV uses (row, col) = (y, x)
        return (x, y)

    def get_best_anchor(self, image: Image.Image) -> Optional[Tuple[int, int]]:
        """
        Get best anchor point for cropping using saliency analysis.

        Args:
            image: PIL Image

        Returns:
            (x, y) coordinates of anchor point, or None if analysis fails
        """
        saliency_map = self.analyze_saliency(image)
        if saliency_map is None:
            return None

        return self.find_interest_points(saliency_map)
