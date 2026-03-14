"""Saliency, composition, and text analysis for detection filtering."""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image

_EAST_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "frozen_east_text_detection.pb"
_EAST_MAX_DIM = 320  # Rescale crops for speed (~50-150ms per crop)
_EAST_CONF = 0.5


class TextDetector:
    """Detects text regions using EAST to filter signs/labels from art."""

    def __init__(self):
        self._net = None

    def _load(self):
        if self._net is not None:
            return True
        if not _EAST_MODEL_PATH.exists():
            return False
        self._net = cv2.dnn.readNet(str(_EAST_MODEL_PATH))
        return True

    def text_ratio(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> float:
        """Fraction of a detection region covered by text (0.0-1.0).

        Crops the bbox from the image, rescales to _EAST_MAX_DIM, and runs
        EAST text detection. Returns the ratio of text pixel area to total area.
        """
        if not self._load():
            return 0.0

        bx1, by1, bx2, by2 = bbox
        crop = image.crop((bx1, by1, bx2, by2))
        w, h = crop.size
        if w < 32 or h < 32:
            return 0.0

        # Rescale for speed
        if max(w, h) > _EAST_MAX_DIM:
            if w >= h:
                crop = crop.resize((_EAST_MAX_DIM, max(32, int(h * _EAST_MAX_DIM / w))), Image.LANCZOS)
            else:
                crop = crop.resize((max(32, int(w * _EAST_MAX_DIM / h)), _EAST_MAX_DIM), Image.LANCZOS)

        img = np.array(crop)
        ih, iw = img.shape[:2]
        new_w = max(32, (iw // 32) * 32)
        new_h = max(32, (ih // 32) * 32)
        resized = cv2.resize(img, (new_w, new_h))

        blob = cv2.dnn.blobFromImage(resized, 1.0, (new_w, new_h),
                                      (123.68, 116.78, 103.94), True, False)
        self._net.setInput(blob)
        scores, geometry = self._net.forward(
            ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'])

        num_rows, num_cols = scores.shape[2:4]
        rects, confs = [], []
        for y in range(num_rows):
            for x in range(num_cols):
                if scores[0, 0, y, x] < _EAST_CONF:
                    continue
                ox, oy = x * 4.0, y * 4.0
                h_box = geometry[0, 0, y, x] + geometry[0, 2, y, x]
                w_box = geometry[0, 1, y, x] + geometry[0, 3, y, x]
                ex = int(ox + geometry[0, 1, y, x])
                ey = int(oy + geometry[0, 2, y, x])
                rects.append((int(ex - w_box), int(ey - h_box), ex, ey))
                confs.append(float(scores[0, 0, y, x]))

        if not rects:
            return 0.0

        boxes_nms = [[r[0], r[1], r[2] - r[0], r[3] - r[1]] for r in rects]
        indices = cv2.dnn.NMSBoxes(boxes_nms, confs, _EAST_CONF, 0.4)

        text_area = 0
        if len(indices) > 0:
            for i in indices.flatten():
                sx, sy, ex, ey = rects[i]
                sx, sy = max(0, sx), max(0, sy)
                ex, ey = min(new_w, ex), min(new_h, ey)
                text_area += (ex - sx) * (ey - sy)

        return text_area / (new_w * new_h)


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
