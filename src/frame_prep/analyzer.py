"""Saliency and composition analysis for fallback cropping."""

from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image


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
