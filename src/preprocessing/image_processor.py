"""
Image Preprocessing and Normalization Module.
Handles rotational deskewing, DPI normalization, contrast enhancement,
and fax noise/speckle removal.
"""

import logging
from typing import Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Normalizes scanned clinical documents for optimal OCR, HTR, OMR, and barcode decoding.
    """

    def __init__(self, standard_page_width: int = 2480):
        # 2480 pixels ~ 300 DPI for US Letter / A4 width
        self.standard_page_width = standard_page_width

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Runs the full normalization pipeline:
          1. Contrast & illumination normalization
          2. Rotational deskewing
          3. Resolution standardization
        Returns: (normalized_image, detected_skew_angle)
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid.")

        # 1. Deskew
        deskewed, angle = self.deskew(image)

        # 2. DPI / dimension standardization
        normalized = self.standardize_resolution(deskewed)

        return normalized, angle

    def deskew(self, image: np.ndarray, max_angle: float = 20.0) -> Tuple[np.ndarray, float]:
        """
        Detects document skew angle using minAreaRect on text contours and rotates.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        # Invert: white text on black background
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Dilate text horizontally to form solid text lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        # Find text block coordinates
        coords = np.column_stack(np.where(dilated > 0))
        if len(coords) < 100:
            return image, 0.0

        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust OpenCV angle convention
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle

        # Guard against extreme erroneous rotations
        if abs(angle) > max_angle or abs(angle) < 0.2:
            return image, 0.0

        # Rotate image around center
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

        return rotated, angle

    def standardize_resolution(self, image: np.ndarray) -> np.ndarray:
        """Scales low-resolution images to standard ~300 DPI, preserving native high-res images."""
        (h, w) = image.shape[:2]
        # If image is already high resolution (>= 1600px wide, e.g. 200-300 DPI scans), preserve original pixels
        if w >= 1600:
            return image

        scale = self.standard_page_width / float(w)
        target_h = int(h * scale)
        return cv2.resize(image, (self.standard_page_width, target_h), interpolation=cv2.INTER_LANCZOS4)


    def enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def remove_fax_noise(self, binary: np.ndarray) -> np.ndarray:
        """Removes isolated 1-2px salt-and-pepper noise typical in fax transmissions."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return opened
