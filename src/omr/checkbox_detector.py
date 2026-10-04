"""
Optical Mark Recognition (OMR) & Checkbox Classifier.
Detects checkbox contours and classifies mark state:
  - CHECKED (tick, X, fill)
  - UNCHECKED (empty interior)
  - CROSSED_OUT (correction / void strike)
  - INDETERMINATE (ambiguous)
"""

import logging
from typing import List, Tuple, Optional
import cv2
import numpy as np

from src.schemas.clinical_schema import BoundingBox, CheckboxField, MarkState

logger = logging.getLogger(__name__)


class CheckboxDetector:
    """
    Locates checkbox contours in clinical forms and classifies their marked state.
    """

    def __init__(
        self,
        min_box_size: int = 10,
        max_box_size: int = 55,
        aspect_ratio_range: Tuple[float, float] = (0.70, 1.35),
        checked_threshold: float = 0.14,
        unchecked_threshold: float = 0.06
    ):
        self.min_box_size = min_box_size
        self.max_box_size = max_box_size
        self.aspect_ratio_min, self.aspect_ratio_max = aspect_ratio_range
        self.checked_threshold = checked_threshold
        self.unchecked_threshold = unchecked_threshold

    def test_mark_at(
        self,
        gray_image: np.ndarray,
        center_x: int,
        center_y: int,
        box_size: int = 14,
        darkness_threshold: float = 160.0
    ) -> Tuple[MarkState, float]:
        """
        Directly measures visual mark state at a specific expected coordinate.
        Useful when testing checkbox boxes adjacent to known text labels.
        """
        h, w = gray_image.shape[:2]
        half = box_size // 2
        y1 = max(0, center_y - half)
        y2 = min(h, center_y + half)
        x1 = max(0, center_x - half)
        x2 = min(w, center_x + half)

        crop = gray_image[y1:y2, x1:x2]
        if crop.size == 0:
            return MarkState.INDETERMINATE, 0.5

        mean_val = float(np.mean(crop))
        # Dark crop (low value in grayscale) indicates checked / filled
        if mean_val < darkness_threshold:
            conf = min(0.99, max(0.85, (darkness_threshold - mean_val) / darkness_threshold + 0.8))
            return MarkState.CHECKED, round(conf, 3)
        else:
            conf = min(0.99, max(0.85, (mean_val - darkness_threshold) / (255.0 - darkness_threshold) + 0.8))
            return MarkState.UNCHECKED, round(conf, 3)

    def detect_and_classify(
        self,
        image: np.ndarray,
        page_number: int = 1,
        roi_bbox: Optional[BoundingBox] = None
    ) -> List[CheckboxField]:
        """
        Detects all checkbox elements on the page or inside a given ROI.
        """
        if image is None or image.size == 0:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        offset_x, offset_y = 0, 0
        if roi_bbox:
            offset_y, offset_x = roi_bbox.ymin, roi_bbox.xmin
            gray = gray[roi_bbox.ymin:roi_bbox.ymax, roi_bbox.xmin:roi_bbox.xmax]
            if gray.size == 0:
                return []

        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
        )

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        checkboxes: List[CheckboxField] = []
        seen_boxes = []

        for i, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)

            if not (self.min_box_size <= w <= self.max_box_size and self.min_box_size <= h <= self.max_box_size):
                continue

            aspect_ratio = float(w) / float(h)
            if not (self.aspect_ratio_min <= aspect_ratio <= self.aspect_ratio_max):
                continue

            area = cv2.contourArea(c)
            rect_area = w * h
            extent = area / float(rect_area) if rect_area > 0 else 0
            if extent < 0.20:
                continue

            cx, cy = x + w // 2, y + h // 2
            duplicate = False
            for (prev_cx, prev_cy, prev_w, prev_h) in seen_boxes:
                if abs(cx - prev_cx) < 8 and abs(cy - prev_cy) < 8:
                    duplicate = True
                    break
            if duplicate:
                continue

            seen_boxes.append((cx, cy, w, h))

            margin_x = max(1, int(w * 0.20))
            margin_y = max(1, int(h * 0.20))
            
            inner_crop = binary[y + margin_y : y + h - margin_y, x + margin_x : x + w - margin_x]
            mark_state, confidence = self.classify_mark(inner_crop)

            global_bbox = BoundingBox(
                ymin=int(y + offset_y),
                xmin=int(x + offset_x),
                ymax=int(y + h + offset_y),
                xmax=int(x + w + offset_x)
            )

            checkboxes.append(CheckboxField(
                field_key=f"checkbox_{global_bbox.ymin}_{global_bbox.xmin}",
                mark_state=mark_state,
                confidence=confidence,
                bounding_box=global_bbox,
                page_number=page_number
            ))

        return checkboxes

    def classify_mark(self, inner_crop: np.ndarray) -> Tuple[MarkState, float]:
        """Classifies an isolated checkbox interior crop."""
        if inner_crop is None or inner_crop.size == 0:
            return MarkState.INDETERMINATE, 0.5

        total_pixels = inner_crop.size
        ink_pixels = cv2.countNonZero(inner_crop)
        fill_ratio = ink_pixels / float(total_pixels)

        if fill_ratio < self.unchecked_threshold:
            conf = max(0.85, 1.0 - (fill_ratio / self.unchecked_threshold) * 0.15)
            return MarkState.UNCHECKED, round(conf, 3)

        if fill_ratio >= self.checked_threshold:
            if self._is_strikethrough(inner_crop):
                return MarkState.CROSSED_OUT, 0.92
            
            conf = min(0.99, 0.85 + (fill_ratio * 0.14))
            return MarkState.CHECKED, round(conf, 3)

        has_cross_or_tick, feature_conf = self._detect_cross_or_tick(inner_crop)
        if has_cross_or_tick:
            return MarkState.CHECKED, round(feature_conf, 3)

        return MarkState.INDETERMINATE, 0.65

    def _is_strikethrough(self, crop: np.ndarray) -> bool:
        h, w = crop.shape[:2]
        if h < 4 or w < 4:
            return False
        mid_y = h // 2
        center_row = crop[max(0, mid_y - 1):min(h, mid_y + 2), :]
        center_ink = cv2.countNonZero(center_row)
        top_ink = cv2.countNonZero(crop[:max(1, mid_y - 2), :])
        bottom_ink = cv2.countNonZero(crop[min(h, mid_y + 3):, :])
        
        if center_ink > 0 and (top_ink + bottom_ink) < (center_ink * 0.3):
            return True
        return False

    def _detect_cross_or_tick(self, crop: np.ndarray) -> Tuple[bool, float]:
        h, w = crop.shape[:2]
        if h < 5 or w < 5:
            return False, 0.5
        
        diag1_count = 0
        diag2_count = 0
        min_dim = min(h, w)
        for d in range(min_dim):
            if crop[d, d] > 0:
                diag1_count += 1
            if crop[d, min_dim - 1 - d] > 0:
                diag2_count += 1

        diag_ratio = max(diag1_count, diag2_count) / float(min_dim)
        if diag_ratio > 0.45:
            return True, 0.88
        return False, 0.5
