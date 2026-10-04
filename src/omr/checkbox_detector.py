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
        min_box_size: int = 14,
        max_box_size: int = 55,
        aspect_ratio_range: Tuple[float, float] = (0.75, 1.30),
        checked_threshold: float = 0.14,
        unchecked_threshold: float = 0.06
    ):
        self.min_box_size = min_box_size
        self.max_box_size = max_box_size
        self.aspect_ratio_min, self.aspect_ratio_max = aspect_ratio_range
        self.checked_threshold = checked_threshold
        self.unchecked_threshold = unchecked_threshold

    def detect_and_classify(
        self,
        image: np.ndarray,
        page_number: int = 1,
        roi_bbox: Optional[BoundingBox] = None
    ) -> List[CheckboxField]:
        """
        Detects all checkbox elements on the page or inside a given ROI (e.g. within a specific table cell).
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

        # Adaptive thresholding to isolate sharp borders
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
        )

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        checkboxes: List[CheckboxField] = []
        seen_boxes = []

        for i, c in enumerate(contours):
            # Check contour hierarchy: we want contours with or without children
            x, y, w, h = cv2.boundingRect(c)

            # Filter by dimension
            if not (self.min_box_size <= w <= self.max_box_size and self.min_box_size <= h <= self.max_box_size):
                continue

            aspect_ratio = float(w) / float(h)
            if not (self.aspect_ratio_min <= aspect_ratio <= self.aspect_ratio_max):
                continue

            # Check approximate rectangularity
            area = cv2.contourArea(c)
            rect_area = w * h
            extent = area / float(rect_area) if rect_area > 0 else 0
            # Outline extent for a box is typically > 0.4
            if extent < 0.25:
                continue

            # Non-maximum suppression / deduplication of nested borders
            cx, cy = x + w // 2, y + h // 2
            duplicate = False
            for (prev_cx, prev_cy, prev_w, prev_h) in seen_boxes:
                if abs(cx - prev_cx) < 8 and abs(cy - prev_cy) < 8:
                    duplicate = True
                    break
            if duplicate:
                continue

            seen_boxes.append((cx, cy, w, h))

            # Crop interior region (exclude 20% border on each side to avoid counting the box frame itself)
            margin_x = max(2, int(w * 0.22))
            margin_y = max(2, int(h * 0.22))
            
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
        """
        Classifies an isolated checkbox interior crop.
        """
        if inner_crop is None or inner_crop.size == 0:
            return MarkState.INDETERMINATE, 0.5

        total_pixels = inner_crop.size
        ink_pixels = cv2.countNonZero(inner_crop)
        fill_ratio = ink_pixels / float(total_pixels)

        # 1. Unchecked: Very little ink inside
        if fill_ratio < self.unchecked_threshold:
            conf = max(0.85, 1.0 - (fill_ratio / self.unchecked_threshold) * 0.15)
            return MarkState.UNCHECKED, round(conf, 3)

        # 2. Strong fill / clear checkmark / 'X'
        if fill_ratio >= self.checked_threshold:
            # Check for single horizontal strike-through (strikethrough / voided mark)
            if self._is_strikethrough(inner_crop):
                return MarkState.CROSSED_OUT, 0.92
            
            conf = min(0.99, 0.85 + (fill_ratio * 0.14))
            return MarkState.CHECKED, round(conf, 3)

        # 3. Intermediate ink level: test for delicate checkmarks or crosses
        has_cross_or_tick, feature_conf = self._detect_cross_or_tick(inner_crop)
        if has_cross_or_tick:
            return MarkState.CHECKED, round(feature_conf, 3)

        # 4. Ambiguous zone
        return MarkState.INDETERMINATE, 0.65

    def _is_strikethrough(self, crop: np.ndarray) -> bool:
        """Detects a horizontal strike through the middle of the box."""
        h, w = crop.shape[:2]
        if h < 4 or w < 4:
            return False
        mid_y = h // 2
        # Check if ink is concentrated along a single horizontal slice
        center_row = crop[max(0, mid_y - 1):min(h, mid_y + 2), :]
        center_ink = cv2.countNonZero(center_row)
        top_ink = cv2.countNonZero(crop[:max(1, mid_y - 2), :])
        bottom_ink = cv2.countNonZero(crop[min(h, mid_y + 3):, :])
        
        if center_ink > 0 and (top_ink + bottom_ink) < (center_ink * 0.3):
            return True
        return False

    def _detect_cross_or_tick(self, crop: np.ndarray) -> Tuple[bool, float]:
        """Detects characteristic diagonal strokes of a checkmark or X."""
        h, w = crop.shape[:2]
        if h < 5 or w < 5:
            return False, 0.5
        
        # Diagonal projections
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
