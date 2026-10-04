"""
Morphological Table & Grid Cell Segmenter.
Discovers table structures, rows, columns, and individual form cells
dynamically across varying healthcare provider layouts.
"""

import logging
from typing import List, Tuple
import cv2
import numpy as np

from src.schemas.clinical_schema import BoundingBox

logger = logging.getLogger(__name__)


class TableCell:
    def __init__(self, bbox: BoundingBox, is_header: bool = False):
        self.bbox = bbox
        self.is_header = is_header
        self.label_region: Optional[BoundingBox] = None
        self.value_region: Optional[BoundingBox] = None


class TableSegmenter:
    """
    Extracts table grids and cell boundaries using morphological line detection.
    Enables provider-adaptive extraction by decoupling cell layout from hardcoded coordinates.
    """

    def __init__(
        self,
        min_cell_width: int = 40,
        min_cell_height: int = 18,
        line_min_length_ratio: float = 0.03
    ):
        self.min_cell_width = min_cell_width
        self.min_cell_height = min_cell_height
        self.line_min_length_ratio = line_min_length_ratio

    def segment_cells(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Detects all rectangular table cells on the form page.
        """
        if image is None or image.size == 0:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        h, w = gray.shape[:2]

        # Invert binary: lines are white (255), background is black (0)
        binary = cv2.adaptiveThreshold(
            ~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2
        )

        # 1. Isolate horizontal lines
        scale_w = max(15, int(w * self.line_min_length_ratio))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale_w, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

        # 2. Isolate vertical lines
        scale_h = max(15, int(h * self.line_min_length_ratio))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale_h))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

        # 3. Combine grid lines
        table_grid = cv2.add(horizontal, vertical)

        # Dilate slightly to bridge tiny gaps at intersections
        grid_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        table_grid = cv2.dilate(table_grid, grid_kernel, iterations=1)

        # 4. Find enclosed cell contours
        contours, hierarchy = cv2.findContours(table_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cells: List[BoundingBox] = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            # Filter out the full-page boundary or excessively large blocks
            if cw >= (w * 0.98) and ch >= (h * 0.95):
                continue

            # Must satisfy minimum cell dimensions
            if cw >= self.min_cell_width and ch >= self.min_cell_height:
                # Discard long thin single lines
                aspect = cw / float(ch)
                if aspect > 45 or aspect < 0.02:
                    continue

                cells.append(BoundingBox(
                    ymin=int(y),
                    xmin=int(x),
                    ymax=int(y + ch),
                    xmax=int(x + cw)
                ))

        # Sort cells in reading order: top-to-bottom, left-to-right
        cells = self._sort_reading_order(cells)
        return cells

    @staticmethod
    def _sort_reading_order(boxes: List[BoundingBox], y_tolerance: int = 12) -> List[BoundingBox]:
        """
        Sorts boxes in natural human reading order with row tolerance.
        """
        if not boxes:
            return []

        # Sort primarily by ymin, then xmin
        sorted_by_y = sorted(boxes, key=lambda b: (b.ymin, b.xmin))
        
        # Group into rows within y_tolerance
        rows: List[List[BoundingBox]] = []
        current_row = [sorted_by_y[0]]

        for b in sorted_by_y[1:]:
            if abs(b.ymin - current_row[0].ymin) <= y_tolerance:
                current_row.append(b)
            else:
                rows.append(sorted(current_row, key=lambda item: item.xmin))
                current_row = [b]
        if current_row:
            rows.append(sorted(current_row, key=lambda item: item.xmin))

        # Flatten
        ordered = []
        for r in rows:
            ordered.extend(r)
        return ordered
