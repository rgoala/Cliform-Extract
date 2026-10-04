"""
Text Recognition Engine (Printed OCR & Handwritten HTR).
Extracts text lines and tokens with character/word bounding boxes.
Automatically discovers and configures Tesseract on Windows.
"""

import logging
import os
import shutil
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.schemas.clinical_schema import BoundingBox, ExtractedField, ModalityType

logger = logging.getLogger(__name__)

# Check for pytesseract
try:
    import pytesseract
    HAS_TESSERACT = True

    # Auto-detect Tesseract binary on Windows if not already on PATH
    if not shutil.which("tesseract"):
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
        ]
        for p in common_paths:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                logger.info(f"Configured Tesseract binary at: {p}")
                break
except ImportError:
    HAS_TESSERACT = False

# Check for PaddleOCR
try:
    from paddleocr import PaddleOCR
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False


class TextEngine:
    """
    Wrapper for printed OCR and handwriting recognition,
    producing spatially grounded tokens with coordinates and confidence.
    """

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._paddle_ocr = None
        if HAS_PADDLE:
            try:
                self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            except Exception as e:
                logger.debug(f"Failed to init PaddleOCR: {e}")

    def extract_document_tokens(self, image: np.ndarray, page_number: int = 1) -> List[ExtractedField]:
        """
        Transcribes all text on the page into ExtractedField objects with bounding boxes.
        """
        if image is None or image.size == 0:
            return []

        tokens: List[ExtractedField] = []

        # 1. Primary: PaddleOCR (if installed)
        if self._paddle_ocr is not None:
            try:
                result = self._paddle_ocr.ocr(image, cls=True)
                if result and result[0]:
                    for line in result[0]:
                        coords, (text, conf) = line
                        xs = [int(p[0]) for p in coords]
                        ys = [int(p[1]) for p in coords]
                        bbox = BoundingBox(ymin=min(ys), xmin=min(xs), ymax=max(ys), xmax=max(xs))
                        tokens.append(ExtractedField(
                            field_key="token",
                            raw_text=text.strip(),
                            normalized_value=text.strip(),
                            confidence=round(float(conf), 3),
                            modality=ModalityType.PRINTED,
                            bounding_box=bbox,
                            page_number=page_number
                        ))
                if tokens:
                    return tokens
            except Exception as e:
                logger.debug(f"PaddleOCR extraction failed: {e}")

        # 2. PyTesseract with bounding boxes
        if HAS_TESSERACT:
            try:
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                n_boxes = len(data["text"])
                for i in range(n_boxes):
                    text = data["text"][i].strip()
                    conf = float(data["conf"][i])
                    if text and conf > 15:
                        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                        bbox = BoundingBox(ymin=y, xmin=x, ymax=y + h, xmax=x + w)
                        tokens.append(ExtractedField(
                            field_key="token",
                            raw_text=text,
                            normalized_value=text,
                            confidence=round(conf / 100.0, 3),
                            modality=ModalityType.PRINTED,
                            bounding_box=bbox,
                            page_number=page_number
                        ))
                return tokens
            except Exception as e:
                logger.error(f"PyTesseract extraction error: {e}")

        return tokens

    def extract_crop_text(self, image: np.ndarray, bbox: BoundingBox) -> Tuple[str, float]:
        """Transcribes text inside a specific bounding box crop."""
        y1, y2 = max(0, bbox.ymin), min(image.shape[0], bbox.ymax)
        x1, x2 = max(0, bbox.xmin), min(image.shape[1], bbox.xmax)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return "", 0.0

        if HAS_TESSERACT:
            try:
                text = pytesseract.image_to_string(crop, config="--psm 6").strip()
                return text, 0.90
            except Exception:
                pass

        return "", 0.0
