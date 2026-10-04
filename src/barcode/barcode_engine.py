"""
Multi-Engine Barcode Detection & Decoding Subsystem.
Supports 2D (DataMatrix, PDF417, QR Code) and 1D (Code128, Code39, EAN-13) symbologies.
Includes pre-processing for faxed, blurred, and low-contrast clinical documents.
"""

import logging
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.schemas.clinical_schema import BarcodeItem, BarcodeSymbology, BoundingBox

logger = logging.getLogger(__name__)

# Try optional external barcode libraries
try:
    import zxingcpp
    HAS_ZXING = True
except ImportError:
    HAS_ZXING = False

try:
    import pylibdmtx.pylibdmtx as dmtx
    HAS_DMTX = True
except ImportError:
    HAS_DMTX = False

try:
    from pdf417decoder import PDF417Decoder
    HAS_PDF417 = True
except ImportError:
    HAS_PDF417 = False

try:
    from pyzbar import pyzbar
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False


class BarcodeEngine:
    """
    Robust barcode decoder combining computer vision candidate localization
    with multi-engine fallback (ZXing, OpenCV, PyLibDMTX, PDF417Decoder, PyZBar).
    """

    def __init__(self, enhance_contrast: bool = True):
        self.enhance_contrast = enhance_contrast
        self._cv2_barcode_detector = None
        self._init_opencv_detector()

    def _init_opencv_detector(self):
        """Initialize OpenCV's built-in 1D/2D BarcodeDetector if available."""
        try:
            if hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
                self._cv2_barcode_detector = cv2.barcode.BarcodeDetector()
        except Exception as e:
            logger.debug(f"OpenCV BarcodeDetector unavailable: {e}")

    def decode_document(self, image: np.ndarray, page_number: int = 1) -> List[BarcodeItem]:
        """
        Scan a full document image for any 1D or 2D barcodes across all supported engines.
        """
        if image is None or image.size == 0:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        h, w = gray.shape[:2]

        results: List[BarcodeItem] = []
        seen_values = set()

        # Engine 1: ZXing C++ (Best for DataMatrix, PDF417, QR, Code128)
        if HAS_ZXING:
            zxing_results = self._decode_with_zxing(gray, page_number)
            for item in zxing_results:
                if item.raw_value not in seen_values:
                    seen_values.add(item.raw_value)
                    results.append(item)

        # Engine 2: PyLibDMTX (Specialized DataMatrix engine)
        if HAS_DMTX and not any(r.symbology == BarcodeSymbology.DATAMATRIX for r in results):
            dmtx_results = self._decode_with_pylibdmtx(gray, page_number)
            for item in dmtx_results:
                if item.raw_value not in seen_values:
                    seen_values.add(item.raw_value)
                    results.append(item)

        # Engine 3: PDF417Decoder (Specialized PDF417 engine)
        if HAS_PDF417 and not any(r.symbology == BarcodeSymbology.PDF417 for r in results):
            pdf417_results = self._decode_with_pdf417decoder(image, page_number)
            for item in pdf417_results:
                if item.raw_value not in seen_values:
                    seen_values.add(item.raw_value)
                    results.append(item)

        # Engine 4: OpenCV BarcodeDetector (Native C++ 1D / 2D)
        if self._cv2_barcode_detector is not None:
            cv_results = self._decode_with_opencv(gray, page_number)
            for item in cv_results:
                if item.raw_value not in seen_values:
                    seen_values.add(item.raw_value)
                    results.append(item)

        # Engine 5: PyZBar (General 1D & QR)
        if HAS_PYZBAR and len(results) == 0:
            pyzbar_results = self._decode_with_pyzbar(gray, page_number)
            for item in pyzbar_results:
                if item.raw_value not in seen_values:
                    seen_values.add(item.raw_value)
                    results.append(item)

        # If full-image failed and contrast enhancement is on, try localized ROI crops
        if len(results) == 0 and self.enhance_contrast:
            roi_results = self._scan_barcode_rois(gray, image, page_number)
            for item in roi_results:
                if item.raw_value not in seen_values:
                    seen_values.add(item.raw_value)
                    results.append(item)

        return results

    def _decode_with_zxing(self, gray: np.ndarray, page_number: int) -> List[BarcodeItem]:
        items = []
        try:
            barcodes = zxingcpp.read_barcodes(gray)
            for b in barcodes:
                symbology = self._map_zxing_format(b.format)
                bbox = None
                if b.position:
                    xs = [p.x for p in [b.position.top_left, b.position.top_right, b.position.bottom_right, b.position.bottom_left]]
                    ys = [p.y for p in [b.position.top_left, b.position.top_right, b.position.bottom_right, b.position.bottom_left]]
                    bbox = BoundingBox(ymin=int(min(ys)), xmin=int(min(xs)), ymax=int(max(ys)), xmax=int(max(xs)))

                items.append(BarcodeItem(
                    symbology=symbology,
                    raw_value=b.text,
                    is_valid_checksum=b.is_valid if hasattr(b, 'is_valid') else True,
                    confidence=1.0,
                    bounding_box=bbox,
                    page_number=page_number
                ))
        except Exception as e:
            logger.debug(f"ZXing decode error: {e}")
        return items

    def _decode_with_pylibdmtx(self, gray: np.ndarray, page_number: int) -> List[BarcodeItem]:
        items = []
        try:
            dmtx_res = dmtx.decode(gray, timeout=200)
            for d in dmtx_res:
                val = d.data.decode("utf-8", errors="replace")
                bbox = BoundingBox(
                    ymin=int(gray.shape[0] - (d.rect.top + d.rect.height)),
                    xmin=int(d.rect.left),
                    ymax=int(gray.shape[0] - d.rect.top),
                    xmax=int(d.rect.left + d.rect.width)
                )
                items.append(BarcodeItem(
                    symbology=BarcodeSymbology.DATAMATRIX,
                    raw_value=val,
                    is_valid_checksum=True,
                    confidence=1.0,
                    bounding_box=bbox,
                    page_number=page_number
                ))
        except Exception as e:
            logger.debug(f"PyLibDMTX decode error: {e}")
        return items

    def _decode_with_pdf417decoder(self, image: np.ndarray, page_number: int) -> List[BarcodeItem]:
        items = []
        try:
            from PIL import Image
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            decoder = PDF417Decoder(pil_img)
            if decoder.decode() > 0:
                val = decoder.barcode_data_index_to_string(0)
                items.append(BarcodeItem(
                    symbology=BarcodeSymbology.PDF417,
                    raw_value=val,
                    is_valid_checksum=True,
                    confidence=1.0,
                    page_number=page_number
                ))
        except Exception as e:
            logger.debug(f"PDF417 decoder error: {e}")
        return items

    def _decode_with_opencv(self, gray: np.ndarray, page_number: int) -> List[BarcodeItem]:
        items = []
        try:
            retval, decoded_info, decoded_type, points = self._cv2_barcode_detector.detectAndDecode(gray)
            if retval and decoded_info:
                for text, btype, pts in zip(decoded_info, decoded_type, points):
                    if text:
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        bbox = BoundingBox(ymin=int(min(ys)), xmin=int(min(xs)), ymax=int(max(ys)), xmax=int(max(xs)))
                        items.append(BarcodeItem(
                            symbology=self._map_generic_type(btype),
                            raw_value=text,
                            is_valid_checksum=True,
                            confidence=0.95,
                            bounding_box=bbox,
                            page_number=page_number
                        ))
        except Exception as e:
            logger.debug(f"OpenCV barcode decode error: {e}")
        return items

    def _decode_with_pyzbar(self, gray: np.ndarray, page_number: int) -> List[BarcodeItem]:
        items = []
        try:
            decoded = pyzbar.decode(gray)
            for d in decoded:
                sym = self._map_generic_type(d.type)
                val = d.data.decode("utf-8", errors="replace")
                rect = d.rect
                bbox = BoundingBox(ymin=rect.top, xmin=rect.left, ymax=rect.top + rect.height, xmax=rect.left + rect.width)
                items.append(BarcodeItem(
                    symbology=sym,
                    raw_value=val,
                    is_valid_checksum=True,
                    confidence=1.0,
                    bounding_box=bbox,
                    page_number=page_number
                ))
        except Exception as e:
            logger.debug(f"PyZBar decode error: {e}")
        return items

    def _scan_barcode_rois(self, gray: np.ndarray, color_img: np.ndarray, page_number: int) -> List[BarcodeItem]:
        """
        Locate barcode candidates via morphological gradient (high horizontal frequency clusters)
        and test enhanced local crops.
        """
        items = []
        # Scharr gradient in X direction to isolate dense vertical/matrix bars
        grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(grad_x, grad_y)
        gradient = cv2.convertScaleAbs(gradient)

        # Blur and threshold
        blurred = cv2.blur(gradient, (9, 9))
        _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

        # Morphological close to join adjacent barcode elements
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)

        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 40 and h > 20 and (w * h) > 1200:
                # Add padding
                pad = 15
                y1 = max(0, y - pad)
                x1 = max(0, x - pad)
                y2 = min(gray.shape[0], y + h + pad)
                x2 = min(gray.shape[1], x + w + pad)
                crop_gray = gray[y1:y2, x1:x2]

                # Contrast enhancement via CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(crop_gray)

                if HAS_ZXING:
                    crop_items = self._decode_with_zxing(enhanced, page_number)
                    for item in crop_items:
                        if item.bounding_box:
                            # Remap bounding box to full document coordinates
                            item.bounding_box.ymin += y1
                            item.bounding_box.ymax += y1
                            item.bounding_box.xmin += x1
                            item.bounding_box.xmax += x1
                        items.append(item)
        return items

    @staticmethod
    def _map_zxing_format(fmt) -> BarcodeSymbology:
        name = str(fmt).upper()
        if "DATAMATRIX" in name:
            return BarcodeSymbology.DATAMATRIX
        elif "PDF417" in name or "PDF_417" in name:
            return BarcodeSymbology.PDF417
        elif "QR" in name:
            return BarcodeSymbology.QR_CODE
        elif "CODE_128" in name or "CODE128" in name:
            return BarcodeSymbology.CODE_128
        elif "CODE_39" in name or "CODE39" in name:
            return BarcodeSymbology.CODE_39
        elif "EAN_13" in name or "EAN13" in name:
            return BarcodeSymbology.EAN_13
        return BarcodeSymbology.UNKNOWN

    @staticmethod
    def _map_generic_type(btype: str) -> BarcodeSymbology:
        s = str(btype).upper()
        if "DATAMATRIX" in s or "DATA_MATRIX" in s:
            return BarcodeSymbology.DATAMATRIX
        elif "PDF417" in s or "PDF_417" in s:
            return BarcodeSymbology.PDF417
        elif "QR" in s:
            return BarcodeSymbology.QR_CODE
        elif "128" in s:
            return BarcodeSymbology.CODE_128
        elif "39" in s:
            return BarcodeSymbology.CODE_39
        elif "EAN" in s:
            return BarcodeSymbology.EAN_13
        return BarcodeSymbology.UNKNOWN
