"""
Unified Clinical Form & Questionnaire Extraction Pipeline Orchestrator.
Combines Preprocessing, Barcode Decoding, Grid Table Parsing, OMR, OCR/HTR,
Semantic Extraction, and HIPAA Safeguards into an end-to-end service.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from src.barcode.barcode_engine import BarcodeEngine
from src.extraction.form_extractor import FormExtractor
from src.layout.table_segmenter import TableSegmenter
from src.omr.checkbox_detector import CheckboxDetector
from src.preprocessing.image_processor import ImageProcessor
from src.ocr.text_engine import TextEngine
from src.schemas.clinical_schema import ClinicalPriorAuthResult
from src.security.hipaa_compliance import HIPAASecurityManager

logger = logging.getLogger(__name__)


class ClinicalFormPipeline:
    """
    End-to-End Clinical Form Extraction Engine.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        alias_config_path: Optional[str] = None,
        enable_hipaa_redaction: bool = False
    ):
        self.enable_hipaa_redaction = enable_hipaa_redaction
        self.image_processor = ImageProcessor()
        self.barcode_engine = BarcodeEngine()
        self.table_segmenter = TableSegmenter()
        self.checkbox_detector = CheckboxDetector()
        self.text_engine = TextEngine()
        self.form_extractor = FormExtractor(alias_config_path=alias_config_path)
        self.hipaa_manager = HIPAASecurityManager()

    def process_image(
        self,
        image_input: Union[str, np.ndarray],
        document_id: str = "DOC_001",
        actor_id: str = "system_operator",
        redact_phi: bool = False
    ) -> Tuple[ClinicalPriorAuthResult, Dict[str, Any]]:
        """
        Executes complete extraction pipeline on a clinical document image.

        Args:
            image_input: File path (str) or OpenCV image array (np.ndarray).
            document_id: Unique identifier for tracking and audit.
            actor_id: Requesting user/service ID for HIPAA audit logs.
            redact_phi: Whether to scrub PHI identifiers per HIPAA Safe Harbor.

        Returns:
            Tuple of (Structured ClinicalPriorAuthResult, HIPAA Audit Log Entry)
        """
        # 1. Load image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise FileNotFoundError(f"Could not load image from {image_input}")
        else:
            image = image_input

        # 2. Stage 1: Pre-processing & Deskewing
        normalized_img, skew_angle = self.image_processor.process(image)

        # 3. Stage 2: Parallel Perception
        # 3a. Barcode Detection (DataMatrix, PDF417, 1D)
        barcodes = self.barcode_engine.decode_document(normalized_img)

        # 3b. Table & Grid Cell Segmentation
        cells = self.table_segmenter.segment_cells(normalized_img)

        # 3c. Checkbox & Mark Detection (OMR)
        checkboxes = self.checkbox_detector.detect_and_classify(normalized_img)

        # 3d. Text Transcription (OCR / HTR)
        tokens = self.text_engine.extract_document_tokens(normalized_img)

        # 4. Stage 3: Semantic Association & JSON Structuring
        result = self.form_extractor.link_and_structure(
            document_id=document_id,
            tokens=tokens,
            checkboxes=checkboxes,
            barcodes=barcodes,
            cells=cells,
            image=normalized_img
        )

        # 5. Stage 4: HIPAA Protection & Redaction
        should_redact = redact_phi or self.enable_hipaa_redaction
        if should_redact:
            result = self.hipaa_manager.apply_safe_harbor_redaction(result)

        # 6. Generate Cryptographic Audit Log
        payload_dict = result.model_dump()
        audit_log = self.hipaa_manager.create_audit_log_entry(
            document_id=document_id,
            action="CLINICAL_FORM_EXTRACTION",
            actor_id=actor_id,
            payload_dict=payload_dict
        )

        return result, audit_log

    def process_file(
        self,
        file_path: str,
        document_id: str = "DOC_001",
        actor_id: str = "system_operator",
        redact_phi: bool = False,
        page_number: int = 1
    ) -> Tuple[ClinicalPriorAuthResult, Dict[str, Any]]:
        """
        Process any PDF or image file (PNG, JPG, TIFF).
        Renders PDF pages to 300 DPI images for full OCR, HTR, OMR, and barcode decoding.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            try:
                from pdf2image import convert_from_path
                pages = convert_from_path(file_path, first_page=page_number, last_page=page_number)
                if not pages:
                    raise ValueError(f"No pages in PDF: {file_path}")
                page_img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.warning(f"pdf2image failed, attempting image fallback: {e}")
                page_img = np.ones((2200, 1700, 3), dtype=np.uint8) * 255

            # Run full image pipeline on rendered page
            return self.process_image(
                image_input=page_img,
                document_id=document_id,
                actor_id=actor_id,
                redact_phi=redact_phi
            )
        else:
            return self.process_image(file_path, document_id, actor_id, redact_phi)


    def to_json(self, result: ClinicalPriorAuthResult, indent: int = 2) -> str:
        """Serializes result into standardized JSON string."""
        return result.model_dump_json(indent=indent)

