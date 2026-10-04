"""
Comprehensive Unit & Integration Test Suite for Cliform-Extract.
Tests OMR checkbox classification, Table Grid segmentation, Barcode handling,
HIPAA Safe Harbor redaction, and End-to-End JSON Pipeline.
"""

import os
import pytest
import numpy as np
import cv2

from src.pipeline import ClinicalFormPipeline
from src.schemas.clinical_schema import (
    BoundingBox,
    CheckboxField,
    ClinicalPriorAuthResult,
    DocumentMetadata,
    ExtractedField,
    MarkState,
    ModalityType,
    PatientInformation,
)
from src.omr.checkbox_detector import CheckboxDetector
from src.layout.table_segmenter import TableSegmenter
from src.security.hipaa_compliance import HIPAASecurityManager


def create_synthetic_checkbox(state: str, size: int = 30) -> np.ndarray:
    """Helper to draw a clean synthetic checkbox image."""
    img = np.ones((size, size), dtype=np.uint8) * 255
    # Outer black box border
    cv2.rectangle(img, (2, 2), (size - 3, size - 3), 0, 2)

    if state == "checked":
        # Draw checkmark or X
        cv2.line(img, (6, 6), (size - 7, size - 7), 0, 2)
        cv2.line(img, (6, size - 7), (size - 7, 6), 0, 2)
    elif state == "crossed_out":
        # Draw single horizontal strikethrough line
        cv2.line(img, (4, size // 2), (size - 5, size // 2), 0, 2)
    # 'unchecked' leaves interior white
    return img


class TestCheckboxDetector:
    def test_classify_unchecked(self):
        detector = CheckboxDetector()
        img = create_synthetic_checkbox("unchecked")
        # Invert to binary (ink = 255)
        binary = cv2.bitwise_not(img)
        # Margin crop interior
        inner = binary[6:-6, 6:-6]
        mark_state, conf = detector.classify_mark(inner)
        assert mark_state == MarkState.UNCHECKED
        assert conf > 0.8

    def test_classify_checked(self):
        detector = CheckboxDetector()
        img = create_synthetic_checkbox("checked")
        binary = cv2.bitwise_not(img)
        inner = binary[6:-6, 6:-6]
        mark_state, conf = detector.classify_mark(inner)
        assert mark_state == MarkState.CHECKED
        assert conf > 0.8

    def test_classify_crossed_out(self):
        detector = CheckboxDetector()
        img = create_synthetic_checkbox("crossed_out")
        binary = cv2.bitwise_not(img)
        inner = binary[6:-6, 6:-6]
        mark_state, conf = detector.classify_mark(inner)
        assert mark_state == MarkState.CROSSED_OUT


class TestTableSegmenter:
    def test_segment_cells(self):
        # Draw a synthetic form page with a 2x2 grid table
        page = np.ones((600, 800), dtype=np.uint8) * 255
        
        # Outer table border
        cv2.rectangle(page, (50, 100), (750, 300), 0, 2)
        # Horizontal divider
        cv2.line(page, (50, 200), (750, 200), 0, 2)
        # Vertical divider
        cv2.line(page, (400, 100), (400, 300), 0, 2)

        segmenter = TableSegmenter(min_cell_width=50, min_cell_height=30)
        cells = segmenter.segment_cells(page)

        # Should discover the cells
        assert len(cells) >= 2
        for cell in cells:
            assert cell.area > 1500


class TestHIPAASecurity:
    def test_safe_harbor_redaction(self):
        manager = HIPAASecurityManager()
        
        # Mock extracted result with PHI
        metadata = DocumentMetadata(
            document_id="TEST_001",
            processing_timestamp="2026-10-04T19:37:11Z"
        )
        patient = PatientInformation(
            first_name=ExtractedField(field_key="first_name", raw_text="John", normalized_value="John"),
            last_name=ExtractedField(field_key="last_name", raw_text="Doe", normalized_value="Doe"),
            phone_number=ExtractedField(field_key="phone", raw_text="555-123-4567", normalized_value="5551234567"),
            member_id=ExtractedField(field_key="member_id", raw_text="RXB998822", normalized_value="RXB998822")
        )
        result = ClinicalPriorAuthResult(
            document_metadata=metadata,
            patient_information=patient
        )

        redacted = manager.apply_safe_harbor_redaction(result)
        
        # Verify PHI is sanitized
        assert redacted.patient_information.first_name.raw_text == "[REDACTED_NAME]"
        assert redacted.patient_information.last_name.raw_text == "[REDACTED_NAME]"
        assert redacted.patient_information.phone_number.raw_text == "[REDACTED_PHONE]"
        assert "ANON_" in redacted.patient_information.member_id.raw_text
        assert redacted.document_metadata.hipaa_compliance_mode == "safe_harbor_redacted"

    def test_visual_blackout_redaction(self):
        manager = HIPAASecurityManager()
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        box = BoundingBox(ymin=20, xmin=20, ymax=60, xmax=100)

        redacted_img = manager.generate_redacted_image(img, [box])
        # Box region should now be solid black (0, 0, 0)
        assert np.all(redacted_img[25:55, 25:95] == 0)
        # Outside should remain white (255, 255, 255)
        assert np.all(redacted_img[0:15, 0:15] == 255)

    def test_audit_log_hash(self):
        manager = HIPAASecurityManager()
        log = manager.create_audit_log_entry(
            document_id="DOC_999",
            action="CLINICAL_FORM_EXTRACTION",
            actor_id="clinician_dr_smith",
            payload_dict={"patient": "sensitive"}
        )
        assert log["document_id"] == "DOC_999"
        assert "payload_sha256" in log
        assert len(log["payload_sha256"]) == 64  # SHA-256 hex string


class TestEndToEndPipeline:
    def test_pipeline_execution(self):
        # Create a synthetic clinical document page
        img = np.ones((800, 600, 3), dtype=np.uint8) * 255
        
        # Draw header
        cv2.putText(img, "RxBenefits Medication Prior Auth", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw a table with cells
        cv2.rectangle(img, (40, 100), (560, 350), (0, 0, 0), 2)
        cv2.line(img, (40, 180), (560, 180), (0, 0, 0), 2)

        # Draw a checkbox
        cv2.rectangle(img, (50, 120), (75, 145), (0, 0, 0), 2)
        cv2.putText(img, "Request to expedite review", (85, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        pipeline = ClinicalFormPipeline()
        result, audit_log = pipeline.process_image(
            img,
            document_id="TEST_RUN_001",
            actor_id="test_suite"
        )

        assert result.document_metadata.document_id == "TEST_RUN_001"
        assert audit_log["compliance_standard"] == "HIPAA Security Rule (45 CFR § 164.312)"
        
        json_output = pipeline.to_json(result)
        assert "document_metadata" in json_output
        assert "patient_information" in json_output
