"""
Semantic Key-Value Linker & Form Extractor.
Binds OCR tokens, OMR checkboxes, table cells, and barcodes into standardized clinical entities
while handling provider-specific variations via semantic aliasing.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.schemas.clinical_schema import (
    BarcodeItem,
    BoundingBox,
    CheckboxField,
    ClinicalPriorAuthResult,
    DocumentMetadata,
    ExtractedField,
    MarkState,
    ModalityType,
    PatientInformation,
    PrescriberInformation,
    DispensingPharmacyInformation,
    MedicationAndClinicalInformation,
    AttestationAndSignature,
)

logger = logging.getLogger(__name__)


class FormExtractor:
    """
    Extracts structured clinical schemas from raw tokens, checkboxes, and table cells.
    """

    def __init__(self, alias_config_path: Optional[str] = None):
        if alias_config_path is None:
            # Default to configs/provider_aliases.json relative to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            alias_config_path = os.path.join(base_dir, "configs", "provider_aliases.json")

        self.aliases: Dict[str, List[str]] = {}
        self.provider_signatures: Dict[str, List[str]] = {}
        self._load_aliases(alias_config_path)

    def _load_aliases(self, path: str):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.aliases = data.get("aliases", {})
                    self.provider_signatures = data.get("provider_signatures", {})
            except Exception as e:
                logger.error(f"Error loading aliases from {path}: {e}")

    def identify_provider(self, tokens: List[ExtractedField]) -> Tuple[str, Optional[str]]:
        """
        Detects healthcare provider / PBM identity and form version date from header tokens.
        """
        all_text = " ".join([t.raw_text or "" for t in tokens]).lower()

        detected_provider = "Unknown Provider"
        for provider, sigs in self.provider_signatures.items():
            for sig in sigs:
                if sig.lower() in all_text:
                    detected_provider = provider
                    break
            if detected_provider != "Unknown Provider":
                break

        # Check for version date (e.g., "Updated on 05.15.2025")
        date_match = re.search(r"updated\s+on\s+([0-9]{2}[./-][0-9]{2}[./-][0-9]{4})", all_text)
        version_date = date_match.group(1) if date_match else None

        return detected_provider, version_date

    def link_and_structure(
        self,
        document_id: str,
        tokens: List[ExtractedField],
        checkboxes: List[CheckboxField],
        barcodes: List[BarcodeItem],
        cells: List[BoundingBox],
        page_number: int = 1
    ) -> ClinicalPriorAuthResult:
        """
        Synthesizes all multimodal inputs into a complete ClinicalPriorAuthResult.
        """
        provider_name, version_date = self.identify_provider(tokens)

        metadata = DocumentMetadata(
            document_id=document_id,
            form_type="Medication Prior Authorization Request Form",
            provider_name=provider_name,
            form_version_date=version_date,
            total_pages=1,
            source_dpi=300,
            hipaa_compliance_mode="operational_encrypted",
            processing_timestamp=datetime.now(timezone.utc).isoformat()
        )

        result = ClinicalPriorAuthResult(
            document_metadata=metadata,
            barcodes=barcodes
        )

        # 1. Match Checkboxes to Clinical Intent
        self._map_checkboxes(checkboxes, tokens, result)

        # 2. Extract Key-Value Pairs from Table Cells / Spatial Proximity
        self._map_key_values(tokens, cells, result)

        # 3. Detect Signature attestation
        self._detect_attestation(tokens, result)

        return result

    def _map_checkboxes(
        self,
        checkboxes: List[CheckboxField],
        tokens: List[ExtractedField],
        result: ClinicalPriorAuthResult
    ):
        """
        Associates each checkbox with its neighboring text label and maps to clinical schema fields.
        """
        for cb in checkboxes:
            cb_box = cb.bounding_box
            if not cb_box:
                continue

            # Look for adjacent label to the right of the checkbox (within reasonable horizontal distance)
            nearest_text = self._find_nearest_right_label(cb_box, tokens)
            if not nearest_text:
                continue

            cb.label_text = nearest_text.strip()
            norm_label = nearest_text.strip().lower()

            # Map to schema keys
            if self._matches_alias("expedite_review", norm_label):
                result.request_to_expedite_review = cb
            elif self._matches_alias("gender_male", norm_label):
                if cb.mark_state == MarkState.CHECKED:
                    result.patient_information.gender = ExtractedField(
                        field_key="gender",
                        raw_text="Male",
                        normalized_value="Male",
                        confidence=cb.confidence,
                        modality=ModalityType.CHECKBOX,
                        bounding_box=cb.bounding_box
                    )
            elif self._matches_alias("gender_female", norm_label):
                if cb.mark_state == MarkState.CHECKED:
                    result.patient_information.gender = ExtractedField(
                        field_key="gender",
                        raw_text="Female",
                        normalized_value="Female",
                        confidence=cb.confidence,
                        modality=ModalityType.CHECKBOX,
                        bounding_box=cb.bounding_box
                    )
            elif self._matches_alias("dispense_as_written", norm_label):
                result.medication_clinical_info.dispense_as_written = cb
            elif self._matches_alias("generic_substitution_permitted", norm_label):
                result.medication_clinical_info.generic_substitution_permitted = cb
            elif self._matches_alias("new_therapy", norm_label):
                result.medication_clinical_info.new_therapy = cb
            elif self._matches_alias("continuation_of_therapy", norm_label):
                result.medication_clinical_info.continuation_of_therapy = cb
            elif self._matches_alias("pharmacy_benefit", norm_label):
                result.medication_clinical_info.pharmacy_benefit = cb
            elif self._matches_alias("medical_benefit", norm_label):
                result.medication_clinical_info.medical_benefit = cb
            elif self._matches_alias("documentation_provided_yes", norm_label):
                result.medication_clinical_info.documentation_provided_yes = cb
            elif self._matches_alias("documentation_provided_no", norm_label):
                result.medication_clinical_info.documentation_provided_no = cb

    def _map_key_values(
        self,
        tokens: List[ExtractedField],
        cells: List[BoundingBox],
        result: ClinicalPriorAuthResult
    ):
        """
        Uses spatial proximity to map label tokens to adjacent user values.
        """
        # Group tokens into lines/phrases
        for i, t in enumerate(tokens):
            if not t.raw_text:
                continue

            text_lower = t.raw_text.lower().strip()

            # Check for label matches
            if self._matches_alias("first_name", text_lower):
                val = self._find_value_right_or_below(t, tokens)
                if val:
                    # Distinguish between patient and prescriber first name by vertical position
                    if t.bounding_box and t.bounding_box.ymin < 400:
                        result.patient_information.first_name = val
                    else:
                        result.prescriber_information.first_name = val

            elif self._matches_alias("last_name", text_lower):
                val = self._find_value_right_or_below(t, tokens)
                if val:
                    if t.bounding_box and t.bounding_box.ymin < 400:
                        result.patient_information.last_name = val
                    else:
                        result.prescriber_information.last_name = val

            elif self._matches_alias("date_of_birth", text_lower):
                val = self._find_value_right_or_below(t, tokens)
                if val:
                    result.patient_information.date_of_birth = val

            elif self._matches_alias("member_id", text_lower):
                val = self._find_value_right_or_below(t, tokens)
                if val:
                    result.patient_information.member_id = val

            elif self._matches_alias("npi_number", text_lower):
                val = self._find_value_right_or_below(t, tokens)
                if val:
                    result.prescriber_information.npi_number = val

            elif self._matches_alias("prescriber_specialty", text_lower):
                val = self._find_value_right_or_below(t, tokens)
                if val:
                    result.prescriber_information.specialty = val

    def _detect_attestation(self, tokens: List[ExtractedField], result: ClinicalPriorAuthResult):
        """Detects presence of prescriber signature line and date."""
        for t in tokens:
            if not t.raw_text:
                continue
            text = t.raw_text.lower()
            if "signature" in text or "prescriber signature" in text:
                result.attestation.signature_detected = True
                result.attestation.signature_confidence = 0.95
                result.attestation.signature_bounding_box = t.bounding_box
            if "date:" in text and t.bounding_box and t.bounding_box.ymin > 850:
                val = self._find_value_right_or_below(t, tokens)
                if val:
                    result.attestation.date_signed = val

    def _matches_alias(self, canonical_key: str, text: str) -> bool:
        """Checks if text matches any alias for the canonical key."""
        target = text.lower().replace(":", "").strip()
        aliases = self.aliases.get(canonical_key, [])
        for alias in aliases:
            if alias.lower() in target:
                return True
        return False

    def _find_nearest_right_label(
        self,
        cb_box: BoundingBox,
        tokens: List[ExtractedField],
        max_dist_x: int = 400,
        max_dist_y: int = 25
    ) -> Optional[str]:
        """Finds text immediately to the right of a checkbox."""
        candidates = []
        for t in tokens:
            tbox = t.bounding_box
            if not tbox:
                continue
            # Text must be to the right of checkbox
            if tbox.xmin >= cb_box.xmax and (tbox.xmin - cb_box.xmax) < max_dist_x:
                # Vertical centers should be roughly aligned
                cb_center_y = (cb_box.ymin + cb_box.ymax) // 2
                t_center_y = (tbox.ymin + tbox.ymax) // 2
                if abs(cb_center_y - t_center_y) < max_dist_y:
                    candidates.append((tbox.xmin - cb_box.xmax, t.raw_text))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]
        return None

    def _find_value_right_or_below(
        self,
        label_token: ExtractedField,
        all_tokens: List[ExtractedField],
        max_dist_x: int = 350
    ) -> Optional[ExtractedField]:
        """Finds non-label value token adjacent to a label."""
        lbox = label_token.bounding_box
        if not lbox:
            return None

        candidates = []
        for t in all_tokens:
            if t == label_token:
                continue
            tbox = t.bounding_box
            if not tbox:
                continue

            # Must be to the right on the same line
            if tbox.xmin > lbox.xmax and (tbox.xmin - lbox.xmax) < max_dist_x:
                l_mid = (lbox.ymin + lbox.ymax) // 2
                t_mid = (tbox.ymin + tbox.ymax) // 2
                if abs(l_mid - t_mid) < 20:
                    candidates.append((tbox.xmin - lbox.xmax, t))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            best_token = candidates[0][1]
            return ExtractedField(
                field_key=label_token.field_key + "_value",
                raw_text=best_token.raw_text,
                normalized_value=best_token.raw_text,
                modality=best_token.modality,
                confidence=best_token.confidence,
                bounding_box=best_token.bounding_box,
                page_number=best_token.page_number
            )
        return None
