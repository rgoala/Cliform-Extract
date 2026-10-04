"""
Fine-tune specific label tokens for Cliform-Extract.
"""
from datetime import datetime, timezone
import re
from typing import Dict, List, Optional, Tuple

import cv2
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
from src.omr.checkbox_detector import CheckboxDetector

# Exclude these text tokens from being returned as values
LABEL_KEYWORDS = {
    "patient", "information", "prescriber", "dispensing", "pharmacy", "medication",
    "medical", "first", "name:", "last", "phone", "number:", "address:", "city:",
    "state:", "zip", "code:", "date", "birth:", "bith:", "member", "id:", "specialty:",
    "npi", "(individual):", "fax", "(in", "hipaa", "compliant", "area):",
    "name", "strength:", "directions", "use:", "quantity:", "day", "supply:",
    "duration", "therapy:", "icd", "10", "codes(s)", "diagnosis", "height",
    "weight", "(in/cm):", "(in/em):", "(lb/kg):", "benefit", "signature:",
    "yes", "no", "male", "female", "dispense", "written", "generic", "substitution",
    "permitted*", "permitted", "representative", "authorized", "representative:",
    "(if", "applicable):"
}


class FormExtractor:
    def __init__(self, alias_config_path: Optional[str] = None):
        self.checkbox_detector = CheckboxDetector()

    def identify_provider(self, tokens: List[ExtractedField]) -> Tuple[str, Optional[str]]:
        all_text = " ".join([t.raw_text or "" for t in tokens]).lower()
        provider = "RxBenefits" if "rxbenefits" in all_text or "promptpa" in all_text else "Unknown Provider"
        date_match = re.search(r"updated\s+on\s+([0-9]{2}[./-][0-9]{2}[./-][0-9]{4})", all_text)
        version_date = date_match.group(1) if date_match else "05.15.2025"
        return provider, version_date

    def link_and_structure(
        self,
        document_id: str,
        tokens: List[ExtractedField],
        checkboxes: List[CheckboxField],
        barcodes: List[BarcodeItem],
        cells: List[BoundingBox],
        image: Optional[np.ndarray] = None,
        page_number: int = 1
    ) -> ClinicalPriorAuthResult:
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

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if (image is not None and len(image.shape) == 3) else image
        sections = self._discover_sections(tokens)

        self._extract_checkboxes(tokens, gray, sections, result)
        self._extract_fields(tokens, sections, result)

        return result

    def _discover_sections(self, tokens: List[ExtractedField]) -> Dict[str, int]:
        y_pat, y_presc, y_pharm, y_med, y_attest = 0, 0, 0, 0, 0
        sorted_tokens = sorted(tokens, key=lambda t: t.bounding_box.ymin if t.bounding_box else 0)

        for t in sorted_tokens:
            if not t.raw_text or not t.bounding_box:
                continue
            txt = t.raw_text.strip()
            x = t.bounding_box.xmin
            y = t.bounding_box.ymin

            if txt == "Patient" and y_pat == 0 and x > 400:
                y_pat = y
            elif txt == "Prescriber" and y_presc == 0 and x > 400 and y > y_pat:
                y_presc = y
            elif txt == "Dispensing" and y_pharm == 0 and y > y_presc:
                y_pharm = y
            elif txt == "Medication" and y_med == 0 and x > 400 and y > y_pharm:
                y_med = y
            elif txt == "Prescriber" and y_attest == 0 and y > y_med and y_med > 0:
                y_attest = y

        max_y = max([t.bounding_box.ymax for t in tokens if t.bounding_box] or [2200])
        if y_pat == 0: y_pat = int(max_y * 0.24)
        if y_presc == 0: y_presc = int(max_y * 0.40)
        if y_pharm == 0: y_pharm = int(max_y * 0.54)
        if y_med == 0: y_med = int(max_y * 0.59)
        if y_attest == 0: y_attest = int(max_y * 0.86)

        return {"patient": y_pat, "prescriber": y_presc, "pharmacy": y_pharm, "medication": y_med, "attestation": y_attest}

    def _extract_checkboxes(
        self,
        tokens: List[ExtractedField],
        gray: Optional[np.ndarray],
        sections: Dict[str, int],
        result: ClinicalPriorAuthResult
    ):
        def check_left(label_token: ExtractedField, offset_x=22, box_size=14) -> Tuple[MarkState, float]:
            if gray is None or not label_token.bounding_box:
                return MarkState.INDETERMINATE, 0.5
            box = label_token.bounding_box
            cx = box.xmin - offset_x
            cy = (box.ymin + box.ymax) // 2
            return self.checkbox_detector.test_mark_at(gray, cx, cy, box_size=box_size)

        for t in tokens:
            if not t.raw_text or not t.bounding_box:
                continue
            txt = t.raw_text.strip()
            y = t.bounding_box.ymin

            # Expedite review
            if "expedite" in txt.lower() and y < sections["patient"]:
                state, conf = check_left(t, offset_x=45)
                result.request_to_expedite_review = CheckboxField(
                    field_key="expedite_review",
                    label_text="Request to expedite review",
                    mark_state=state,
                    confidence=conf,
                    bounding_box=t.bounding_box
                )

            # Gender Male / Female
            elif txt == "Male" and sections["patient"] <= y < sections["prescriber"]:
                state, conf = check_left(t, offset_x=20)
                if state == MarkState.CHECKED:
                    result.patient_information.gender = ExtractedField(
                        field_key="gender", raw_text="Male", normalized_value="Male",
                        confidence=conf, modality=ModalityType.CHECKBOX, bounding_box=t.bounding_box
                    )
            elif txt == "Female" and sections["patient"] <= y < sections["prescriber"]:
                state, conf = check_left(t, offset_x=20)
                if state == MarkState.CHECKED and result.patient_information.gender is None:
                    result.patient_information.gender = ExtractedField(
                        field_key="gender", raw_text="Female", normalized_value="Female",
                        confidence=conf, modality=ModalityType.CHECKBOX, bounding_box=t.bounding_box
                    )

            # Dispense as written
            elif txt == "Dispense" and sections["medication"] <= y < sections["attestation"]:
                state, conf = check_left(t, offset_x=22)
                result.medication_clinical_info.dispense_as_written = CheckboxField(
                    field_key="dispense_as_written", label_text="Dispense as written",
                    mark_state=state, confidence=conf, bounding_box=t.bounding_box
                )

            # Generic substitution permitted
            elif "Generic" in txt and sections["medication"] <= y < sections["attestation"]:
                state, conf = check_left(t, offset_x=22)
                result.medication_clinical_info.generic_substitution_permitted = CheckboxField(
                    field_key="generic_substitution_permitted", label_text="Generic substitution permitted*",
                    mark_state=state, confidence=conf, bounding_box=t.bounding_box
                )

            # Pharmacy Benefit (checkbox is to the left of Pharmacy)
            elif txt == "Pharmacy" and sections["medication"] <= y < (sections["medication"] + 450) and t.bounding_box.xmin > 1000:
                state, conf = check_left(t, offset_x=22)
                result.medication_clinical_info.pharmacy_benefit = CheckboxField(
                    field_key="pharmacy_benefit", label_text="Pharmacy Benefit",
                    mark_state=state, confidence=conf, bounding_box=t.bounding_box
                )

            # Medical Benefit (checkbox is to the left of Medical)
            elif txt == "Medical" and sections["medication"] <= y < (sections["medication"] + 550) and t.bounding_box.xmin > 1000:
                state, conf = check_left(t, offset_x=22)
                result.medication_clinical_info.medical_benefit = CheckboxField(
                    field_key="medical_benefit", label_text="Medical Benefit",
                    mark_state=state, confidence=conf, bounding_box=t.bounding_box
                )

            # Documentation provided Yes / No
            elif txt == "Yes" and y >= (sections["attestation"] - 140):
                state, conf = check_left(t, offset_x=22)
                result.medication_clinical_info.documentation_provided_yes = CheckboxField(
                    field_key="documentation_provided_yes", label_text="Yes",
                    mark_state=state, confidence=conf, bounding_box=t.bounding_box
                )
            elif "No" in txt and y >= (sections["attestation"] - 140) and t.bounding_box.xmin > 900:
                state, conf = check_left(t, offset_x=22)
                result.medication_clinical_info.documentation_provided_no = CheckboxField(
                    field_key="documentation_provided_no", label_text="No",
                    mark_state=state, confidence=conf, bounding_box=t.bounding_box
                )

    def _extract_fields(self, tokens: List[ExtractedField], sections: Dict[str, int], result: ClinicalPriorAuthResult):
        def get_below(label_token: ExtractedField, max_dx=90, min_dy=12, max_dy=70) -> Optional[ExtractedField]:
            lbox = label_token.bounding_box
            if not lbox:
                return None
            cands = []
            for t in tokens:
                if not t.raw_text or not t.bounding_box:
                    continue
                tbox = t.bounding_box
                dy = tbox.ymin - lbox.ymin
                dx = abs(tbox.xmin - lbox.xmin)
                if min_dy <= dy <= max_dy and dx <= max_dx:
                    raw_lower = t.raw_text.lower().strip()
                    if raw_lower not in LABEL_KEYWORDS:
                        cands.append(t)
            if cands:
                cands.sort(key=lambda x: x.bounding_box.xmin)
                val_text = " ".join([c.raw_text for c in cands]).strip()
                bbox = BoundingBox(
                    ymin=min(c.bounding_box.ymin for c in cands),
                    xmin=min(c.bounding_box.xmin for c in cands),
                    ymax=max(c.bounding_box.ymax for c in cands),
                    xmax=max(c.bounding_box.xmax for c in cands)
                )
                return ExtractedField(
                    field_key=label_token.field_key + "_val", raw_text=val_text, normalized_value=val_text,
                    confidence=round(sum(c.confidence for c in cands) / len(cands), 3), bounding_box=bbox
                )
            return None

        def get_right(label_token: ExtractedField, max_dx=350, max_dy=25) -> Optional[ExtractedField]:
            lbox = label_token.bounding_box
            if not lbox:
                return None
            cands = []
            for t in tokens:
                if not t.raw_text or not t.bounding_box:
                    continue
                tbox = t.bounding_box
                if tbox.xmin > lbox.xmax and (tbox.xmin - lbox.xmax) < max_dx:
                    if abs(tbox.ymin - lbox.ymin) < max_dy:
                        raw_lower = t.raw_text.lower().strip()
                        if raw_lower not in LABEL_KEYWORDS and raw_lower not in ["male", "female", "yes", "no"]:
                            cands.append(t)
            if cands:
                cands.sort(key=lambda x: x.bounding_box.xmin)
                val_text = " ".join([c.raw_text for c in cands]).strip()
                bbox = BoundingBox(
                    ymin=min(c.bounding_box.ymin for c in cands),
                    xmin=min(c.bounding_box.xmin for c in cands),
                    ymax=max(c.bounding_box.ymax for c in cands),
                    xmax=max(c.bounding_box.xmax for c in cands)
                )
                return ExtractedField(
                    field_key=label_token.field_key + "_val", raw_text=val_text, normalized_value=val_text,
                    confidence=round(sum(c.confidence for c in cands) / len(cands), 3), bounding_box=bbox
                )
            return None

        for t in tokens:
            if not t.raw_text or not t.bounding_box:
                continue
            txt = t.raw_text.strip()
            y = t.bounding_box.ymin

            # 1. PATIENT SECTION
            if sections["patient"] <= y < sections["prescriber"]:
                if "First" in txt and result.patient_information.first_name is None:
                    result.patient_information.first_name = get_below(t)
                elif "Last" in txt and result.patient_information.last_name is None:
                    result.patient_information.last_name = get_below(t)
                elif "Phone" in txt and result.patient_information.phone_number is None:
                    result.patient_information.phone_number = get_below(t)
                elif "Address:" in txt and result.patient_information.address is None:
                    result.patient_information.address = get_below(t, max_dx=140)
                elif "City:" in txt and result.patient_information.city is None:
                    result.patient_information.city = get_below(t)
                elif "State:" in txt and result.patient_information.state is None:
                    result.patient_information.state = get_below(t)
                elif "Zip" in txt and result.patient_information.zip_code is None:
                    result.patient_information.zip_code = get_below(t)
                elif ("Date" in txt or "Birth" in txt or "Bith" in txt) and result.patient_information.date_of_birth is None:
                    dob = get_right(t, max_dx=220)
                    if dob:
                        clean_dob = re.sub(r"[^0-9/]", "", dob.raw_text)
                        dob.normalized_value = clean_dob if clean_dob else dob.raw_text
                        result.patient_information.date_of_birth = dob
                elif "Member" in txt and result.patient_information.member_id is None:
                    result.patient_information.member_id = get_right(t, max_dx=350)
                elif ("Representative" in txt or "Authorized" in txt) and t.bounding_box.xmin < 500 and result.patient_information.authorized_rep_name is None:
                    result.patient_information.authorized_rep_name = get_below(t, max_dx=180)
                elif "Number:" in txt and t.bounding_box.xmin > 700 and result.patient_information.authorized_rep_phone is None:
                    result.patient_information.authorized_rep_phone = get_below(t, max_dx=180)

            # 2. PRESCRIBER SECTION
            elif sections["prescriber"] <= y < sections["pharmacy"]:
                if "First" in txt and result.prescriber_information.first_name is None:
                    result.prescriber_information.first_name = get_below(t)
                elif "Last" in txt and result.prescriber_information.last_name is None:
                    result.prescriber_information.last_name = get_below(t)
                elif "Specialty:" in txt and result.prescriber_information.specialty is None:
                    result.prescriber_information.specialty = get_below(t)
                elif "Address:" in txt and result.prescriber_information.address is None:
                    result.prescriber_information.address = get_below(t, max_dx=140)
                elif "City:" in txt and result.prescriber_information.city is None:
                    result.prescriber_information.city = get_below(t)
                elif "State:" in txt and result.prescriber_information.state is None:
                    result.prescriber_information.state = get_below(t)
                elif "Zip" in txt and result.prescriber_information.zip_code is None:
                    result.prescriber_information.zip_code = get_below(t)
                elif "(individual):" in txt and result.prescriber_information.npi_number is None:
                    result.prescriber_information.npi_number = get_right(t, max_dx=250)
                elif "Phone" in txt and result.prescriber_information.phone_number is None:
                    result.prescriber_information.phone_number = get_right(t, max_dx=250)
                elif "area):" in txt and result.prescriber_information.fax_number is None:
                    result.prescriber_information.fax_number = get_right(t, max_dx=250)

            # 3. PHARMACY SECTION
            elif sections["pharmacy"] <= y < sections["medication"]:
                if "Name:" in txt and t.bounding_box.xmin < 300 and result.dispensing_pharmacy.pharmacy_name is None:
                    result.dispensing_pharmacy.pharmacy_name = get_right(t, max_dx=350)
                elif "area):" in txt and result.dispensing_pharmacy.pharmacy_fax_number is None:
                    result.dispensing_pharmacy.pharmacy_fax_number = get_right(t, max_dx=350)

            # 4. MEDICATION SECTION
            elif sections["medication"] <= y < sections["attestation"]:
                if txt == "Medication" and t.bounding_box.xmin < 200 and result.medication_clinical_info.medication_name_and_strength is None:
                    result.medication_clinical_info.medication_name_and_strength = get_below(t, max_dx=120)
                elif txt == "Directions" and result.medication_clinical_info.directions_for_use is None:
                    result.medication_clinical_info.directions_for_use = get_below(t, max_dx=180)
                elif "Quantity" in txt and result.medication_clinical_info.quantity is None:
                    result.medication_clinical_info.quantity = get_below(t, max_dx=80)
                elif "Supply" in txt and result.medication_clinical_info.day_supply is None:
                    result.medication_clinical_info.day_supply = get_below(t, max_dx=80)
                elif "ICD" in txt and result.medication_clinical_info.icd10_codes_and_diagnosis is None:
                    icd_val = get_below(t, max_dx=80)
                    if icd_val and icd_val.raw_text:
                        icd_val.normalized_value = icd_val.raw_text.replace(" ", ".")
                    result.medication_clinical_info.icd10_codes_and_diagnosis = icd_val
                elif ("(in/cm):" in txt or "(in/em):" in txt) and result.medication_clinical_info.patient_height is None:
                    h_val = get_right(t, max_dx=200)
                    if h_val:
                        num = re.search(r"([0-9]{2,3})", h_val.raw_text)
                        if num:
                            h_val.normalized_value = int(num.group(1))
                        result.medication_clinical_info.patient_height = h_val
                elif "(lb/kg):" in txt and result.medication_clinical_info.patient_weight is None:
                    w_val = get_right(t, max_dx=200)
                    if w_val:
                        num = re.search(r"([0-9]{2,3})", w_val.raw_text)
                        if num:
                            w_val.normalized_value = int(num.group(1))
                        result.medication_clinical_info.patient_weight = w_val


            # 5. ATTESTATION
            elif y >= sections["attestation"]:
                if "Signature:" in txt:
                    result.attestation.signature_detected = True
                    result.attestation.signature_confidence = 0.95
                    result.attestation.signature_bounding_box = t.bounding_box
                elif "Date:" in txt and t.bounding_box.xmin > 800:
                    result.attestation.date_signed = get_right(t, max_dx=200)
