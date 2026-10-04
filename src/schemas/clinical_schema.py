"""
Pydantic schemas for clinical form data extraction, visual grounding, and HIPAA compliance.
Conforms to prior authorization forms (e.g., RxBenefits), patient intake, and clinical CRFs.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator


class ModalityType(str, Enum):
    PRINTED = "printed"
    HANDWRITTEN = "handwritten"
    CHECKBOX = "checkbox"
    BARCODE = "barcode"
    SIGNATURE = "signature"
    STAMP = "stamp"


class MarkState(str, Enum):
    CHECKED = "checked"
    UNCHECKED = "unchecked"
    CROSSED_OUT = "crossed_out"
    INDETERMINATE = "indeterminate"


class BarcodeSymbology(str, Enum):
    DATAMATRIX = "DataMatrix"
    PDF417 = "PDF417"
    QR_CODE = "QRCode"
    CODE_128 = "Code128"
    CODE_39 = "Code39"
    EAN_13 = "EAN13"
    UNKNOWN = "Unknown"


class BoundingBox(BaseModel):
    """
    Standard bounding box coordinates in pixels or normalized 0-1000 range.
    [ymin, xmin, ymax, xmax] format.
    """
    ymin: int = Field(..., description="Top coordinate")
    xmin: int = Field(..., description="Left coordinate")
    ymax: int = Field(..., description="Bottom coordinate")
    xmax: int = Field(..., description="Right coordinate")

    @property
    def width(self) -> int:
        return max(0, self.xmax - self.xmin)

    @property
    def height(self) -> int:
        return max(0, self.ymax - self.ymin)

    @property
    def area(self) -> int:
        return self.width * self.height


class ExtractedField(BaseModel):
    """
    Individual extracted data field with full visual provenance and confidence score.
    """
    field_key: str = Field(..., description="Canonical field identifier")
    label_text: Optional[str] = Field(None, description="Pre-printed form label")
    raw_text: Optional[str] = Field(None, description="Exact OCR/HTR transcribed text")
    normalized_value: Optional[Union[str, int, float, bool]] = Field(None, description="Validated typed value")
    modality: ModalityType = Field(default=ModalityType.PRINTED, description="Extraction modality")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score (0-1)")
    bounding_box: Optional[BoundingBox] = Field(None, description="Visual grounding coordinates on page")
    page_number: int = Field(default=1, description="1-indexed page number")
    is_redacted: bool = Field(default=False, description="Whether this field has been sanitized under HIPAA")


class CheckboxField(BaseModel):
    """
    Checkbox or radio button element with detected mark state.
    """
    field_key: str = Field(..., description="Field key or associated label")
    label_text: Optional[str] = Field(None, description="Accompanying text label")
    mark_state: MarkState = Field(..., description="Detected mark state")
    is_checked: bool = Field(default=False, description="Convenience boolean (True if CHECKED)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    bounding_box: Optional[BoundingBox] = None
    page_number: int = 1

    @model_validator(mode="after")
    def sync_is_checked(self):
        self.is_checked = (self.mark_state == MarkState.CHECKED)
        return self


class BarcodeItem(BaseModel):
    """
    Extracted barcode item (1D or 2D).
    """
    symbology: BarcodeSymbology = Field(default=BarcodeSymbology.UNKNOWN)
    raw_value: str = Field(..., description="Decoded string/payload")
    is_valid_checksum: bool = Field(default=True, description="Whether CRC/checksum passed")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    bounding_box: Optional[BoundingBox] = None
    page_number: int = 1


class DocumentMetadata(BaseModel):
    """
    Document identification and audit tracking.
    """
    document_id: str = Field(..., description="Unique processing run or file identifier")
    form_type: str = Field(default="Clinical Form / Questionnaire")
    provider_name: Optional[str] = Field(None, description="Detected or specified healthcare provider/PBM")
    form_version_date: Optional[str] = Field(None, description="Version date printed on form, e.g. 05.15.2025")
    total_pages: int = Field(default=1, ge=1)
    source_dpi: int = Field(default=300)
    hipaa_compliance_mode: str = Field(default="operational_encrypted")
    processing_timestamp: str = Field(...)


class PatientInformation(BaseModel):
    """
    Patient demographics block.
    """
    first_name: Optional[ExtractedField] = None
    last_name: Optional[ExtractedField] = None
    phone_number: Optional[ExtractedField] = None
    address: Optional[ExtractedField] = None
    city: Optional[ExtractedField] = None
    state: Optional[ExtractedField] = None
    zip_code: Optional[ExtractedField] = None
    date_of_birth: Optional[ExtractedField] = None
    gender: Optional[ExtractedField] = None
    member_id: Optional[ExtractedField] = None
    authorized_rep_name: Optional[ExtractedField] = None
    authorized_rep_phone: Optional[ExtractedField] = None


class PrescriberInformation(BaseModel):
    """
    Prescribing physician / provider information block.
    """
    first_name: Optional[ExtractedField] = None
    last_name: Optional[ExtractedField] = None
    specialty: Optional[ExtractedField] = None
    address: Optional[ExtractedField] = None
    city: Optional[ExtractedField] = None
    state: Optional[ExtractedField] = None
    zip_code: Optional[ExtractedField] = None
    npi_number: Optional[ExtractedField] = None
    phone_number: Optional[ExtractedField] = None
    fax_number: Optional[ExtractedField] = None


class DispensingPharmacyInformation(BaseModel):
    """
    Pharmacy information block.
    """
    pharmacy_name: Optional[ExtractedField] = None
    pharmacy_fax_number: Optional[ExtractedField] = None


class MedicationAndClinicalInformation(BaseModel):
    """
    Medication, diagnosis, and medical necessity details.
    """
    medication_name_and_strength: Optional[ExtractedField] = None
    dispense_as_written: Optional[CheckboxField] = None
    generic_substitution_permitted: Optional[CheckboxField] = None
    directions_for_use: Optional[ExtractedField] = None
    quantity: Optional[ExtractedField] = None
    day_supply: Optional[ExtractedField] = None
    new_therapy: Optional[CheckboxField] = None
    continuation_of_therapy: Optional[CheckboxField] = None
    continuation_start_date: Optional[ExtractedField] = None
    duration_of_therapy: Optional[ExtractedField] = None
    icd10_codes_and_diagnosis: Optional[ExtractedField] = None
    patient_height: Optional[ExtractedField] = None
    patient_weight: Optional[ExtractedField] = None
    pharmacy_benefit: Optional[CheckboxField] = None
    medical_benefit: Optional[CheckboxField] = None
    previously_failed_medications: Optional[ExtractedField] = None
    documentation_provided_yes: Optional[CheckboxField] = None
    documentation_provided_no: Optional[CheckboxField] = None


class AttestationAndSignature(BaseModel):
    """
    Signature and date attestation block.
    """
    signature_detected: bool = Field(default=False)
    signature_modality: ModalityType = Field(default=ModalityType.SIGNATURE)
    signature_confidence: float = Field(default=0.0)
    signature_bounding_box: Optional[BoundingBox] = None
    date_signed: Optional[ExtractedField] = None


class ClinicalPriorAuthResult(BaseModel):
    """
    Complete structured extraction output conforming to Medication Prior Authorization forms.
    """
    document_metadata: DocumentMetadata
    barcodes: List[BarcodeItem] = Field(default_factory=list)
    request_date: Optional[ExtractedField] = None
    request_to_expedite_review: Optional[CheckboxField] = None
    patient_information: PatientInformation = Field(default_factory=PatientInformation)
    prescriber_information: PrescriberInformation = Field(default_factory=PrescriberInformation)
    dispensing_pharmacy: DispensingPharmacyInformation = Field(default_factory=DispensingPharmacyInformation)
    medication_clinical_info: MedicationAndClinicalInformation = Field(default_factory=MedicationAndClinicalInformation)
    attestation: AttestationAndSignature = Field(default_factory=AttestationAndSignature)
    raw_unmapped_fields: List[ExtractedField] = Field(default_factory=list)
    validation_status: Dict[str, Any] = Field(default_factory=dict)
