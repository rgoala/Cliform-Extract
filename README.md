# Cliform-Extract

**Cliform-Extract** is a high-precision, HIPAA-compliant extraction engine tailored for clinical forms, questionnaires, and prior authorization requests (e.g., RxBenefits, CVS Caremark, Express Scripts, and clinical trial Case Report Forms).

It decouples deterministic perception (barcodes, checkboxes, table grid cells) from contextual understanding to eliminate hallucinations, enforce visual grounding, and adapt to healthcare provider layout variations.

---

## Key Capabilities

1. **Provider-Adaptive Layout Parsing**:
   - Uses morphological line and contour detection (`TableSegmenter`) to segment arbitrary table grid cells rather than relying on rigid coordinate templates.
   - Leverages semantic label aliasing (`configs/provider_aliases.json`) to standardize varying label names across healthcare payers and pharmacy benefit managers (PBMs).
2. **Multi-Engine Barcode Subsystem**:
   - Native support for **2D symbologies (DataMatrix ECC 200, PDF417, QR Code)** and **1D symbologies (Code 128, Code 39, EAN-13)**.
   - Morphological candidate region localization and multi-engine fallback (`zxing-cpp`, `OpenCV`, `pdf417decoder`, `pylibdmtx`, `pyzbar`).
3. **Clinical Checkbox (OMR) Detection & 4-State Classification**:
   - Detects checkboxes and classifies mark states: `CHECKED` (tick, fill, X), `UNCHECKED`, `CROSSED_OUT` (error correction / strikethrough), and `INDETERMINATE`.
4. **HIPAA & Protected Health Information (PHI) Security**:
   - Safe Harbor 18-identifier automated redaction and pseudonymization.
   - Visual blackout redaction burning solid black masks directly onto image pixel arrays for de-identified research use.
   - Cryptographic audit trail logging SHA-256 payload hashes rather than cleartext PHI.
5. **Auditable JSON Output Contract**:
   - Every extracted field provides visual provenance: bounding box coordinates `[ymin, xmin, ymax, xmax]`, page number, and confidence score.

---

## Repository Structure

```text
Cliform-Extract/
├── configs/
│   ├── pipeline_config.yaml         # Detection thresholds, DPI, and engine preferences
│   └── provider_aliases.json        # Cross-provider semantic label dictionary
├── src/
│   ├── barcode/                     # Multi-engine DataMatrix, PDF417, and 1D decoders
│   ├── extraction/                  # Key-value linking and provider identification
│   ├── layout/                      # Dynamic table grid and cell segmenter
│   ├── ocr/                         # Printed and handwritten text recognizer
│   ├── omr/                         # Checkbox detector and 4-state classifier
│   ├── preprocessing/               # Deskewing, DPI normalization, contrast enhancement
│   ├── schemas/                     # Pydantic clinical models and provenance schemas
│   ├── security/                    # HIPAA Safe Harbor redaction and audit logger
│   └── pipeline.py                  # End-to-end extraction orchestrator
├── tests/                           # Pytest integration and component test suite
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/Cliform-Extract.git
cd Cliform-Extract

# Install dependencies
pip install -r requirements.txt
```

### 2. Python Usage

```python
import cv2
from src.pipeline import ClinicalFormPipeline

# Initialize the pipeline
pipeline = ClinicalFormPipeline()

# Process a clinical form image (e.g. Prior Authorization Request)
result, audit_log = pipeline.process_image(
    image_input="path/to/clinical_form.png",
    document_id="CRF_2026_001",
    actor_id="dr_provider_user",
    redact_phi=False  # Set to True for Safe Harbor HIPAA de-identification
)

# Export validated JSON
json_output = pipeline.to_json(result)
print(json_output)
```

### 3. Running the Test Suite

```bash
python -m pytest tests/test_pipeline.py -v
```

---

## Architecture Blueprint

For the in-depth architectural design, evaluation benchmarks, dataset synthesis strategy, and multi-phase roadmap, refer to:
- [`clinical_form_extraction_plan.md`](file:///C:/Users/rahul/.gemini/antigravity/brain/b549b90f-f0b7-440b-90da-ab5ebb1edc48/clinical_form_extraction_plan.md)
