"""
FastAPI REST Service for Cliform-Extract.
Enables containerized or microservice deployment with HIPAA-compliant endpoints.
Usage:
  uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

import os
import shutil
import tempfile
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.pipeline import ClinicalFormPipeline

app = FastAPI(
    title="Cliform-Extract API",
    description="HIPAA-Compliant Clinical Form & Questionnaire Extraction Service",
    version="1.0.0"
)

# Initialize pipeline once on startup
pipeline = ClinicalFormPipeline()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Cliform-Extract",
        "version": "1.0.0",
        "engines": {
            "barcode": "multi-engine (2D/1D)",
            "omr": "4-state contour classifier",
            "layout": "morphological grid segmenter",
            "hipaa": "Safe Harbor de-identification enabled"
        }
    }


@app.post("/api/v1/extract")
async def extract_clinical_form(
    file: UploadFile = File(..., description="PDF or image of clinical form"),
    doc_id: str = Form("API_UPLOAD_001"),
    actor_id: str = Form("api_client"),
    redact_phi: bool = Form(False),
    page: int = Form(1)
):
    """
    Extracts structured JSON from uploaded clinical form / questionnaire.
    """
    suffix = os.path.splitext(file.filename)[1].lower() if file.filename else ".pdf"
    if suffix not in [".pdf", ".png", ".jpg", ".jpeg", ".tiff"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {suffix}")

    # Process securely in temporary file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, f"upload{suffix}")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result, audit_log = pipeline.process_file(
            file_path=temp_path,
            document_id=doc_id,
            actor_id=actor_id,
            redact_phi=redact_phi,
            page_number=page
        )

        return {
            "data": result.model_dump(),
            "audit": audit_log
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    finally:
        # Secure cleanup: remove temp file immediately
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
