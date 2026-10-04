"""
Demonstration script for Cliform-Extract.
Runs extraction on the sample Medication Prior Authorization Request Form.
"""

import json
import os
from src.pipeline import ClinicalFormPipeline


def run_demo():
    sample_pdf = os.path.join("data", "samples", "sample_prior_auth.pdf")
    if not os.path.exists(sample_pdf):
        print(f"Sample file not found at: {sample_pdf}")
        return

    print("=================================================================")
    print(" Cliform-Extract: Clinical Form Extraction Engine Demonstration")
    print("=================================================================")
    print(f"Loading document: {sample_pdf}\n")

    pipeline = ClinicalFormPipeline()

    # 1. Operational Mode (Full Clinical Data with provenance)
    print(">>> 1. Processing in Operational Clinical Mode...")
    result, audit = pipeline.process_file(
        file_path=sample_pdf,
        document_id="RXB_DEMO_001",
        actor_id="dr_clinical_reviewer",
        redact_phi=False
    )

    print(f"[+] Form Type:        {result.document_metadata.form_type}")
    print(f"[+] Provider Detected:{result.document_metadata.provider_name}")
    print(f"[+] Form Version:     {result.document_metadata.form_version_date}")
    print(f"[+] Audit SHA-256:    {audit['payload_sha256']}")

    # Save operational JSON
    output_operational = "output_operational.json"
    with open(output_operational, "w", encoding="utf-8") as f:
        f.write(pipeline.to_json(result))
    print(f"[+] Saved operational extraction to: {output_operational}\n")

    # 2. HIPAA De-Identified Mode (Safe Harbor Redacted for Research/Analytics)
    print(">>> 2. Processing with HIPAA Safe Harbor PHI Redaction...")
    redacted_result, redacted_audit = pipeline.process_file(
        file_path=sample_pdf,
        document_id="RXB_DEMO_001_ANON",
        actor_id="research_analyst",
        redact_phi=True
    )

    print(f"[+] HIPAA Mode:       {redacted_result.document_metadata.hipaa_compliance_mode}")
    print(f"[+] Audit SHA-256:    {redacted_audit['payload_sha256']}")

    output_redacted = "output_redacted.json"
    with open(output_redacted, "w", encoding="utf-8") as f:
        f.write(pipeline.to_json(redacted_result))
    print(f"[+] Saved HIPAA-redacted extraction to: {output_redacted}\n")


    print("=================================================================")
    print(" Extraction complete! Inspect output_operational.json for results.")
    print("=================================================================")


if __name__ == "__main__":
    run_demo()
