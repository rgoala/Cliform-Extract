"""
Command Line Interface (CLI) for Cliform-Extract.
Usage:
  python -m src.cli --input data/samples/sample_prior_auth.pdf --output result.json
  python -m src.cli --input data/samples/sample_prior_auth.pdf --redact-phi --output sanitized.json
"""

import argparse
import json
import os
import sys

from src.pipeline import ClinicalFormPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Cliform-Extract: Clinical Form and Prior Authorization Extraction Engine"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input clinical document (.pdf, .png, .jpg, .tiff)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to output JSON file (defaults to printing to stdout)"
    )
    parser.add_argument(
        "--doc-id",
        default="DOC_CLI_001",
        help="Custom document ID for audit logging"
    )
    parser.add_argument(
        "--actor-id",
        default="cli_user",
        help="Actor / operator identity for HIPAA audit trail"
    )
    parser.add_argument(
        "--redact-phi",
        action="store_true",
        help="Enable HIPAA Safe Harbor de-identification and masking"
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="1-indexed page number to extract (default: 1)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Processing document: {args.input}", file=sys.stderr)
    pipeline = ClinicalFormPipeline(enable_hipaa_redaction=args.redact_phi)

    try:
        result, audit_log = pipeline.process_file(
            file_path=args.input,
            document_id=args.doc_id,
            actor_id=args.actor_id,
            redact_phi=args.redact_phi,
            page_number=args.page
        )

        json_str = pipeline.to_json(result, indent=2)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"[+] Extracted JSON saved to: {args.output}", file=sys.stderr)
            print(f"[+] HIPAA Audit SHA-256: {audit_log['payload_sha256']}", file=sys.stderr)

        else:
            print(json_str)

    except Exception as e:
        print(f"[!] Processing failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
