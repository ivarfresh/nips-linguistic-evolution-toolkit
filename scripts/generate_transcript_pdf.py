#!/usr/bin/env python3
"""Generate PDF transcripts from saved simulation state JSON files."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.transcript import write_pdf_transcript_from_state_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate full PDF transcript(s) from saved simulation JSON state."
    )
    parser.add_argument(
        "state_json",
        nargs="+",
        help="Saved .json, .checkpoint.json, or .checkpoint.json.error.json state file(s).",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output PDF path. Only valid when generating one transcript.",
    )
    args = parser.parse_args()

    if args.output and len(args.state_json) != 1:
        parser.error("--output can only be used with one input file")

    for state_path in args.state_json:
        output_path = args.output if args.output else None
        pdf_path = write_pdf_transcript_from_state_file(state_path, output_path)
        print(f"Wrote {pdf_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
