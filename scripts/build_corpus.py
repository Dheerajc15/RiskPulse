"""
Build the FOMC corpus by scraping federalreserve.gov.

"""

import argparse
import logging
import sys
from pathlib import Path

# Make core importable when running this as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.document_loader import build_fomc_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FOMC minutes corpus.")
    parser.add_argument("--years", type=int, default=5, help="Years back to scrape (default 5)")
    parser.add_argument("--output-dir", default="data/documents/fomc")
    parser.add_argument("--manifest", default="data/documents/corpus_manifest.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    report = build_fomc_corpus(
        years_back=args.years,
        output_dir=Path(args.output_dir),
        manifest_path=Path(args.manifest),
    )

    print("\n" + "=" * 60)
    print("Corpus build complete")
    print("=" * 60)
    print(f"Meetings attempted:  {report.total_meetings_attempted}")
    print(f"Documents fetched:   {report.documents_fetched}")
    print(f"Documents rejected:  {report.documents_rejected}")
    if report.rejection_reasons:
        print("Rejection reasons:")
        for reason, count in report.rejection_reasons.items():
            print(f"  - {reason}: {count}")
    print(f"\nOutput dir: {report.output_dir}")
    print(f"Manifest:   {report.manifest_path}")

    return 0 if report.documents_fetched > 0 else 1


if __name__ == "__main__":
    sys.exit(main())