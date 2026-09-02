#!/usr/bin/env python3
"""CLI: python scripts/ingest_repo.py --repo <path> --repo-id <name>"""
import argparse
import sys
from pathlib import Path

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.core.pipeline import IngestError, run_ingestion


def main():
    parser = argparse.ArgumentParser(description="Ingest a code repository into codebase-intel.")
    parser.add_argument("--repo", required=True, help="Path to repository root")
    parser.add_argument("--repo-id", required=True, help="Identifier for this repo (e.g. 'requests')")
    args = parser.parse_args()

    try:
        summary = run_ingestion(args.repo, args.repo_id, progress=print)
    except IngestError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(
        f"\nIndexed {summary['files_indexed']} files, "
        f"{summary['chunks_indexed']} chunks, "
        f"{summary['symbols_extracted']} symbols, "
        f"{summary['edges_in_graph']} graph edges."
    )
    print(f"Embedding backend: {summary['embedding_backend']}")
    print(f"Saved to data/indexes/{args.repo_id}.index")


if __name__ == "__main__":
    main()
