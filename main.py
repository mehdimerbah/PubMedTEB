"""Entry point for the PubMed XML preprocessing pipeline.

Usage:
    uv run python main.py run --xml-dir /path/to/xml --output /path/to/output.parquet
    uv run python main.py validate --parquet /path/to/output.parquet
    uv run python main.py mesh --desc-xml /path/to/desc2025.xml --output mesh_mapping.json
"""

import sys

from preprocessing.cli import main

if __name__ == "__main__":
    sys.exit(main())
