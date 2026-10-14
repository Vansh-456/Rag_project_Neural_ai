"""Download the arXiv corpus listed in ``data/corpus/papers.yaml`` (rate-limited, idempotent).

Usage: python scripts/download_papers.py [--config base] [--limit N]
Equivalent to ``agvrag download``.
"""

import sys

from agvrag.cli import main

if __name__ == "__main__":
    sys.exit(main(["download", *sys.argv[1:]]))
