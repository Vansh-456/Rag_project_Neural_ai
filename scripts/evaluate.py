"""Run one experiment config: python scripts/evaluate.py hybrid --split dev

Writes results/<exp_id>/{config.yaml, git_commit.txt, metrics.json, per_query.csv, failures.json}.
"""

import sys

from agvrag.cli import main

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/evaluate.py <config> [--split dev|test|all]")
    sys.exit(main(["-c", sys.argv[1], "evaluate", *sys.argv[2:]]))
