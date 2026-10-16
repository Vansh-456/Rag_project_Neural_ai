"""Print router decision, ranked evidence (scores + sources) and optionally an answer.

Usage: python scripts/query.py "What is LoRA?" [-c adaptive] [--answer] [--verify]
Equivalent to ``agvrag [-c CONFIG] query "..."``.
"""

import sys

from agvrag.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    config = []
    if "-c" in argv:
        i = argv.index("-c")
        config = ["-c", argv[i + 1]]
        argv = argv[:i] + argv[i + 2 :]
    sys.exit(main([*config, "query", *argv]))
