#!/usr/bin/env bash
# Reproduce every number in docs/ablation_report.md and docs/experiments.md from scratch.
# CPU-only runtime on a 12-core laptop: ~1.5 h for corpus+index+graph+ablation, + ~2 h for the sweep.
set -euo pipefail
agvrag download                       # 79 arXiv PDFs, title-verified
agvrag parse --force --compare        # parse report + layout-vs-plain parser comparison
agvrag chunk-sweep                    # Phase 3 chunk statistics
agvrag index                          # base index (structure_aware 256/32, bge-small)
agvrag graph                          # rule-based knowledge graph + stats + audit sample
agvrag coverage                       # every label must map to >= 1 chunk
agvrag ablate --split dev --no-oracle # development comparison
agvrag ablate --split test            # FINAL: frozen test split + router oracle -> docs/ablation_report.md
agvrag sweep --split dev              # Phase 5 retrieval grid (chunker x size x embedder)
