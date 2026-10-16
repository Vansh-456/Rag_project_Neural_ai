"""Small concurrent load test for the API.

Usage: python scripts/load_test.py --url http://localhost:8000 --n 100 --concurrency 8 \
           [--pipeline adaptive] [--user admin --password ...]
Reports throughput, error count and latency percentiles. Queries are drawn from the evaluation
set; repeated queries exercise the result cache (see /metrics for hit rates).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--pipeline", default="adaptive")
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    queries = [
        q["text"] for q in json.loads((ROOT / "data/evaluation/queries.json").read_text("utf-8"))
    ]
    rng = random.Random(args.seed)
    auth = (args.user, args.password) if args.user and args.password else None
    client = httpx.Client(base_url=args.url, timeout=120, auth=auth)

    def one(_: int) -> tuple[float, int]:
        t0 = time.perf_counter()
        r = client.post("/query", json={"query": rng.choice(queries), "pipeline": args.pipeline})
        return (time.perf_counter() - t0) * 1000, r.status_code

    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(one, range(args.n)))
    wall = time.perf_counter() - t0
    lat = sorted(ms for ms, code in results if code == 200)
    errors = [code for _, code in results if code != 200]

    def pct(p: float) -> float:
        return lat[min(len(lat) - 1, int(p * len(lat)))] if lat else 0.0

    print(
        json.dumps(
            {
                "requests": args.n,
                "concurrency": args.concurrency,
                "ok": len(lat),
                "errors": len(errors),
                "error_codes": sorted(set(errors)),
                "throughput_rps": round(args.n / wall, 2),
                "latency_ms": {
                    "p50": round(pct(0.5), 1),
                    "p95": round(pct(0.95), 1),
                    "max": round(lat[-1], 1) if lat else 0,
                    "mean": round(statistics.fmean(lat), 1) if lat else 0,
                },
                "server_metrics": client.get("/metrics").json()
                if not errors or 401 not in errors
                else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
