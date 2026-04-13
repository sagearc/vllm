#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Build a Perfetto summary trace from vLLM frontend + GPU worker traces.

Extracts the key VLM pipeline spans and aligns them by wall-clock time.
Both traces use CLOCK_MONOTONIC microseconds (same machine), so timestamps
are directly comparable — no clock conversion needed.

Output tracks:
  Frontend   — media: url_download, media: pil_decode, mm_processor: process_multimodal
  GPU Worker — mm_encoder: forward, gpu_model_runner: forward

Usage:
    python tools/summary_trace.py \\
        traces/frontend.json.gz traces/rank0.json.gz traces/summary.json.gz
"""

import argparse
import gzip
import json
from pathlib import Path

FRONTEND_SPANS = {
    "media: url_download",
    "media: pil_decode",
    "mm_processor: process_multimodal",
}

GPU_SPANS = {
    "mm_encoder: forward",
    "gpu_model_runner: forward",
}

TID_FRONTEND = 1
TID_GPU = 2


def load_spans(path: Path, names: set) -> list:
    with gzip.open(path) as f:
        data = json.load(f)
    return [
        e for e in data["traceEvents"] if e.get("ph") == "X" and e.get("name") in names
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "frontend", type=Path, help="Frontend trace (async_llm.*.json.gz)"
    )
    parser.add_argument("gpu", type=Path, help="GPU worker trace (rank0.*.json.gz)")
    parser.add_argument("output", type=Path, help="Output summary trace (.json.gz)")
    args = parser.parse_args()

    frontend_spans = load_spans(args.frontend, FRONTEND_SPANS)
    gpu_spans = load_spans(args.gpu, GPU_SPANS)

    all_spans = frontend_spans + gpu_spans
    if not all_spans:
        raise SystemExit(
            "No matching spans found — check that VLLM_CUSTOM_SCOPES_FOR_PROFILING=1"
        )

    # Normalize timestamps: t=0 is the start of the earliest span.
    t0 = min(e["ts"] for e in all_spans)

    events = []
    for e in all_spans:
        tid = TID_GPU if e["name"] in GPU_SPANS else TID_FRONTEND
        events.append(
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": e["name"],
                "pid": 1,
                "tid": tid,
                "ts": e["ts"] - t0,
                "dur": e["dur"],
            }
        )

    events += [
        {
            "ph": "M",
            "pid": 1,
            "tid": TID_FRONTEND,
            "name": "thread_name",
            "args": {"name": "Frontend"},
        },
        {
            "ph": "M",
            "pid": 1,
            "tid": TID_GPU,
            "name": "thread_name",
            "args": {"name": "GPU Worker"},
        },
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt") as f:
        json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, f)

    # Print summary table
    span_events = sorted(
        (e for e in events if e.get("ph") == "X"), key=lambda e: e["ts"]
    )
    print(f"\n{'Span':<45} {'start':>9}  {'dur':>9}")
    print("-" * 67)
    for e in span_events:
        print(f"{e['name']:<45} {e['ts'] / 1000:>8.1f}ms  {e['dur'] / 1000:>8.1f}ms")
    print(f"\nWritten: {args.output}")


if __name__ == "__main__":
    main()
