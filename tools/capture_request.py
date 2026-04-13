#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Send one profiled VLM request to a running vLLM server.

Wraps the request with /start_profile and /stop_profile so the trace
contains exactly one request. Supports URL and base64 image delivery.

Usage:
    # URL mode (server fetches the image):
    python tools/capture_request.py \\
        --endpoint http://localhost:8000 \\
        --model google/gemma-3-27b-it \\
        --image https://example.com/cat.jpg

    # base64 mode (image encoded and embedded in the request body):
    python tools/capture_request.py \\
        --endpoint http://localhost:8000 \\
        --model google/gemma-3-27b-it \\
        --image https://example.com/cat.jpg \\
        --base64
"""

import argparse
import json
import sys
import urllib.request as R
from urllib.error import URLError

import pybase64 as base64


def _post(
    url: str, body: bytes | None = None, content_type: str | None = None
) -> bytes:
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    req = R.Request(url, data=body, headers=headers, method="POST")
    try:
        return R.urlopen(req).read()
    except URLError as e:
        print(f"ERROR: {url} → {e}", file=sys.stderr)
        raise


def build_image_content(image_url: str, base64_mode: bool) -> dict:
    if not base64_mode:
        return {"type": "image_url", "image_url": {"url": image_url}}

    print(f"Downloading {image_url} for base64 encoding...", file=sys.stderr)
    img_bytes = R.urlopen(image_url).read()

    # Infer MIME type from URL extension; default to jpeg.
    lower = image_url.lower()
    if lower.endswith(".png"):
        mime = "image/png"
    elif lower.endswith(".webp"):
        mime = "image/webp"
    elif lower.endswith(".gif"):
        mime = "image/gif"
    else:
        mime = "image/jpeg"

    b64 = base64.b64encode(img_bytes).decode()
    print(
        f"Encoded {len(img_bytes):,} bytes → {len(b64):,} base64 chars  mime={mime}",
        file=sys.stderr,
    )
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
        help="vLLM server base URL (default: http://localhost:8000)",
    )
    parser.add_argument("--model", required=True, help="Model name as served by vLLM")
    parser.add_argument(
        "--image",
        required=True,
        metavar="URL",
        help="Image URL to send (fetched from server for URL mode, "
        "or downloaded locally and base64-encoded for --base64)",
    )
    parser.add_argument(
        "--base64",
        action="store_true",
        help="Download image locally and embed as base64 data URL",
    )
    parser.add_argument(
        "--prompt",
        default="Describe this image in one sentence.",
        help="Text prompt to send alongside the image",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    base = args.endpoint.rstrip("/")
    image_content = build_image_content(args.image, args.base64)

    payload = json.dumps(
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        image_content,
                        {"type": "text", "text": args.prompt},
                    ],
                }
            ],
            "max_tokens": args.max_tokens,
        }
    ).encode()

    print("Starting profiler...", file=sys.stderr)
    _post(f"{base}/start_profile")

    print("Sending request...", file=sys.stderr)
    resp_bytes = _post(f"{base}/v1/chat/completions", payload, "application/json")
    result = json.loads(resp_bytes)
    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    prompt_t = usage.get("prompt_tokens", 0)
    compl_t = usage.get("completion_tokens", 0)
    print(
        f"Response ({prompt_t}→{compl_t} tokens): {content[:120]}",
        file=sys.stderr,
    )

    print("Stopping profiler...", file=sys.stderr)
    _post(f"{base}/stop_profile")
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
