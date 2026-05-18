"""Manual terminal helper that triggers a federated training request."""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description="Trigger a federated training request on the Flower control API.")
    parser.add_argument(
        "--control-url",
        default=os.getenv("FLOWER_CONTROL_URL", "http://localhost:8081"),
        help="Base URL of the Flower control API.",
    )
    parser.add_argument(
        "--model",
        choices=("alex5050", "mustafa"),
        required=True,
        help="Model family to trigger for this run.",
    )
    parser.add_argument(
        "--server-address",
        default=os.getenv("FLOWER_SERVER_ADDRESS", "flower-server:8080"),
        help="Flower server address used by the client listener.",
    )
    parser.add_argument("--min-samples", type=int, default=int(os.getenv("FLOWER_MIN_SAMPLES", "1")))
    parser.add_argument("--test-split", type=float, default=float(os.getenv("FLOWER_TEST_SPLIT", "0.2")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("FLOWER_BATCH_SIZE", "32")))
    parser.add_argument(
        "--model-gap-seconds",
        type=int,
        default=int(os.getenv("FLOWER_MODEL_GAP_SECONDS", "120")),
        help="Pause duration before starting the next model when request contains multiple models.",
    )
    parser.add_argument(
        "--no-sync-latest-model",
        action="store_true",
        help="Do not ask clients to sync the latest model before training.",
    )
    parser.add_argument(
        "--requested-by",
        default=os.getenv("USER") or os.getenv("USERNAME") or "manual-terminal",
        help="Optional user label stored in the request metadata.",
    )
    args = parser.parse_args()

    endpoint = args.control_url.rstrip("/") + "/training/trigger"
    payload = {
        "models": [args.model],
        "server_address": args.server_address,
        "min_samples": args.min_samples,
        "test_split": args.test_split,
        "batch_size": args.batch_size,
        "model_gap_seconds": max(0, args.model_gap_seconds),
        "sync_latest_model": not args.no_sync_latest_model,
        "trigger": "manual-terminal",
        "source": "manual-command",
        "requested_by": args.requested_by,
    }

    response = requests.post(endpoint, json=payload, timeout=15)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())