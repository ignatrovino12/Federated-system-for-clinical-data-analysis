"""Manual terminal helper that triggers a federated training request."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import sys
from contextlib import suppress

import requests


def _try_trigger_with_retries(endpoint: str, payload: dict, attempts: int, initial_delay_seconds: float) -> requests.Response:
    delay_seconds = max(0.1, initial_delay_seconds)
    response: requests.Response | None = None
    last_exception: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(endpoint, json=payload, timeout=15)
            if response.status_code >= 500 and attempt < attempts:
                print(
                    f"Attempt {attempt}/{attempts} failed with HTTP {response.status_code}; retrying in {delay_seconds:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(delay_seconds)
                delay_seconds *= 2
                continue

            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exception = exc
            if attempt >= attempts:
                raise

            print(
                f"Attempt {attempt}/{attempts} failed: {exc}; retrying in {delay_seconds:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(delay_seconds)
            delay_seconds *= 2

    raise RuntimeError(f"Trigger call failed without response (last exception: {last_exception})")


def _wait_for_local_control_api(base_url: str, timeout_seconds: float = 8.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = requests.get(base_url.rstrip("/") + "/health", timeout=1.5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.25)
    return False


def _pick_available_local_port(preferred_port: int) -> int:
    if preferred_port > 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", preferred_port))
                return preferred_port
            except OSError:
                pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _start_port_forward(namespace: str, service_name: str, local_port: int) -> subprocess.Popen:
    command = [
        "kubectl",
        "-n",
        namespace,
        "port-forward",
        f"svc/{service_name}",
        f"{local_port}:8081",
        "--address",
        "127.0.0.1",
    ]
    try:
        return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("kubectl is not installed or not available in PATH") from exc


def _ensure_port_forward_ready(port_forward: subprocess.Popen, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if port_forward.poll() is not None:
            stderr_output = ""
            with suppress(Exception):
                stderr_output = (port_forward.stderr.read() or "").strip() if port_forward.stderr else ""
            details = f" ({stderr_output})" if stderr_output else ""
            raise RuntimeError(f"kubectl port-forward exited before becoming ready{details}")
        time.sleep(0.1)


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
        "--sync-latest-model",
        dest="sync_latest_model",
        action="store_true",
        help="Ask clinics to sync the latest model from MinIO before training.",
    )
    parser.add_argument(
        "--no-sync-latest-model",
        dest="sync_latest_model",
        action="store_false",
        help="Do not ask clinics to sync the latest model before training.",
    )
    parser.add_argument(
        "--requested-by",
        default=os.getenv("USER") or os.getenv("USERNAME") or "manual-terminal",
        help="Optional user label stored in the request metadata.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=int(os.getenv("FLOWER_TRIGGER_RETRIES", "5")),
        help="How many times to retry trigger calls for transient 5xx/network errors.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=float(os.getenv("FLOWER_TRIGGER_RETRY_DELAY_SECONDS", "1.0")),
        help="Initial delay between retries, doubled after each failed attempt.",
    )
    parser.add_argument(
        "--kubectl-fallback",
        dest="kubectl_fallback",
        action="store_true",
        help="If remote trigger fails, try local kubectl port-forward to the control API.",
    )
    parser.add_argument(
        "--no-kubectl-fallback",
        dest="kubectl_fallback",
        action="store_false",
        help="Disable kubectl port-forward fallback.",
    )
    parser.add_argument(
        "--kubectl-namespace",
        default=os.getenv("FLOWER_K8S_NAMESPACE", "healthcheck"),
        help="Namespace used by kubectl fallback.",
    )
    parser.add_argument(
        "--kubectl-service",
        default=os.getenv("FLOWER_K8S_SERVICE", "flower-server"),
        help="Service name used by kubectl fallback.",
    )
    parser.add_argument(
        "--kubectl-local-port",
        type=int,
        default=int(os.getenv("FLOWER_K8S_LOCAL_CONTROL_PORT", "18081")),
        help="Local port to bind for kubectl fallback (0 = auto-select free port).",
    )
    parser.set_defaults(sync_latest_model=os.getenv("FLOWER_SYNC_LATEST_MODEL", "false").lower() == "true")
    parser.set_defaults(kubectl_fallback=os.getenv("FLOWER_KUBECTL_FALLBACK", "true").lower() == "true")
    args = parser.parse_args()

    endpoint = args.control_url.rstrip("/") + "/training/trigger"
    payload = {
        "models": [args.model],
        "server_address": args.server_address,
        "min_samples": args.min_samples,
        "test_split": args.test_split,
        "batch_size": args.batch_size,
        "model_gap_seconds": max(0, args.model_gap_seconds),
        "sync_latest_model": bool(args.sync_latest_model),
        "trigger": "manual-terminal",
        "source": "manual-command",
        "requested_by": args.requested_by,
    }

    attempts = max(1, args.retries)

    try:
        response = _try_trigger_with_retries(endpoint, payload, attempts, args.retry_delay_seconds)
    except requests.RequestException:
        if not args.kubectl_fallback:
            raise

        selected_local_port = _pick_available_local_port(args.kubectl_local_port)
        if args.kubectl_local_port > 0 and selected_local_port != args.kubectl_local_port:
            print(
                f"Requested local port {args.kubectl_local_port} is busy; using {selected_local_port} for kubectl fallback.",
                file=sys.stderr,
            )

        local_base = f"http://127.0.0.1:{selected_local_port}"
        local_endpoint = local_base + "/training/trigger"
        port_forward = _start_port_forward(args.kubectl_namespace, args.kubectl_service, selected_local_port)
        try:
            _ensure_port_forward_ready(port_forward, timeout_seconds=5.0)
            if not _wait_for_local_control_api(local_base):
                raise RuntimeError("kubectl fallback could not reach local control API")

            print(
                f"Remote control URL failed; using kubectl fallback on {local_base}.",
                file=sys.stderr,
            )
            response = _try_trigger_with_retries(local_endpoint, payload, attempts, args.retry_delay_seconds)
        finally:
            with suppress(Exception):
                port_forward.terminate()
            with suppress(Exception):
                port_forward.wait(timeout=3)

    print(json.dumps(response.json(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())