"""Server-side control plane for starting federated training rounds.

The Flower server remains the aggregation engine. This module only
manages the request to start a round, persists it in MinIO, and exposes
a small HTTP control API for manual triggers and client polling.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlparse

from minio import Minio
from minio.error import S3Error


logger = logging.getLogger("flower-training-control")


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _normalize_models(models: Optional[Iterable[str]]) -> list[str]:
    normalized = []
    for model in models or ("alex5050", "mustafa"):
        cleaned = str(model).strip().lower()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized or ["alex5050", "mustafa"]


@dataclass
class TrainingRequest:
    request_id: str
    trigger: str
    source: str
    requested_by: Optional[str]
    created_at: str
    models: list[str]
    server_address: str
    min_samples: int
    test_split: float
    batch_size: int
    sync_latest_model: bool
    model_gap_seconds: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "trigger": self.trigger,
            "source": self.source,
            "requested_by": self.requested_by,
            "created_at": self.created_at,
            "models": self.models,
            "server_address": self.server_address,
            "min_samples": self.min_samples,
            "test_split": self.test_split,
            "batch_size": self.batch_size,
            "sync_latest_model": self.sync_latest_model,
            "model_gap_seconds": self.model_gap_seconds,
        }


class TrainingCoordinator:
    """Stores and publishes federated training requests."""

    def __init__(self, minio_client: Optional[Minio], bucket_name: str = "models") -> None:
        self.minio_client = minio_client
        self.bucket_name = bucket_name
        self._lock = threading.Lock()
        self._latest_request: Optional[TrainingRequest] = None
        self._object_name = os.getenv(
            "FLOWER_TRAINING_CONTROL_OBJECT",
            "control/training/latest.json",
        )

    def _ensure_bucket(self) -> None:
        if not self.minio_client:
            return

        if not self.minio_client.bucket_exists(self.bucket_name):
            self.minio_client.make_bucket(self.bucket_name)

    def _persist_request(self, request: TrainingRequest) -> None:
        if not self.minio_client:
            return

        self._ensure_bucket()
        payload = json.dumps(request.to_dict(), sort_keys=True).encode("utf-8")
        self.minio_client.put_object(
            bucket_name=self.bucket_name,
            object_name=self._object_name,
            data=io.BytesIO(payload),
            length=len(payload),
            content_type="application/json",
        )

    def _load_request_from_minio(self) -> Optional[TrainingRequest]:
        if not self.minio_client:
            return None

        try:
            response = self.minio_client.get_object(self.bucket_name, self._object_name)
            try:
                payload = json.loads(response.read().decode("utf-8"))
            finally:
                response.close()
                response.release_conn()
            return TrainingRequest(**payload)
        except S3Error:
            return None
        except Exception as exc:
            logger.warning("Could not load latest training request from MinIO: %s", exc)
            return None

    def latest_request(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._latest_request is not None:
                return self._latest_request.to_dict()

        loaded = self._load_request_from_minio()
        if loaded is None:
            return None

        with self._lock:
            self._latest_request = loaded
            return loaded.to_dict()

    def create_request(
        self,
        *,
        models: Optional[Iterable[str]] = None,
        server_address: Optional[str] = None,
        min_samples: int = 1,
        test_split: float = 0.2,
        batch_size: int = 32,
        sync_latest_model: bool = True,
        model_gap_seconds: Optional[int] = None,
        trigger: str = "manual",
        source: str = "flower-server",
        requested_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_gap_seconds = (
            int(os.getenv("FLOWER_MODEL_GAP_SECONDS", "120"))
            if model_gap_seconds is None
            else int(model_gap_seconds)
        )

        request = TrainingRequest(
            request_id=uuid.uuid4().hex,
            trigger=trigger,
            source=source,
            requested_by=requested_by,
            created_at=_utc_now_iso(),
            models=_normalize_models(models),
            server_address=server_address or os.getenv("FLOWER_SERVER_ADDRESS", "flower-server:8080"),
            min_samples=int(min_samples),
            test_split=float(test_split),
            batch_size=int(batch_size),
            sync_latest_model=bool(sync_latest_model),
            model_gap_seconds=max(0, resolved_gap_seconds),
        )

        with self._lock:
            self._latest_request = request
            self._persist_request(request)

        logger.info("Published federated training request: %s", request.to_dict())
        return request.to_dict()


class TrainingControlHandler(BaseHTTPRequestHandler):
    coordinator: Optional[TrainingCoordinator] = None

    def _write_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - - %s", self.address_string(), format % args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
            return

        if parsed.path == "/training/latest":
            if not self.coordinator:
                self._write_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "coordinator_not_ready"})
                return

            payload = self.coordinator.latest_request()
            if payload is None:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "no_training_request"})
                return

            self._write_json(HTTPStatus.OK, payload)
            return

        self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/training/trigger":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        if not self.coordinator:
            self._write_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "coordinator_not_ready"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return

        payload = self.coordinator.create_request(
            models=body.get("models"),
            server_address=body.get("server_address"),
            min_samples=body.get("min_samples", 1),
            test_split=body.get("test_split", 0.2),
            batch_size=body.get("batch_size", 32),
            sync_latest_model=body.get("sync_latest_model", True),
            model_gap_seconds=body.get("model_gap_seconds"),
            trigger=body.get("trigger", "manual"),
            source=body.get("source", "manual-command"),
            requested_by=body.get("requested_by"),
        )
        self._write_json(HTTPStatus.ACCEPTED, payload)


def start_control_server(
    coordinator: TrainingCoordinator,
    host: str = "0.0.0.0",
    port: int = 8081,
) -> ThreadingHTTPServer:
    TrainingControlHandler.coordinator = coordinator
    server = ThreadingHTTPServer((host, port), TrainingControlHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Training control API available on %s:%s", host, port)
    return server


def start_weekly_scheduler(
    coordinator: TrainingCoordinator,
    *,
    interval_seconds: Optional[int] = None,
    start_immediately: bool = False,
) -> threading.Thread:
    interval = interval_seconds or int(os.getenv("FLOWER_TRAINING_INTERVAL_SECONDS", str(7 * 24 * 60 * 60)))
    interval = max(60, interval)

    def _loop() -> None:
        if start_immediately:
            coordinator.create_request(trigger="manual", source="weekly-scheduler")

        while True:
            time.sleep(interval)
            try:
                coordinator.create_request(trigger="weekly-scheduler", source="flower-server")
            except Exception as exc:
                logger.exception("Weekly federated training trigger failed: %s", exc)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    logger.info("Weekly federated training scheduler started (interval=%ss)", interval)
    return thread
