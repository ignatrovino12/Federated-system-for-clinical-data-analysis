from __future__ import annotations

import io
import ipaddress
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
from cryptography.x509.oid import NameOID
from flwr.common import Metrics, Parameters, parameters_to_ndarrays
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from minio import Minio
from minio.error import S3Error
from prometheus_client import start_http_server

from metrics import (
    record_flower_checkpoint_failure,
    record_flower_checkpoint_saved,
    record_flower_round,
)

try:
    from training_coordinator import TrainingCoordinator, start_control_server, start_weekly_scheduler
except ImportError:
    from flower_server.training_coordinator import (  # type: ignore
        TrainingCoordinator,
        start_control_server,
        start_weekly_scheduler,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("flower-server")


def _resolve_port() -> int:
    candidates = [
        os.getenv("FLOWER_PORT"),
        os.getenv("FLOWER_SERVER_PORT"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        value = str(candidate).strip()
        # Kubernetes can inject values like tcp://10.43.89.20:8080 for SERVICE_PORT vars.
        if value.startswith("tcp://") and ":" in value:
            value = value.rsplit(":", 1)[-1]
        try:
            return int(value)
        except ValueError:
            continue
    return 8080


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}

    accuracies = [num_examples * float(metric_map.get("accuracy", 0.0)) for num_examples, metric_map in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}
    return {"accuracy": sum(accuracies) / total_examples}


def weighted_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}

    weighted: Dict[str, float] = {}
    totals: Dict[str, float] = {}
    for num_examples, metric_map in metrics:
        for metric_name, metric_value in metric_map.items():
            if not isinstance(metric_value, (int, float)):
                continue
            weighted[metric_name] = weighted.get(metric_name, 0.0) + float(num_examples) * float(metric_value)
            totals[metric_name] = totals.get(metric_name, 0.0) + float(num_examples)

    aggregated: Dict[str, float] = {}
    for metric_name, weighted_sum in weighted.items():
        total_examples = totals.get(metric_name, 0.0)
        if total_examples > 0:
            aggregated[metric_name] = weighted_sum / total_examples
    return aggregated


def _generate_self_signed_cert(common_name: str) -> Tuple[bytes, bytes, bytes]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow())
        .not_valid_after(datetime.utcnow().replace(year=datetime.utcnow().year + 2))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName(common_name),
                    x509.DNSName("localhost"),
                    x509.DNSName("flower-server"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )
    cert_bytes = cert.public_bytes(Encoding.PEM)
    key_bytes = private_key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption())
    return cert_bytes, cert_bytes, key_bytes


def _load_tls_certificates() -> Optional[Tuple[bytes, bytes, bytes]]:
    enabled = os.getenv("FLOWER_TLS_ENABLED", "true").lower() == "true"
    if not enabled:
        return None

    ca_path = os.getenv("FLOWER_TLS_CA_CERTIFICATE")
    cert_path = os.getenv("FLOWER_TLS_CERTIFICATE")
    key_path = os.getenv("FLOWER_TLS_PRIVATE_KEY")
    if cert_path and key_path:
        cert_bytes = Path(cert_path).read_bytes()
        key_bytes = Path(key_path).read_bytes()
        ca_bytes = Path(ca_path).read_bytes() if ca_path else cert_bytes
        return (ca_bytes, cert_bytes, key_bytes)

    if cert_path or key_path:
        logger.warning(
            "FLOWER_TLS_CERTIFICATE and FLOWER_TLS_PRIVATE_KEY must both be set; falling back to self-signed TLS"
        )

    common_name = os.getenv("FLOWER_TLS_COMMON_NAME", "flower.vladar34.xyz")
    logger.info("Generating self-signed TLS certificate for Flower common name %s", common_name)
    return _generate_self_signed_cert(common_name)


def _fit_config(server_round: int) -> Dict[str, float]:
    if server_round <= 2:
        return {"local_epochs": 3, "learning_rate": 0.0025, "label_smoothing": 0.05}
    if server_round <= 4:
        return {"local_epochs": 2, "learning_rate": 0.0015, "label_smoothing": 0.03}
    return {"local_epochs": 1, "learning_rate": 0.001, "label_smoothing": 0.02}


class FederatedStrategy(FedAvg):
    def __init__(self, *args, minio_client=None, bucket_name: str = "models", **kwargs):
        super().__init__(*args, **kwargs)
        self.round_count = 0
        self.minio_client = minio_client
        self.bucket_name = bucket_name
        self.export_round_history = os.getenv("FLOWER_EXPORT_ROUND_HISTORY", "false").lower() == "true"
        self.run_started_at = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.round_history: Dict[int, Dict[str, object]] = {}
        self.round_history_path: Optional[Path] = None
        self.final_round: Optional[int] = None

    def _round_history_root(self) -> Path:
        return Path(__file__).resolve().parents[1] / "reports" / "model_quality"

    def _ensure_round_history_path(self, model_name: str) -> Path:
        if self.round_history_path is None:
            safe_model = model_name if model_name and model_name != "unknown" else "unknown"
            self.round_history_path = (
                self._round_history_root() / f"federated_round_history_{safe_model}_{self.run_started_at}.json"
            )
            self.round_history_path.parent.mkdir(parents=True, exist_ok=True)
        return self.round_history_path

    def _update_round_history(self, model_name: str, server_round: int, section: str, payload: Dict[str, object]) -> None:
        if not self.export_round_history:
            return

        history_path = self._ensure_round_history_path(model_name)
        entry = self.round_history.setdefault(int(server_round), {"round": int(server_round), "model": model_name})
        entry["model"] = model_name
        entry[section] = payload
        ordered_history = [self.round_history[key] for key in sorted(self.round_history)]
        history_payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "run_started_at": self.run_started_at,
            "model": model_name,
            "rounds": ordered_history,
        }
        history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

    def aggregate_fit(self, server_round: int, results, failures) -> Tuple[Optional[Parameters], Dict]:
        self.round_count = server_round
        round_start = perf_counter()

        model_name = "unknown"
        if results:
            try:
                model_name = self._infer_model_name(parameters_to_ndarrays(results[0][1].parameters))
            except Exception:
                model_name = "unknown"

        logger.info("Round %s: Received %s successful updates", server_round, len(results))
        if failures:
            logger.warning("Round %s: %s client failures", server_round, len(failures))

        aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if self.minio_client and aggregated_params and self.final_round == server_round:
            self._save_checkpoint(server_round, aggregated_params)
            self._publish_model_update_notice(model_name, server_round)
        elif self.minio_client and aggregated_params:
            logger.info(
                "Round %s: skipping MinIO checkpoint save until final round %s",
                server_round,
                self.final_round,
            )

        fit_examples = sum(int(getattr(fit_res, "num_examples", 0)) for _, fit_res in results) if results else 0
        fit_metrics = weighted_fit_metrics(
            [(int(getattr(fit_res, "num_examples", 0)), getattr(fit_res, "metrics", {})) for _, fit_res in results]
        )
        self._update_round_history(
            model_name=model_name,
            server_round=server_round,
            section="fit",
            payload={
                "clients": int(len(results)),
                "examples": int(fit_examples),
                "duration_seconds": float(perf_counter() - round_start),
                "status": "success" if aggregated_params else "no_aggregation",
                "metrics": fit_metrics,
            },
        )

        record_flower_round(
            model=model_name,
            round_number=server_round,
            clients=len(results),
            duration_seconds=perf_counter() - round_start,
            status="success" if aggregated_params else "no_aggregation",
        )
        return aggregated_params, aggregated_metrics

    def aggregate_evaluate(self, server_round: int, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        model_name = "unknown"
        if results:
            try:
                model_name = self._infer_model_name(parameters_to_ndarrays(results[0][1].parameters))
            except Exception:
                model_name = "unknown"

        total_examples = sum(int(getattr(evaluate_res, "num_examples", 0)) for _, evaluate_res in results) if results else 0
        weighted_accuracy = None
        if results and total_examples > 0:
            accuracies = [
                int(getattr(evaluate_res, "num_examples", 0))
                * float(getattr(evaluate_res, "metrics", {}).get("accuracy", 0.0))
                for _, evaluate_res in results
            ]
            weighted_accuracy = sum(accuracies) / total_examples

        self._update_round_history(
            model_name=model_name,
            server_round=server_round,
            section="evaluate",
            payload={
                "clients": int(len(results)),
                "examples": int(total_examples),
                "loss": float(aggregated_loss) if aggregated_loss is not None else None,
                "accuracy": float(weighted_accuracy) if weighted_accuracy is not None else None,
                "metrics": aggregated_metrics,
            },
        )
        return aggregated_loss, aggregated_metrics

    def _publish_model_update_notice(self, model_name: str, round_num: int) -> None:
        try:
            if not self.minio_client.bucket_exists(self.bucket_name):
                self.minio_client.make_bucket(self.bucket_name)

            payload = {
                "model": model_name,
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "latest_object": f"models/{model_name}/latest.npz",
            }
            notice_object = f"control/model-updates/{model_name}.json"
            notice_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.minio_client.put_object(
                bucket_name=self.bucket_name,
                object_name=notice_object,
                data=io.BytesIO(notice_bytes),
                length=len(notice_bytes),
                content_type="application/json",
            )
            logger.info("Round %s: published update notice for %s", round_num, model_name)
        except Exception as exc:
            logger.warning("Round %s: failed to publish update notice for %s: %s", round_num, model_name, exc)

    def _infer_model_name(self, ndarrays: List) -> str:
        if not ndarrays:
            return "unknown"
        first_weight = ndarrays[0]
        if len(first_weight.shape) == 2 and first_weight.shape[1] == 8:
            return "mustafa"
        return "alex5050"

    def _save_checkpoint(self, round_num: int, params: Parameters) -> None:
        try:
            ndarrays = [np.asarray(arr, dtype=np.float32) for arr in parameters_to_ndarrays(params)]
            if not ndarrays:
                logger.warning("Round %s: No parameters to save", round_num)
                return

            model_name = self._infer_model_name(ndarrays)
            if not self.minio_client.bucket_exists(self.bucket_name):
                self.minio_client.make_bucket(self.bucket_name)

            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            versioned_object = f"models/{model_name}/versions/version_{timestamp}.npz"
            latest_object = f"models/{model_name}/latest.npz"

            checkpoint_payload = {
                "round": np.array(round_num, dtype=np.int64),
                "model_name": np.array(model_name),
            }
            for index, arr in enumerate(ndarrays):
                checkpoint_payload[f"param_{index}"] = arr

            model_buffer = io.BytesIO()
            np.savez_compressed(model_buffer, **checkpoint_payload)
            payload = model_buffer.getvalue()

            self.minio_client.put_object(
                bucket_name=self.bucket_name,
                object_name=versioned_object,
                data=io.BytesIO(payload),
                length=len(payload),
                content_type="application/octet-stream",
            )
            self.minio_client.put_object(
                bucket_name=self.bucket_name,
                object_name=latest_object,
                data=io.BytesIO(payload),
                length=len(payload),
                content_type="application/octet-stream",
            )

            logger.info(
                "Round %s: Saved %s checkpoint to MinIO: %s and updated %s",
                round_num,
                model_name,
                versioned_object,
                latest_object,
            )
            record_flower_checkpoint_saved(model_name, round_num)
        except S3Error as exc:
            logger.error("Round %s: Failed to save checkpoint to MinIO: %s", round_num, exc)
            record_flower_checkpoint_failure(model_name, "minio")
        except Exception as exc:
            logger.error("Round %s: Unexpected error saving checkpoint: %s", round_num, exc)
            record_flower_checkpoint_failure(model_name, "unexpected")


def create_minio_client() -> Optional[Minio]:
    try:
        endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY")
        secret_key = os.getenv("MINIO_SECRET_KEY")
        use_ssl = os.getenv("MINIO_USE_SSL", "false").lower() == "true"

        client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=use_ssl,
        )
        logger.info("MinIO client initialized for endpoint: %s", endpoint)
        return client
    except Exception as exc:
        logger.warning("Failed to initialize MinIO client: %s", exc)
        logger.info("Proceeding without model persistence to MinIO")
        return None


def create_strategy(minio_client: Optional[Minio] = None) -> FederatedStrategy:
    min_fit_clients = int(os.getenv("FLOWER_MIN_FIT_CLIENTS", "1"))
    min_evaluate_clients = int(os.getenv("FLOWER_MIN_EVALUATE_CLIENTS", str(min_fit_clients)))
    min_available_clients = int(os.getenv("FLOWER_MIN_AVAILABLE_CLIENTS", str(min_fit_clients)))

    return FederatedStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        fit_metrics_aggregation_fn=weighted_fit_metrics,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=_fit_config,
        minio_client=minio_client,
        bucket_name="models",
    )


def start_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    num_rounds: int = 5,
) -> None:
    logger.info("=" * 60)
    logger.info("Starting Flower Federated Learning Server")
    logger.info("=" * 60)
    logger.info("Server Address: %s:%s", host, port)
    logger.info("Number of Rounds: %s", num_rounds)
    logger.info(
        "Client thresholds: min_fit=%s, min_evaluate=%s, min_available=%s",
        os.getenv("FLOWER_MIN_FIT_CLIENTS", "1"),
        os.getenv("FLOWER_MIN_EVALUATE_CLIENTS", os.getenv("FLOWER_MIN_FIT_CLIENTS", "1")),
        os.getenv("FLOWER_MIN_AVAILABLE_CLIENTS", os.getenv("FLOWER_MIN_FIT_CLIENTS", "1")),
    )
    logger.info("Model Family: Auto-detected from client weights (Alex5050 or Mustafa)")
    logger.info("=" * 60)

    control_port = int(os.getenv("FLOWER_CONTROL_PORT", "8081"))
    weekly_scheduler_enabled = os.getenv("FLOWER_WEEKLY_SCHEDULER_ENABLED", "true").lower() == "true"
    prometheus_port = int(os.getenv("FLOWER_PROMETHEUS_PORT", "8001"))

    start_http_server(prometheus_port, addr="0.0.0.0")
    logger.info("Prometheus metrics available on 0.0.0.0:%s/metrics", prometheus_port)

    certificates = _load_tls_certificates()
    if certificates is not None:
        logger.info("Flower gRPC TLS is enabled")
    else:
        logger.info("Flower gRPC TLS is disabled; using insecure transport")

    minio_client = create_minio_client()
    coordinator = TrainingCoordinator(minio_client=minio_client, bucket_name="models")
    start_control_server(coordinator, host="0.0.0.0", port=control_port)
    if weekly_scheduler_enabled:
        start_weekly_scheduler(coordinator)

    strategy = create_strategy(minio_client=minio_client)
    strategy.final_round = num_rounds

    config = ServerConfig(
        num_rounds=num_rounds,
        round_timeout=float(os.getenv("FLOWER_ROUND_TIMEOUT", "600")),
    )
    fl.server.start_server(
        server_address=f"{host}:{port}",
        config=config,
        strategy=strategy,
        certificates=certificates,
    )


if __name__ == "__main__":
    start_server(
        host=os.getenv("FLOWER_SERVER_HOST", "0.0.0.0"),
        port=_resolve_port(),
        num_rounds=int(os.getenv("FLOWER_NUM_ROUNDS", "5")),
    )