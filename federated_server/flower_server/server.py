"""
Flower Server for Federated Learning
Orchestrates model training across distributed clients (patient devices/clinics)
"""

import logging
import os
import io
import json
from time import perf_counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import flwr as fl
import torch
import torch.nn as nn
from flwr.server.strategy import FedAvg
from flwr.common import Metrics, Parameters, parameters_to_ndarrays
from flwr.server import ServerConfig
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("flower-server")


# Define the neural network architectures that clients will train
class DiabetesNet(nn.Module):
    """Alex5050 model architecture for federated learning"""
    def __init__(self, input_size: int = 21):
        super(DiabetesNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.dropout3(self.relu3(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))
        return x


class MustafaNet(nn.Module):
    """Mustafa model architecture for federated learning"""
    def __init__(self, input_size: int = 8):
        super(MustafaNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate metrics from all clients weighted by sample counts.
    Called after each round of federated learning.
    """
    if not metrics:
        return {}
    
    # Unpack metrics
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, m in metrics]
    
    # Aggregate
    return {"accuracy": sum(accuracies) / sum(examples)}


def weighted_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate fit metrics reported by clients using example-count weighting."""
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


def _fit_config(server_round: int) -> Dict[str, float]:
    """Ask clients to do a little more local work so the update can move off the prior."""
    if server_round <= 2:
        return {"local_epochs": 3, "learning_rate": 0.0025, "label_smoothing": 0.05}
    if server_round <= 4:
        return {"local_epochs": 2, "learning_rate": 0.0015, "label_smoothing": 0.03}
    return {"local_epochs": 1, "learning_rate": 0.001, "label_smoothing": 0.02}


class FederatedStrategy(FedAvg):
    """
    Custom Flower strategy for federated learning with MinIO persistence.
    Extends FedAvg with model checkpointing to MinIO after each round.
    Automatically detects model family (Alex5050 or Mustafa) from aggregated weight shapes.
    """
    
    def __init__(self, *args, minio_client=None, bucket_name="models", **kwargs):
        super().__init__(*args, **kwargs)
        self.round_count = 0
        self.minio_client = minio_client
        self.bucket_name = bucket_name
        self.export_round_history = os.getenv("FLOWER_EXPORT_ROUND_HISTORY", "false").lower() == "true"
        self.run_started_at = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.round_history: Dict[int, Dict[str, object]] = {}
        self.round_history_path: Optional[Path] = None
        # final_round will be assigned by the server at startup
        self.final_round: Optional[int] = None

    def _round_history_root(self) -> Path:
        return Path(__file__).resolve().parents[1] / "reports" / "model_quality"

    def _ensure_round_history_path(self, model_name: str) -> Path:
        if self.round_history_path is None:
            safe_model = model_name if model_name and model_name != "unknown" else "unknown"
            self.round_history_path = self._round_history_root() / f"federated_round_history_{safe_model}_{self.run_started_at}.json"
            self.round_history_path.parent.mkdir(parents=True, exist_ok=True)
        return self.round_history_path

    def _update_round_history(self, model_name: str, server_round: int, section: str, payload: Dict[str, object]) -> None:
        if not self.export_round_history:
            return
        history_path = self._ensure_round_history_path(model_name)
        entry = self.round_history.setdefault(
            int(server_round),
            {
                "round": int(server_round),
                "model": model_name,
            },
        )
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
    
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ) -> Tuple[Optional[Parameters], Dict]:
        """Aggregate model weights and save checkpoint to MinIO."""
        self.round_count = server_round
        round_start = perf_counter()

        model_name = "unknown"
        if results:
            try:
                model_name = self._infer_model_name(parameters_to_ndarrays(results[0][1].parameters))
            except Exception:
                model_name = "unknown"
        
        logger.info(f"Round {server_round}: Received {len(results)} successful updates")
        if failures:
            logger.warning(f"Round {server_round}: {len(failures)} client failures")
        
        # Call parent aggregation
        aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Persist aggregated model to MinIO if available
        if self.minio_client and aggregated_params:
            self._save_checkpoint(server_round, aggregated_params)
            self._publish_model_update_notice(model_name, server_round)

        fit_examples = sum(int(getattr(fit_res, "num_examples", 0)) for _, fit_res in results) if results else 0
        fit_metrics = weighted_fit_metrics([
            (int(getattr(fit_res, "num_examples", 0)), getattr(fit_res, "metrics", {}))
            for _, fit_res in results
        ])
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
        """Aggregate client evaluation metrics and persist a per-round history entry."""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        model_name = "unknown"
        if results:
            try:
                model_name = self._infer_model_name(parameters_to_ndarrays(results[0][1].parameters))
            except Exception:
                model_name = "unknown"

        total_examples = sum(int(getattr(evaluate_res, "num_examples", 0)) for _, evaluate_res in results) if results else 0
        weighted_accuracy = None
        if results:
            accuracies = [
                int(getattr(evaluate_res, "num_examples", 0))
                * float(getattr(evaluate_res, "metrics", {}).get("accuracy", 0.0))
                for _, evaluate_res in results
            ]
            if total_examples > 0:
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
        """Persist a small notice so clinic admins know a newer model is available."""
        try:
            if not self.minio_client.bucket_exists(self.bucket_name):
                self.minio_client.make_bucket(self.bucket_name)

            payload = {
                "model": model_name,
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "latest_object": f"models/{model_name}/latest.pt",
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
        """Infer the model family from the aggregated parameter shapes."""
        if not ndarrays:
            return "unknown"

        first_weight = ndarrays[0]
        if len(first_weight.shape) == 2 and first_weight.shape[1] == 8:
            return "mustafa"
        return "alex5050"
    
    def _build_model(self, model_name: str) -> nn.Module:
        """Build the server-side model for the given model family."""
        if model_name == "mustafa":
            return MustafaNet(input_size=8)
        if model_name == "alex5050":
            return DiabetesNet(input_size=21)
        raise ValueError(f"Unsupported model family: {model_name}")
    
    def _save_checkpoint(self, round_num: int, params: Parameters) -> None:
        """Save a versioned checkpoint and update the latest pointer in MinIO."""
        try:
            ndarrays = parameters_to_ndarrays(params)
            if not ndarrays:
                logger.warning(f"Round {round_num}: No parameters to save")
                return

            model_name = self._infer_model_name(ndarrays)
            inferred_model = self._build_model(model_name)
            state_dict = inferred_model.state_dict()
            keys = list(state_dict.keys())

            if not self.minio_client.bucket_exists(self.bucket_name):
                self.minio_client.make_bucket(self.bucket_name)

            if len(ndarrays) != len(keys):
                logger.error(
                    f"Round {round_num}: Parameter count mismatch for checkpoint save "
                    f"(received={len(ndarrays)}, expected={len(keys)})"
                )
                record_flower_checkpoint_failure(model_name, "parameter_mismatch")
                return

            for key, arr in zip(keys, ndarrays):
                state_dict[key] = torch.tensor(arr, dtype=torch.float32)
            
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            versioned_object = f"models/{model_name}/versions/version_{timestamp}.pt"
            latest_object = f"models/{model_name}/latest.pt"

            # Save model to bytes
            model_buffer = io.BytesIO()
            torch.save(state_dict, model_buffer)
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
                f"Round {round_num}: Saved {model_name} checkpoint to MinIO: {versioned_object} and updated {latest_object}"
            )
            record_flower_checkpoint_saved(model_name, round_num)
            
        except S3Error as e:
            logger.error(f"Round {round_num}: Failed to save checkpoint to MinIO: {e}")
            record_flower_checkpoint_failure(model_name, "minio")
        except Exception as e:
            logger.error(f"Round {round_num}: Unexpected error saving checkpoint: {e}")
            record_flower_checkpoint_failure(model_name, "unexpected")


def create_minio_client() -> Optional[Minio]:
    """Create MinIO client from environment variables."""
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

        logger.info(f"MinIO client initialized for endpoint: {endpoint}")
        return client
    except Exception as e:
        logger.warning(f"Failed to initialize MinIO client: {e}")
        logger.info("Proceeding without model persistence to MinIO")
        return None


def create_strategy(minio_client: Optional[Minio] = None) -> FederatedStrategy:
    """Create and configure the federated learning strategy."""
    min_fit_clients = int(os.getenv("FLOWER_MIN_FIT_CLIENTS", "1"))
    min_evaluate_clients = int(os.getenv("FLOWER_MIN_EVALUATE_CLIENTS", str(min_fit_clients)))
    min_available_clients = int(os.getenv("FLOWER_MIN_AVAILABLE_CLIENTS", str(min_fit_clients)))

    strategy = FederatedStrategy(
        fraction_fit=1.0,  # Sample a percentage of available clients per round (when a lot of clinics are available, we can reduce this to speed up rounds and reduce load)
        fraction_evaluate=1.0,  # Sample percentage for evaluation to avoid overloading
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        fit_metrics_aggregation_fn=weighted_fit_metrics,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=_fit_config,
        minio_client=minio_client,
        bucket_name="models",
    )
    return strategy


def start_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    num_rounds: int = 5,
) -> None:
    """
    Start the Flower federated learning server.
    
    Args:
        host: Server host address
        port: Server port
        num_rounds: Number of federated training rounds
    """
    logger.info("=" * 60)
    logger.info("Starting Flower Federated Learning Server")
    logger.info("=" * 60)
    logger.info(f"Server Address: {host}:{port}")
    logger.info(f"Number of Rounds: {num_rounds}")
    logger.info(
        "Client thresholds: min_fit=%s, min_evaluate=%s, min_available=%s",
        os.getenv("FLOWER_MIN_FIT_CLIENTS", "1"),
        os.getenv("FLOWER_MIN_EVALUATE_CLIENTS", os.getenv("FLOWER_MIN_FIT_CLIENTS", "1")),
        os.getenv("FLOWER_MIN_AVAILABLE_CLIENTS", os.getenv("FLOWER_MIN_FIT_CLIENTS", "1")),
    )
    logger.info("Model Family: Auto-detected from client weights (Alex5050 or Mustafa)")
    logger.info("=" * 60)

    control_port = int(os.getenv("FLOWER_CONTROL_PORT", 8081))
    weekly_scheduler_enabled = os.getenv("FLOWER_WEEKLY_SCHEDULER_ENABLED", "true").lower() == "true"

    prometheus_port = int(os.getenv("FLOWER_PROMETHEUS_PORT", 8001))
    start_http_server(prometheus_port, addr="0.0.0.0")
    logger.info(f"Prometheus metrics available on 0.0.0.0:{prometheus_port}/metrics")
    
    # Initialize MinIO client (optional)
    minio_client = create_minio_client()

    coordinator = TrainingCoordinator(minio_client=minio_client, bucket_name="models")
    start_control_server(coordinator, host="0.0.0.0", port=control_port)
    if weekly_scheduler_enabled:
        start_weekly_scheduler(coordinator)
    
    while True:
        # Create strategy with MinIO persistence; instruct it which final round to persist
        strategy = create_strategy(minio_client=minio_client)
        strategy.final_round = num_rounds

        # Configure one federated training session
        config = ServerConfig(
            num_rounds=num_rounds,
            round_timeout=600,  # 10 minutes per round
        )

        # Run one federated session and then wait for the next client wave
        fl.server.start_server(
            server_address=f"{host}:{port}",
            config=config,
            strategy=strategy,
        )
        logger.info("Federated session completed; keeping server alive for the next trigger")


if __name__ == "__main__":
    # Configuration from environment variables
    host = os.getenv("FLOWER_HOST", "0.0.0.0")
    port = int(os.getenv("FLOWER_PORT", 8080))
    num_rounds = int(os.getenv("FLOWER_ROUNDS", 5))
    
    start_server(host=host, port=port, num_rounds=num_rounds)
