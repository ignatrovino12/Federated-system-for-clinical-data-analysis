"""
Flower Server for Federated Learning
Orchestrates model training across distributed clients (patient devices/clinics)
"""

import logging
import os
import io
from time import perf_counter
from datetime import datetime
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


def _fit_config(server_round: int) -> Dict[str, float]:
    """Ask clients to do a little more local work so the update can move off the prior."""
    if server_round <= 2:
        return {"local_epochs": 5, "learning_rate": 0.005}
    return {"local_epochs": 3, "learning_rate": 0.003}


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
        # final_round will be assigned by the server at startup
        self.final_round: Optional[int] = None
    
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

        record_flower_round(
            model=model_name,
            round_number=server_round,
            clients=len(results),
            duration_seconds=perf_counter() - round_start,
            status="success" if aggregated_params else "no_aggregation",
        )
        
        return aggregated_params, aggregated_metrics

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

            # If configured to save only at the final round, skip intermediate rounds
            if self.final_round is not None and round_num != self.final_round:
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
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
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
    
    # Create strategy with MinIO persistence; instruct it which final round to persist
    strategy = create_strategy(minio_client=minio_client)
    strategy.final_round = num_rounds
    
    # Configure server
    config = ServerConfig(
        num_rounds=num_rounds,
        round_timeout=600,  # 10 minutes per round
    )
    
    # Start server
    fl.server.start_server(
        server_address=f"{host}:{port}",
        config=config,
        strategy=strategy,
    )


if __name__ == "__main__":
    # Configuration from environment variables
    host = os.getenv("FLOWER_HOST", "0.0.0.0")
    port = int(os.getenv("FLOWER_PORT", 8080))
    num_rounds = int(os.getenv("FLOWER_ROUNDS", 5))
    
    start_server(host=host, port=port, num_rounds=num_rounds)
