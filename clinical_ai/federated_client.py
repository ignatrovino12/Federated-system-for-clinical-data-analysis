"""
Federated client implementation for Django clinic instances.
This module lives in the main Django app so clinic deployments can train locally
and send updated weights to the central Flower server.
"""

import logging
import os
from collections import OrderedDict
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple, Union

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from healthcheck.metrics import record_client_evaluate, record_client_fit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("flower-client")


class DiabetesNet(nn.Module):
    """Alex5050 model architecture."""

    def __init__(self, input_size: int = 21):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.dropout3(self.relu3(self.fc3(x)))
        return self.fc4(x)


class MustafaNet(nn.Module):
    """Mustafa model architecture."""

    def __init__(self, input_size: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.fc3(x)


def get_weights(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def load_federated_checkpoint(model_path: Path, model: nn.Module) -> None:
    if model_path.suffix == ".npz":
        with np.load(model_path, allow_pickle=False) as checkpoint:
            param_keys = sorted(
                (key for key in checkpoint.files if key.startswith("param_")),
                key=lambda key: int(key.split("_", 1)[1]),
            )
            weights = [checkpoint[key] for key in param_keys]
        set_weights(model, weights)
        return

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 1,
    lr: float = 0.001,
    device: str = "cpu",
    label_smoothing: float = 0.0,
) -> Tuple[float, List[int]]:
    model.to(device)
    model.train()
    # compute class imbalance from dataset labels and create pos-weight
    try:
        labels_tensor = train_loader.dataset.tensors[1]
        num_pos = int((labels_tensor == 1).sum().item())
        num_total = labels_tensor.size(0)
        num_neg = num_total - num_pos
        pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0
    except Exception:
        pos_weight = 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device), reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    num_batches = 0
    used_record_ids = set()
    smoothing = max(0.0, min(0.2, float(label_smoothing)))

    for epoch in range(epochs):
        for batch in train_loader:
            if len(batch) == 4:
                batch_x, batch_y, batch_weights, batch_record_ids = batch
                used_record_ids.update(int(record_id) for record_id in batch_record_ids.cpu().tolist())
            elif len(batch) == 3:
                batch_x, batch_y, batch_weights = batch
            else:
                batch_x, batch_y = batch
                batch_weights = None

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            if smoothing > 0:
                # Soft labels reduce the incentive for the model to push logits to the extremes.
                batch_y = batch_y * (1.0 - smoothing) + 0.5 * smoothing

            optimizer.zero_grad()
            output = model(batch_x)
            per_sample_loss = criterion(output, batch_y)
            if batch_weights is not None:
                weight_tensor = batch_weights.to(device).unsqueeze(1)
                loss = (per_sample_loss * weight_tensor).mean()
            else:
                loss = per_sample_loss.mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / (num_batches or 1):.4f}")

    return total_loss / (num_batches or 1), sorted(used_record_ids)


def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
) -> Tuple[float, float]:
    model.to(device)
    model.eval()
    criterion = nn.BCELoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            output = model(batch_x)
            loss = nn.BCEWithLogitsLoss()(output, batch_y)
            total_loss += loss.item()
            predictions = (torch.sigmoid(output) > 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total if total > 0 else 0.0
    loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    return loss, accuracy


class FederatedClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_name: str,
        sampling_strength: float = 1.0,
        training_sample_fraction: float = 0.5,
        training_record_ids: Optional[Sequence[int]] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size = int(train_loader.batch_size or 32)
        self.model_name = model_name
        self.sampling_strength = max(0.0, min(4.0, float(sampling_strength)))
        self.training_sample_fraction = max(0.1, min(1.0, float(training_sample_fraction)))
        self.clinic_id = os.getenv("CLINIC_ID", os.getenv("HOSTNAME", "clinic"))
        self.training_record_ids = list(training_record_ids or [])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _parameter_bytes(self, weights: List[np.ndarray]) -> int:
        return int(sum(array.nbytes for array in weights))

    def _rebuild_weighted_train_loader(self, sample_weights: np.ndarray) -> None:
        # Prefer low-round samples by sampling inversely to federated_train_count-derived weights.
        sampler_weights = torch.from_numpy(np.clip(sample_weights, 1e-6, None).astype(np.float64))
        num_samples = max(1, int(round(len(sample_weights) * self.training_sample_fraction)))
        sampler = WeightedRandomSampler(
            weights=sampler_weights,
            num_samples=num_samples,
            replacement=False,
        )
        self.train_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
        )

    def _refresh_training_weights(self) -> None:
        """Recompute local sample weights from persisted federated_train_count values."""
        if not self.training_record_ids:
            return
        if len(self.training_record_ids) != len(self.train_loader.dataset):
            logger.warning(
                "Skipping weight refresh: training_record_ids (%s) and train dataset size (%s) differ",
                len(self.training_record_ids),
                len(self.train_loader.dataset),
            )
            return

        try:
            from clinical_ai.models import PatientClinicalRecord

            id_to_count = {
                row["id"]: int(row["federated_train_count"] or 0)
                for row in PatientClinicalRecord.objects.filter(id__in=self.training_record_ids).values(
                    "id", "federated_train_count"
                )
            }

            # Compute rank-based weights so the lowest federated_train_count records
            # receive higher sampling probability, but compress the dynamic range
            # to avoid a tiny set of records dominating selection forever.
            counts = np.asarray([float(id_to_count.get(rid, 0)) for rid in self.training_record_ids], dtype=np.float32)
            order = np.argsort(counts)  # indices of records sorted by count (ascending)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(counts), 0, -1)  # larger number => preferred

            # Normalize ranks to [0,1] then scale into [min_val, 1.0] to avoid zeros.
            min_val = 1e-3
            if ranks.max() - ranks.min() > 0:
                norm = (ranks - ranks.min()) / (ranks.max() - ranks.min())
            else:
                norm = np.ones_like(ranks, dtype=np.float32)
            base = min_val + (1.0 - min_val) * norm  # in [min_val, 1.0]

            # Apply a modest exponent to increase preference; scale exponent so
            # it doesn't explode for large `sampling_strength` values.
            power = 1.0 + float(self.sampling_strength) * 0.5
            refreshed_weights = (base ** power).astype(np.float32)

            # Debug logging: counts summary and a few top weights to diagnose issues.
            try:
                unique_counts = np.unique(counts)
                top_idx = np.argsort(-refreshed_weights)[:5]
                top_examples = [
                    (int(self.training_record_ids[i]), float(counts[i]), float(refreshed_weights[i]))
                    for i in top_idx
                ]
                logger.info(
                    "Sampling weights summary: counts_min=%s, counts_max=%s, unique_counts=%s, top=%s",
                    float(counts.min()),
                    float(counts.max()),
                    unique_counts[:10].tolist(),
                    top_examples,
                )
            except Exception:
                logger.exception("Failed to log sampling weight debug info")

            dataset_tensors = list(self.train_loader.dataset.tensors)
            dataset_tensors[2] = torch.from_numpy(refreshed_weights).float()
            self.train_loader.dataset.tensors = tuple(dataset_tensors)
            self._rebuild_weighted_train_loader(refreshed_weights)
        except Exception:
            logger.exception("Failed to refresh sample weights from federated training counters")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return get_weights(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        logger.info("Client fit() called")
        started_at = perf_counter()
        self._refresh_training_weights()
        set_weights(self.model, parameters)
        loss, used_record_ids = train(
            self.model,
            self.train_loader,
            epochs=int(config.get("local_epochs", 1)),
            lr=float(config.get("learning_rate", 0.001)),
            device=self.device,
            label_smoothing=float(config.get("label_smoothing", 0.0)),
        )
        updated_weights = get_weights(self.model)
        duration_seconds = perf_counter() - started_at
        if len(self.train_loader.dataset) > 0:
            try:
                sample_weights = self.train_loader.dataset.tensors[2].detach().cpu().numpy()
                effective_examples = max(1, int(round(float(np.sum(sample_weights)))))
            except Exception:
                effective_examples = len(self.train_loader.dataset)
        else:
            effective_examples = 0
        if self.training_record_ids:
            try:
                from django.db.models import F
                from django.utils import timezone
                from clinical_ai.models import PatientClinicalRecord

                updated_rows = PatientClinicalRecord.objects.filter(id__in=used_record_ids).update(
                    federated_train_count=F("federated_train_count") + 1,
                    last_federated_train_at=timezone.now(),
                )
                logger.info("Updated federated counters for %s sampled records", updated_rows)
            except Exception:
                logger.exception("Failed to update federated training counters")
        record_client_fit(
            clinic=self.clinic_id,
            model=self.model_name,
            duration_seconds=duration_seconds,
            num_examples=effective_examples,
            loss=loss,
            parameter_bytes_sent=self._parameter_bytes(updated_weights),
        )
        return updated_weights, effective_examples, {
            "loss": loss,
            "fit_duration_seconds": duration_seconds,
            "effective_examples": effective_examples,
        }

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        logger.info("Client evaluate() called")
        started_at = perf_counter()
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader, device=self.device)
        duration_seconds = perf_counter() - started_at
        parameter_bytes_received = self._parameter_bytes(parameters)
        num_examples = len(self.test_loader.dataset) if self.test_loader else 0
        record_client_evaluate(
            clinic=self.clinic_id,
            model=self.model_name,
            duration_seconds=duration_seconds,
            num_examples=num_examples,
            loss=loss,
            accuracy=accuracy,
            parameter_bytes_received=parameter_bytes_received,
        )
        return loss, num_examples, {"accuracy": accuracy, "evaluate_duration_seconds": duration_seconds}


def create_client(
    model_type: str = "alex5050",
    local_data: Tuple[np.ndarray, np.ndarray] = None,
    local_sample_weights: Optional[np.ndarray] = None,
    record_ids: Optional[Sequence[int]] = None,
    test_split: float = 0.2,
    batch_size: int = 32,
    sampling_strength: float = 1.0,
    training_sample_fraction: float = 0.5,
) -> FederatedClient:
    if model_type.lower() == "alex5050":
        model = DiabetesNet(input_size=21)
        input_size = 21
    elif model_type.lower() == "mustafa":
        model = MustafaNet(input_size=8)
        input_size = 8
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Try to load the latest saved model from disk; if not found, use freshly initialized model
    from clinical_ai.federated_model_sync import get_local_federated_model_path
    local_model_path = get_local_federated_model_path(model_type)
    if local_model_path.exists():
        try:
            load_federated_checkpoint(local_model_path, model)
            logger.info(f"Loaded local federated model from {local_model_path}")
        except Exception as e:
            logger.warning(f"Failed to load local federated model from {local_model_path}: {e}. Using fresh model.")
    else:
        logger.info(f"No local federated model found at {local_model_path}. Using fresh model.")

    if local_data is None:
        logger.warning("No local data provided, creating dummy data")
        features = np.random.randn(100, input_size).astype(np.float32)
        labels = np.random.randint(0, 2, 100).astype(np.float32)
        sample_weights = np.ones(100, dtype=np.float32)
        record_ids = list(range(100))
    else:
        features, labels = local_data
        sample_weights = local_sample_weights if local_sample_weights is not None else np.ones(len(features), dtype=np.float32)
        record_ids = list(record_ids or range(len(features)))

    num_examples = len(features)
    if num_examples == 0:
        raise ValueError("No local samples available for federated training")

    split_idx = int(num_examples * (1 - test_split))
    split_idx = max(1, split_idx)
    if num_examples > 1:
        split_idx = min(split_idx, num_examples - 1)

    train_x, test_x = features[:split_idx], features[split_idx:]
    train_y, test_y = labels[:split_idx], labels[split_idx:]
    train_weights = sample_weights[:split_idx]
    train_record_ids = list(record_ids[:split_idx])

    train_dataset = TensorDataset(
        torch.from_numpy(train_x).float(),
        torch.from_numpy(train_y).float(),
        torch.from_numpy(train_weights).float(),
        torch.tensor(train_record_ids, dtype=torch.long),
    )
    test_dataset = TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float())

    # Use each training sample once per epoch and downweight patients that have already been trained many times.
    train_sampler_weights = np.clip(train_weights, 1e-6, None).astype(np.float64)
    num_samples = max(1, int(round(len(train_sampler_weights) * max(0.1, min(1.0, float(training_sample_fraction))))))
    train_sampler = WeightedRandomSampler(
        weights=torch.from_numpy(train_sampler_weights),
        num_samples=num_samples,
        replacement=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return FederatedClient(
        model,
        train_loader,
        test_loader,
        model_name=model_type.lower(),
        sampling_strength=sampling_strength,
        training_sample_fraction=training_sample_fraction,
        training_record_ids=train_record_ids,
    )
