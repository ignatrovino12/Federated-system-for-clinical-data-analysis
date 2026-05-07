"""
Federated client implementation for Django clinic instances.
This module lives in the main Django app so clinic deployments can train locally
and send updated weights to the central Flower server.
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.dropout3(self.relu3(self.fc3(x)))
        return self.sigmoid(self.fc4(x))


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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.sigmoid(self.fc3(x))


def get_weights(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 1,
    lr: float = 0.001,
    device: str = "cpu",
) -> float:
    model.to(device)
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / (num_batches or 1):.4f}")

    return total_loss / (num_batches or 1)


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
            loss = criterion(output, batch_y)
            total_loss += loss.item()
            predictions = (output > 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total if total > 0 else 0.0
    loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    return loss, accuracy


class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return get_weights(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        logger.info("Client fit() called")
        set_weights(self.model, parameters)
        loss = train(
            self.model,
            self.train_loader,
            epochs=int(config.get("local_epochs", 1)),
            lr=float(config.get("learning_rate", 0.001)),
            device=self.device,
        )
        num_examples = len(self.train_loader.dataset)
        return get_weights(self.model), num_examples, {"loss": loss}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        logger.info("Client evaluate() called")
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader, device=self.device)
        num_examples = len(self.test_loader.dataset) if self.test_loader else 0
        return loss, num_examples, {"accuracy": accuracy}


def create_client(
    model_type: str = "alex5050",
    local_data: Tuple[np.ndarray, np.ndarray] = None,
    test_split: float = 0.2,
    batch_size: int = 32,
) -> FederatedClient:
    if model_type.lower() == "alex5050":
        model = DiabetesNet(input_size=21)
        input_size = 21
    elif model_type.lower() == "mustafa":
        model = MustafaNet(input_size=8)
        input_size = 8
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if local_data is None:
        logger.warning("No local data provided, creating dummy data")
        features = np.random.randn(100, input_size).astype(np.float32)
        labels = np.random.randint(0, 2, 100).astype(np.float32)
    else:
        features, labels = local_data

    num_examples = len(features)
    if num_examples == 0:
        raise ValueError("No local samples available for federated training")

    split_idx = int(num_examples * (1 - test_split))
    split_idx = max(1, split_idx)
    if num_examples > 1:
        split_idx = min(split_idx, num_examples - 1)

    train_x, test_x = features[:split_idx], features[split_idx:]
    train_y, test_y = labels[:split_idx], labels[split_idx:]

    train_dataset = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
    test_dataset = TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return FederatedClient(model, train_loader, test_loader)
