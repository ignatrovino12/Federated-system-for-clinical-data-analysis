from functools import lru_cache
import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
import torch
import torch.nn as nn

from .federated_client import DiabetesNet as FederatedAlexDiabetesNet
from .federated_client import MustafaNet as FederatedMustafaDiabetesNet
from .federated_model_sync import get_local_federated_model_path


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_ROOT = BASE_DIR / "machinelearning"


class AlexDiabetesNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def _file_signature(path: Path) -> Tuple[bool, Optional[float], Optional[int]]:
    if not path.exists():
        return (False, None, None)
    stat_result = path.stat()
    return (True, stat_result.st_mtime, stat_result.st_size)


def _alex_model_signature() -> Tuple[Tuple[bool, Optional[float], Optional[int]], Tuple[bool, Optional[float], Optional[int]], Tuple[bool, Optional[float], Optional[int]]]:
    model_dir = MODEL_ROOT / "alex5050_model_NN"
    model_path = model_dir / "best_alex5050_model.pth"
    federated_model_path = get_local_federated_model_path("alex5050")
    scaler_path = model_dir / "alex5050_scaler.joblib"
    return (_file_signature(federated_model_path), _file_signature(model_path), _file_signature(scaler_path))


def _mustafa_model_signature() -> Tuple[
    Tuple[bool, Optional[float], Optional[int]],
    Tuple[bool, Optional[float], Optional[int]],
    Tuple[bool, Optional[float], Optional[int]],
    Tuple[bool, Optional[float], Optional[int]],
]:
    artifact_dirs = [MODEL_ROOT / "mustafa_model_NN", MODEL_ROOT / "mustafa_model"]
    federated_model_path = get_local_federated_model_path("mustafa")
    for artifact_dir in artifact_dirs:
        if (artifact_dir / "scaler.joblib").exists():
            model_path = artifact_dir / "mustafa_model.pth"
            scaler_path = artifact_dir / "scaler.joblib"
            metadata_path = artifact_dir / "metadata.json"
            return (
                _file_signature(federated_model_path),
                _file_signature(model_path),
                _file_signature(scaler_path),
                _file_signature(metadata_path),
            )
    model_path = artifact_dirs[0] / "mustafa_model.pth"
    scaler_path = artifact_dirs[0] / "scaler.joblib"
    metadata_path = artifact_dirs[0] / "metadata.json"
    return (
        _file_signature(federated_model_path),
        _file_signature(model_path),
        _file_signature(scaler_path),
        _file_signature(metadata_path),
    )


class MustafaDiabetesNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


class MonotonicMustafaDiabetesNet(nn.Module):
    def __init__(self, input_dim, monotonic_feature_idx):
        super().__init__()
        self.monotonic_feature_idx = monotonic_feature_idx
        self.base_feature_idx = [i for i in range(input_dim) if i not in monotonic_feature_idx]

        self.base_net = nn.Sequential(
            nn.Linear(len(self.base_feature_idx), 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.raw_monotonic_weights = nn.Parameter(torch.zeros(len(monotonic_feature_idx)))
        self.monotonic_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x_base = x[:, self.base_feature_idx]
        x_mono = x[:, self.monotonic_feature_idx]
        base_logit = self.base_net(x_base)
        mono_weights = torch.nn.functional.softplus(self.raw_monotonic_weights).unsqueeze(1)
        mono_logit = x_mono @ mono_weights
        return base_logit + mono_logit + self.monotonic_bias


@lru_cache(maxsize=4)
def _load_alex_artifacts_cached(signature):
    model_dir = MODEL_ROOT / "alex5050_model_NN"
    model_path = model_dir / "best_alex5050_model.pth"
    federated_model_path = get_local_federated_model_path("alex5050")
    scaler_path = model_dir / "alex5050_scaler.joblib"

    feature_columns = [
        "HighBP",
        "HighChol",
        "CholCheck",
        "BMI",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "GenHlth",
        "MentHlth",
        "PhysHlth",
        "DiffWalk",
        "Sex",
        "Age",
        "Education",
        "Income",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(scaler_path)
    if federated_model_path.exists():
        model = FederatedAlexDiabetesNet(input_size=len(feature_columns)).to(device)
        state_dict = torch.load(federated_model_path, map_location=device)
        source_path = federated_model_path
    else:
        model = AlexDiabetesNet(input_dim=len(feature_columns)).to(device)
        state_dict = torch.load(model_path, map_location=device)
        source_path = model_path
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded Alex model artifacts from %s", source_path)
    return {
        "feature_columns": feature_columns,
        "threshold": 0.37,
        "scaler": scaler,
        "model": model,
        "device": device,
    }


def _load_alex_artifacts():
    return _load_alex_artifacts_cached(_alex_model_signature())


@lru_cache(maxsize=4)
def _load_mustafa_artifacts_cached(signature):
    artifact_dirs = [MODEL_ROOT / "mustafa_model_NN", MODEL_ROOT / "mustafa_model"]
    federated_model_path = get_local_federated_model_path("mustafa")
    artifact_dir = None
    if federated_model_path.exists():
        artifact_dir = next((path for path in artifact_dirs if (path / "scaler.joblib").exists()), None)
        if artifact_dir is None:
            raise FileNotFoundError("Could not find scaler.joblib for mustafa artifacts")
        model_path = federated_model_path
        metadata_path = artifact_dir / "metadata.json"
        scaler_path = artifact_dir / "scaler.joblib"
    else:
        artifact_dir = next((path for path in artifact_dirs if (path / "mustafa_model.pth").exists()), None)
        if artifact_dir is None:
            raise FileNotFoundError("Could not find mustafa_model.pth in machinelearning/mustafa_model_NN or machinelearning/mustafa_model")
        model_path = artifact_dir / "mustafa_model.pth"
        scaler_path = artifact_dir / "scaler.joblib"
        metadata_path = artifact_dir / "metadata.json"

    import json

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    feature_columns = metadata["feature_cols"]
    threshold = float(metadata["threshold"])
    monotonic_feature_cols = metadata.get("monotonic_feature_cols", ["HbA1c_level", "blood_glucose_level"])
    monotonic_feature_idx = [feature_columns.index(column) for column in monotonic_feature_cols]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(scaler_path)
    state_dict = torch.load(model_path, map_location=device)
    if model_path == federated_model_path:
        model = FederatedMustafaDiabetesNet(input_size=len(feature_columns)).to(device)
    elif any(key.startswith("raw_monotonic_weights") for key in state_dict.keys()):
        model = MonotonicMustafaDiabetesNet(
            input_dim=len(feature_columns),
            monotonic_feature_idx=monotonic_feature_idx,
        ).to(device)
    else:
        model = MustafaDiabetesNet(input_dim=len(feature_columns)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return {
        "feature_columns": feature_columns,
        "threshold": threshold,
        "scaler": scaler,
        "model": model,
        "device": device,
    }


def _load_mustafa_artifacts():
    return _load_mustafa_artifacts_cached(_mustafa_model_signature())


def _validate_payload(payload, feature_columns):
    missing = [column for column in feature_columns if column not in payload or payload[column] is None or payload[column] == ""]
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}")


def predict_alex_probability(payload):
    artifacts = _load_alex_artifacts()
    feature_columns = artifacts["feature_columns"]
    _validate_payload(payload, feature_columns)
    row = pd.DataFrame([payload], columns=feature_columns).astype("float32")
    scaled = artifacts["scaler"].transform(row[feature_columns])

    with torch.no_grad():
        logits = artifacts["model"](torch.tensor(scaled, dtype=torch.float32).to(artifacts["device"]))
        probability = torch.sigmoid(logits).cpu().numpy().ravel()[0]

    return {
        "probability": float(probability),
        "probability_pct": round(float(probability) * 100, 2),
        "threshold": artifacts["threshold"],
        "prediction": int(probability >= artifacts["threshold"]),
    }


def predict_mustafa_probability(payload):
    artifacts = _load_mustafa_artifacts()
    feature_columns = artifacts["feature_columns"]
    _validate_payload(payload, feature_columns)
    row = pd.DataFrame([payload], columns=feature_columns).astype("float32")
    scaled = artifacts["scaler"].transform(row[feature_columns])

    with torch.no_grad():
        logits = artifacts["model"](torch.tensor(scaled, dtype=torch.float32).to(artifacts["device"]))
        probability = torch.sigmoid(logits).cpu().numpy().ravel()[0]

    return {
        "probability": float(probability),
        "probability_pct": round(float(probability) * 100, 2),
        "threshold": artifacts["threshold"],
        "prediction": int(probability >= artifacts["threshold"]),
    }