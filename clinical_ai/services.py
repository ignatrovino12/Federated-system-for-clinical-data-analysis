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
from .calibration import apply_calibration_state, load_calibration_state
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
    calibration_path = model_dir / "calibration.json"
    return (
        _file_signature(federated_model_path),
        _file_signature(model_path),
        _file_signature(scaler_path),
        _file_signature(calibration_path),
    )


def _mustafa_model_signature() -> Tuple[
    Tuple[bool, Optional[float], Optional[int]],
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
            calibration_path = artifact_dir / "calibration.json"
            return (
                _file_signature(federated_model_path),
                _file_signature(model_path),
                _file_signature(scaler_path),
                _file_signature(metadata_path),
                _file_signature(calibration_path),
            )
    model_path = artifact_dirs[0] / "mustafa_model.pth"
    scaler_path = artifact_dirs[0] / "scaler.joblib"
    metadata_path = artifact_dirs[0] / "metadata.json"
    calibration_path = artifact_dirs[0] / "calibration.json"
    return (
        _file_signature(federated_model_path),
        _file_signature(model_path),
        _file_signature(scaler_path),
        _file_signature(metadata_path),
        _file_signature(calibration_path),
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
    scaler_feature_names = getattr(scaler, "feature_names_in_", None)
    calibration_path = model_dir / "calibration.json"
    calibration = load_calibration_state(calibration_path)
    if federated_model_path.exists():
        model = FederatedAlexDiabetesNet(input_size=len(feature_columns)).to(device)
        state_dict = torch.load(federated_model_path, map_location=device)
        source_path = federated_model_path
        outputs_probability = False
    else:
        model = AlexDiabetesNet(input_dim=len(feature_columns)).to(device)
        state_dict = torch.load(model_path, map_location=device)
        source_path = model_path
        outputs_probability = False
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded Alex model artifacts from %s", source_path)
    return {
        "feature_columns": feature_columns,
        "threshold": 0.37,
        "scaler": scaler,
        "scaler_feature_names": scaler_feature_names,
        "model": model,
        "device": device,
        "outputs_probability": outputs_probability,
        "calibration": calibration,
        "calibration_path": calibration_path,
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
    scaler_feature_names = getattr(scaler, "feature_names_in_", None)
    calibration_path = artifact_dir / "calibration.json"
    calibration = load_calibration_state(calibration_path)
    state_dict = torch.load(model_path, map_location=device)
    if model_path == federated_model_path:
        model = FederatedMustafaDiabetesNet(input_size=len(feature_columns)).to(device)
        outputs_probability = False
    elif any(key.startswith("raw_monotonic_weights") for key in state_dict.keys()):
        model = MonotonicMustafaDiabetesNet(
            input_dim=len(feature_columns),
            monotonic_feature_idx=monotonic_feature_idx,
        ).to(device)
        outputs_probability = False
    else:
        model = MustafaDiabetesNet(input_dim=len(feature_columns)).to(device)
        outputs_probability = False
    model.load_state_dict(state_dict)
    model.eval()
    return {
        "feature_columns": feature_columns,
        "threshold": threshold,
        "scaler": scaler,
        "scaler_feature_names": scaler_feature_names,
        "model": model,
        "device": device,
        "outputs_probability": outputs_probability,
        "calibration": calibration,
        "calibration_path": calibration_path,
    }


def _load_mustafa_artifacts():
    return _load_mustafa_artifacts_cached(_mustafa_model_signature())


def _validate_payload(payload, feature_columns):
    missing = [column for column in feature_columns if column not in payload or payload[column] is None or payload[column] == ""]
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}")


def _scale_payload_row(artifacts, payload, feature_columns):
    row = pd.DataFrame([payload], columns=feature_columns).astype("float32")
    scaler_feature_names = artifacts.get("scaler_feature_names")
    if scaler_feature_names is None:
        return artifacts["scaler"].transform(row.to_numpy(dtype="float32", copy=False))

    return artifacts["scaler"].transform(row[list(scaler_feature_names)])


def _calibrated_probability(artifacts, scaled_row):
    with torch.no_grad():
        model_output = artifacts["model"](torch.tensor(scaled_row, dtype=torch.float32).to(artifacts["device"]))
        calibrated = apply_calibration_state(
            model_output.detach().cpu().numpy().ravel(),
            outputs_probability=bool(artifacts.get("outputs_probability", False)),
            calibration_state=artifacts.get("calibration"),
        )

    return float(calibrated.ravel()[0])


def predict_alex_probability(payload):
    artifacts = _load_alex_artifacts()
    feature_columns = artifacts["feature_columns"]
    scaler_feature_names = artifacts.get("scaler_feature_names")
    # prefer scaler's feature order if available
    columns_to_use = list(scaler_feature_names) if scaler_feature_names is not None else feature_columns
    if scaler_feature_names is not None and list(scaler_feature_names) != feature_columns:
        logger.warning("Scaler was fitted with different feature names; reordering input columns to scaler order")
    _validate_payload(payload, columns_to_use)
    scaled = _scale_payload_row(artifacts, payload, columns_to_use)
    probability = _calibrated_probability(artifacts, scaled)

    calibration = artifacts.get("calibration")
    threshold = float(getattr(calibration, "decision_threshold", artifacts["threshold"])) if calibration else artifacts["threshold"]

    return {
        "probability": float(probability),
        "probability_pct": round(float(probability) * 100, 2),
        "threshold": threshold,
        "prediction": int(probability >= threshold),
        "calibration_applied": bool(calibration),
    }


def predict_mustafa_probability(payload):
    artifacts = _load_mustafa_artifacts()
    feature_columns = artifacts["feature_columns"]
    scaler_feature_names = artifacts.get("scaler_feature_names")
    columns_to_use = list(scaler_feature_names) if scaler_feature_names is not None else feature_columns
    if scaler_feature_names is not None and list(scaler_feature_names) != feature_columns:
        logger.warning("Scaler was fitted with different feature names; reordering input columns to scaler order")
    _validate_payload(payload, columns_to_use)
    scaled = _scale_payload_row(artifacts, payload, columns_to_use)
    probability = _calibrated_probability(artifacts, scaled)

    calibration = artifacts.get("calibration")
    threshold = float(getattr(calibration, "decision_threshold", artifacts["threshold"])) if calibration else artifacts["threshold"]

    return {
        "probability": float(probability),
        "probability_pct": round(float(probability) * 100, 2),
        "threshold": threshold,
        "prediction": int(probability >= threshold),
        "calibration_applied": bool(calibration),
    }