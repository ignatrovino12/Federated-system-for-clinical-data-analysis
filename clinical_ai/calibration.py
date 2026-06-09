from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression


_EPSILON = 1e-6


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def probability_to_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, _EPSILON, 1.0 - _EPSILON)
    return np.log(clipped / (1.0 - clipped))


@dataclass(frozen=True)
class CalibrationState:
    method: str = "identity"
    coef: float = 1.0
    intercept: float = 0.0
    fitted_at: str = ""
    n_samples: int = 0
    source_model: str = ""
    calibration_size: float = 0.0

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "coef": float(self.coef),
            "intercept": float(self.intercept),
            "fitted_at": self.fitted_at,
            "n_samples": int(self.n_samples),
            "source_model": self.source_model,
            "calibration_size": float(self.calibration_size),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CalibrationState":
        return cls(
            method=str(payload.get("method", "identity")),
            coef=float(payload.get("coef", 1.0)),
            intercept=float(payload.get("intercept", 0.0)),
            fitted_at=str(payload.get("fitted_at", "")),
            n_samples=int(payload.get("n_samples", 0)),
            source_model=str(payload.get("source_model", "")),
            calibration_size=float(payload.get("calibration_size", 0.0)),
        )


def identity_state(*, source_model: str = "", calibration_size: float = 0.0) -> CalibrationState:
    return CalibrationState(
        method="identity",
        coef=1.0,
        intercept=0.0,
        fitted_at=datetime.now(timezone.utc).isoformat(),
        n_samples=0,
        source_model=source_model,
        calibration_size=calibration_size,
    )


def fit_platt_calibration(
    y_true: np.ndarray,
    raw_scores: np.ndarray,
    *,
    source_model: str = "",
    calibration_size: float = 0.0,
    max_iter: int = 1000,
) -> CalibrationState:
    labels = np.asarray(y_true, dtype=np.int32).ravel()
    scores = np.asarray(raw_scores, dtype=np.float64).ravel()

    if labels.size == 0 or scores.size == 0 or labels.size != scores.size:
        return identity_state(source_model=source_model, calibration_size=calibration_size)
    if np.unique(labels).size < 2:
        return identity_state(source_model=source_model, calibration_size=calibration_size)

    calibrator = LogisticRegression(solver="lbfgs", max_iter=max_iter)
    calibrator.fit(scores.reshape(-1, 1), labels)

    return CalibrationState(
        method="platt",
        coef=float(calibrator.coef_.ravel()[0]),
        intercept=float(calibrator.intercept_.ravel()[0]),
        fitted_at=datetime.now(timezone.utc).isoformat(),
        n_samples=int(labels.size),
        source_model=source_model,
        calibration_size=float(calibration_size),
    )


def fit_temperature_calibration(
    y_true: np.ndarray,
    raw_scores: np.ndarray,
    *,
    source_model: str = "",
    calibration_size: float = 0.0,
    temperature_grid: Optional[np.ndarray] = None,
) -> CalibrationState:
    labels = np.asarray(y_true, dtype=np.int32).ravel()
    scores = np.asarray(raw_scores, dtype=np.float64).ravel()

    if labels.size == 0 or scores.size == 0 or labels.size != scores.size:
        return identity_state(source_model=source_model, calibration_size=calibration_size)
    if np.unique(labels).size < 2:
        return identity_state(source_model=source_model, calibration_size=calibration_size)

    grid = temperature_grid
    if grid is None:
        grid = np.concatenate([
            np.linspace(0.5, 2.0, 16),
            np.linspace(2.5, 10.0, 16),
            np.linspace(12.0, 50.0, 20),
        ])

    best_temperature = 1.0
    best_loss = float("inf")

    for temperature in grid:
        temp = float(temperature)
        if temp <= 0:
            continue
        scaled = _sigmoid(scores / temp)
        clipped = np.clip(scaled, _EPSILON, 1.0 - _EPSILON)
        loss = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))
        if np.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best_temperature = temp

    return CalibrationState(
        method="temperature",
        coef=float(1.0 / best_temperature),
        intercept=0.0,
        fitted_at=datetime.now(timezone.utc).isoformat(),
        n_samples=int(labels.size),
        source_model=source_model,
        calibration_size=float(calibration_size),
    )


def apply_calibration_state(
    model_output: np.ndarray,
    *,
    outputs_probability: bool,
    calibration_state: Optional[CalibrationState],
) -> np.ndarray:
    raw_output = np.asarray(model_output, dtype=np.float64)
    if outputs_probability:
        raw_score = probability_to_logit(raw_output)
    else:
        raw_score = raw_output

    if calibration_state is None or calibration_state.method == "identity":
        return _sigmoid(raw_score)

    if calibration_state.method == "temperature":
        return _sigmoid(raw_score * calibration_state.coef)

    calibrated_logits = (calibration_state.coef * raw_score) + calibration_state.intercept
    return _sigmoid(calibrated_logits)


def save_calibration_state(path: Path, calibration_state: CalibrationState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration_state.to_dict(), indent=2), encoding="utf-8")


def load_calibration_state(path: Path) -> Optional[CalibrationState]:
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    return CalibrationState.from_dict(payload)
