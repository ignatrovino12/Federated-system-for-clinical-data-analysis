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
    temperature: float = 1.0
    intercept: float = 0.0
    decision_threshold: float = 0.5
    target_recall: float = 0.95
    fitted_at: str = ""
    n_samples: int = 0
    source_model: str = ""
    calibration_size: float = 0.0

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "temperature": float(self.temperature),
            "coef": float(self.temperature),
            "intercept": float(self.intercept),
            "decision_threshold": float(self.decision_threshold),
            "target_recall": float(self.target_recall),
            "fitted_at": self.fitted_at,
            "n_samples": int(self.n_samples),
            "source_model": self.source_model,
            "calibration_size": float(self.calibration_size),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CalibrationState":
        return cls(
            method=str(payload.get("method", "identity")),
            temperature=float(payload.get("temperature", payload.get("coef", 1.0))),
            intercept=float(payload.get("intercept", 0.0)),
            decision_threshold=float(payload.get("decision_threshold", 0.5)),
            target_recall=float(payload.get("target_recall", 0.95)),
            fitted_at=str(payload.get("fitted_at", "")),
            n_samples=int(payload.get("n_samples", 0)),
            source_model=str(payload.get("source_model", "")),
            calibration_size=float(payload.get("calibration_size", 0.0)),
        )


def identity_state(*, source_model: str = "", calibration_size: float = 0.0) -> CalibrationState:
    return CalibrationState(
        method="identity",
        temperature=1.0,
        intercept=0.0,
        decision_threshold=0.5,
        target_recall=0.95,
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
        temperature=1.0 / max(1e-6, abs(float(calibrator.coef_.ravel()[0]))),
        intercept=float(calibrator.intercept_.ravel()[0]),
        decision_threshold=0.5,
        target_recall=0.95,
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
        temperature=float(best_temperature),
        intercept=0.0,
        decision_threshold=0.5,
        target_recall=0.95,
        fitted_at=datetime.now(timezone.utc).isoformat(),
        n_samples=int(labels.size),
        source_model=source_model,
        calibration_size=float(calibration_size),
    )


def _decision_threshold_for_recall(y_true: np.ndarray, probabilities: np.ndarray, target_recall: float = 0.95) -> float:
    labels = np.asarray(y_true, dtype=np.int32).ravel()
    probs = np.asarray(probabilities, dtype=np.float64).ravel()
    if labels.size == 0 or probs.size == 0 or labels.size != probs.size:
        return 0.5

    thresholds = np.unique(np.concatenate(([0.0], probs, [1.0])))
    best_threshold = 0.5
    best_recall_gap = float("inf")
    best_specificity = -1.0

    for threshold in thresholds:
        predictions = (probs >= threshold).astype(int)
        tp = int(((predictions == 1) & (labels == 1)).sum())
        fp = int(((predictions == 1) & (labels == 0)).sum())
        tn = int(((predictions == 0) & (labels == 0)).sum())
        fn = int(((predictions == 0) & (labels == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        recall_gap = abs(recall - target_recall)
        if recall >= target_recall and (
            recall_gap < best_recall_gap or (recall_gap == best_recall_gap and specificity > best_specificity)
        ):
            best_threshold = float(threshold)
            best_recall_gap = recall_gap
            best_specificity = specificity

    if best_recall_gap < float("inf"):
        return best_threshold

    # Fallback: choose the threshold with the best recall, preferring specificity.
    best_threshold = 0.5
    best_recall = -1.0
    best_specificity = -1.0
    for threshold in thresholds:
        predictions = (probs >= threshold).astype(int)
        tp = int(((predictions == 1) & (labels == 1)).sum())
        fp = int(((predictions == 1) & (labels == 0)).sum())
        tn = int(((predictions == 0) & (labels == 0)).sum())
        fn = int(((predictions == 0) & (labels == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if recall > best_recall or (recall == best_recall and specificity > best_specificity):
            best_threshold = float(threshold)
            best_recall = recall
            best_specificity = specificity

    return best_threshold


def fit_temperature_bias_calibration(
    y_true: np.ndarray,
    raw_scores: np.ndarray,
    *,
    source_model: str = "",
    calibration_size: float = 0.0,
    target_recall: float = 0.95,
    temperature_grid: Optional[np.ndarray] = None,
    intercept_bounds: tuple[float, float] = (-0.2, 0.2),
    intercept_steps: int = 201,
) -> CalibrationState:
    labels = np.asarray(y_true, dtype=np.int32).ravel()
    scores = np.asarray(raw_scores, dtype=np.float64).ravel()

    if labels.size == 0 or scores.size == 0 or labels.size != scores.size:
        return identity_state(source_model=source_model, calibration_size=calibration_size)
    if np.unique(labels).size < 2:
        return identity_state(source_model=source_model, calibration_size=calibration_size)

    # Two-stage calibration:
    # 1) Fit temperature to preserve the slope behaviour.
    # 2) Fit only the intercept in a narrow band near zero so the baseline shift
    #    is corrected without letting the slope move again.
    temperature_state = fit_temperature_calibration(
        labels,
        scores,
        source_model=source_model,
        calibration_size=calibration_size,
        temperature_grid=temperature_grid,
    )

    best_temperature = float(max(1e-6, temperature_state.temperature))
    low, high = intercept_bounds
    if low > high:
        low, high = high, low
    intercept_grid = np.linspace(float(low), float(high), max(3, int(intercept_steps)))

    best_intercept = 0.0
    best_loss = float("inf")
    scaled_logits = scores / best_temperature
    for intercept in intercept_grid:
        calibrated = _sigmoid(scaled_logits + float(intercept))
        clipped = np.clip(calibrated, _EPSILON, 1.0 - _EPSILON)
        loss = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))
        if np.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best_intercept = float(intercept)

    probabilities = _sigmoid((scores / best_temperature) + best_intercept)
    decision_threshold = _decision_threshold_for_recall(labels, probabilities, target_recall=target_recall)

    return CalibrationState(
        method="temperature_bias",
        temperature=float(best_temperature),
        intercept=float(best_intercept),
        decision_threshold=float(decision_threshold),
        target_recall=float(target_recall),
        fitted_at=datetime.now(timezone.utc).isoformat(),
        n_samples=int(labels.size),
        source_model=source_model,
        calibration_size=float(calibration_size),
    )


def fit_bias_only_post_calibration(
    y_true: np.ndarray,
    raw_scores: np.ndarray,
    *,
    base_state: CalibrationState,
    source_model: str = "",
    calibration_size: float = 0.0,
    intercept_bounds: tuple[float, float] = (-0.1, 0.1),
    intercept_steps: int = 201,
    bias_penalty: float = 0.02,
) -> CalibrationState:
    labels = np.asarray(y_true, dtype=np.int32).ravel()
    scores = np.asarray(raw_scores, dtype=np.float64).ravel()

    if labels.size == 0 or scores.size == 0 or labels.size != scores.size:
        return base_state
    if np.unique(labels).size < 2:
        return base_state

    low, high = intercept_bounds
    if low > high:
        low, high = high, low

    intercept_grid = np.linspace(float(low), float(high), max(3, int(intercept_steps)))
    temperature = float(max(1e-6, base_state.temperature))
    base_logits = scores / temperature

    best_intercept = float(base_state.intercept)
    best_objective = float("inf")

    for intercept in intercept_grid:
        calibrated = _sigmoid(base_logits + float(intercept))
        clipped = np.clip(calibrated, _EPSILON, 1.0 - _EPSILON)
        log_loss = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))
        objective = log_loss + float(bias_penalty) * float(intercept * intercept)
        if np.isfinite(objective) and objective < best_objective:
            best_objective = objective
            best_intercept = float(intercept)

    probabilities = _sigmoid(base_logits + best_intercept)
    decision_threshold = _decision_threshold_for_recall(labels, probabilities, target_recall=base_state.target_recall)

    return CalibrationState(
        method="bias_only",
        temperature=temperature,
        intercept=float(best_intercept),
        decision_threshold=float(decision_threshold),
        target_recall=float(base_state.target_recall),
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
        return _sigmoid((raw_score / max(1e-6, calibration_state.temperature)) + calibration_state.intercept)

    if calibration_state.method == "temperature_bias":
        return _sigmoid((raw_score / max(1e-6, calibration_state.temperature)) + calibration_state.intercept)

    calibrated_logits = (raw_score / max(1e-6, calibration_state.temperature)) + calibration_state.intercept
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
