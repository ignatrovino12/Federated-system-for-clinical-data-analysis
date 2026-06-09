import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from django.core.management.base import BaseCommand, CommandError
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

from clinical_ai import services
from clinical_ai.calibration import apply_calibration_state
from clinical_ai.models import PatientClinicalRecord


def _explicit_diabetes_label(record: PatientClinicalRecord) -> Optional[int]:
    if record.diabetes_status == PatientClinicalRecord.DiabetesStatus.HAS:
        return 1
    if record.diabetes_status == PatientClinicalRecord.DiabetesStatus.HAS_NOT:
        return 0
    return None


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_func,
    n_bootstrap: int,
    seed: int,
    requires_both_classes: bool = False,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    rng = np.random.default_rng(seed)
    values: List[float] = []
    sample_size = len(y_true)

    for _ in range(n_bootstrap):
        indices = rng.integers(0, sample_size, sample_size)
        y_sample = y_true[indices]
        if requires_both_classes and len(np.unique(y_sample)) < 2:
            continue
        try:
            metric_value = float(metric_func(y_sample, y_score[indices]))
            if np.isfinite(metric_value):
                values.append(metric_value)
        except Exception:
            continue

    if not values:
        return None, None, None

    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(np.mean(values)), float(lower), float(upper)


def _calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    if len(np.unique(y_true)) < 2:
        return None, None

    try:
        recalibration_model = LogisticRegression(
            solver="lbfgs",
            penalty=None,
            max_iter=2000,
        )
        recalibration_model.fit(logits, y_true.astype(int))
    except Exception:
        return None, None

    slope = float(recalibration_model.coef_.ravel()[0])
    intercept = float(recalibration_model.intercept_.ravel()[0])
    return slope, intercept


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for index in range(n_bins):
        left = bins[index]
        right = bins[index + 1]
        if index == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)

        count = int(mask.sum())
        if count == 0:
            continue

        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += abs(confidence - accuracy) * (count / total)

    return float(ece)


def _threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    denom = precision[:-1] + recall[:-1]
    f1_values = np.where(denom > 0, 2.0 * precision[:-1] * recall[:-1] / denom, 0.0)
    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index])


def _best_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    youden = tpr - fpr
    best_index = int(np.argmax(youden))
    value = float(thresholds[best_index])
    if not np.isfinite(value):
        return 0.5
    return min(max(value, 0.0), 1.0)


def _build_quality_block(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold_default: float,
    threshold_override: Optional[float],
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, object]:
    threshold_primary = float(threshold_override) if threshold_override is not None else threshold_default
    threshold_f1 = _best_f1_threshold(y_true, probabilities)
    threshold_youden = _best_youden_threshold(y_true, probabilities)

    roc_auc_value = float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else None
    pr_auc_value = float(average_precision_score(y_true, probabilities))
    brier_value = float(brier_score_loss(y_true, probabilities))
    calibration_slope, calibration_intercept = _calibration_slope_intercept(y_true, probabilities)
    ece_value = _expected_calibration_error(y_true, probabilities, n_bins=10)

    roc_auc_boot = _bootstrap_ci(
        y_true=y_true,
        y_score=probabilities,
        metric_func=roc_auc_score,
        n_bootstrap=bootstrap_samples,
        seed=seed,
        requires_both_classes=True,
    )
    pr_auc_boot = _bootstrap_ci(
        y_true=y_true,
        y_score=probabilities,
        metric_func=average_precision_score,
        n_bootstrap=bootstrap_samples,
        seed=seed + 1,
        requires_both_classes=False,
    )
    brier_boot = _bootstrap_ci(
        y_true=y_true,
        y_score=probabilities,
        metric_func=brier_score_loss,
        n_bootstrap=bootstrap_samples,
        seed=seed + 2,
        requires_both_classes=False,
    )

    frac_pos, mean_pred = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")

    return {
        "roc_auc": _safe_float(roc_auc_value),
        "roc_auc_ci95": {
            "mean_bootstrap": _safe_float(roc_auc_boot[0]),
            "lower": _safe_float(roc_auc_boot[1]),
            "upper": _safe_float(roc_auc_boot[2]),
        },
        "pr_auc": _safe_float(pr_auc_value),
        "pr_auc_ci95": {
            "mean_bootstrap": _safe_float(pr_auc_boot[0]),
            "lower": _safe_float(pr_auc_boot[1]),
            "upper": _safe_float(pr_auc_boot[2]),
        },
        "brier_score": _safe_float(brier_value),
        "brier_score_ci95": {
            "mean_bootstrap": _safe_float(brier_boot[0]),
            "lower": _safe_float(brier_boot[1]),
            "upper": _safe_float(brier_boot[2]),
        },
        "calibration_slope": _safe_float(calibration_slope),
        "calibration_intercept": _safe_float(calibration_intercept),
        "expected_calibration_error": _safe_float(ece_value),
        "threshold_metrics": {
            "default_or_override": _threshold_metrics(y_true, probabilities, threshold_primary),
            "best_f1": _threshold_metrics(y_true, probabilities, threshold_f1),
            "best_youden_j": _threshold_metrics(y_true, probabilities, threshold_youden),
        },
        "calibration_curve": {
            "mean_predicted_probability": [float(value) for value in mean_pred.tolist()],
            "fraction_of_positives": [float(value) for value in frac_pos.tolist()],
        },
    }


class Command(BaseCommand):
    help = "Compute Core Model Quality metrics (ROC AUC, PR AUC, calibration, threshold metrics)."

    def add_arguments(self, parser):
        parser.add_argument("--model", choices=["alex5050", "mustafa"], required=True)
        parser.add_argument("--test-size", type=float, default=0.2)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--threshold", type=float, default=None)
        parser.add_argument("--bootstrap-samples", type=int, default=1000)
        parser.add_argument("--output", type=str, default=None)

    def handle(self, *args, **options):
        model_name = options["model"]
        test_size = float(options["test_size"])
        seed = int(options["seed"])
        threshold_override = options["threshold"]
        bootstrap_samples = int(options["bootstrap_samples"])
        output_path = options["output"]

        if not (0.05 <= test_size <= 0.5):
            raise CommandError("--test-size must be between 0.05 and 0.5")
        if threshold_override is not None and not (0.0 <= float(threshold_override) <= 1.0):
            raise CommandError("--threshold must be between 0 and 1")
        if bootstrap_samples < 100:
            raise CommandError("--bootstrap-samples must be at least 100")

        if model_name == "alex5050":
            artifacts = services._load_alex_artifacts()
            feature_builder = lambda rec: rec.alex5050_features()
        else:
            artifacts = services._load_mustafa_artifacts()
            feature_builder = lambda rec: rec.mustafa_features()

        records = (
            PatientClinicalRecord.objects.select_related("patient")
            .filter(data_consent_for_training=True)
            .order_by("id")
        )

        feature_columns = artifacts["feature_columns"]
        scaler_feature_names = artifacts.get("scaler_feature_names")
        scaler_columns = list(scaler_feature_names) if scaler_feature_names is not None else list(feature_columns)

        features: List[List[float]] = []
        labels: List[int] = []
        record_ids: List[int] = []
        skipped_missing_label = 0
        skipped_missing_features = 0

        for record in records:
            label = _explicit_diabetes_label(record)
            if label is None:
                skipped_missing_label += 1
                continue

            payload = feature_builder(record)
            row = [payload.get(column) for column in scaler_columns]
            if any(value is None for value in row):
                skipped_missing_features += 1
                continue

            features.append([float(value) for value in row])
            labels.append(label)
            record_ids.append(record.id)

        if len(features) < 20:
            raise CommandError(
                f"Not enough complete labeled samples for evaluation: {len(features)} (need at least 20)."
            )

        x_all = np.asarray(features, dtype=np.float32)
        y_all = np.asarray(labels, dtype=np.int32)
        ids_all = np.asarray(record_ids, dtype=np.int64)

        class_counts = np.bincount(y_all, minlength=2)
        if class_counts.min() < 2:
            raise CommandError(
                "Need at least 2 samples in each class (HAS and HAS_NOT) to compute robust quality metrics."
            )

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        _, test_idx = next(splitter.split(x_all, y_all))

        x_test = x_all[test_idx]
        y_test = y_all[test_idx]
        ids_test = ids_all[test_idx]

        if scaler_feature_names is None:
            x_test_scaled = artifacts["scaler"].transform(x_test).astype(np.float32)
        else:
            x_test_scaled = artifacts["scaler"].transform(
                pd.DataFrame(x_test, columns=scaler_columns)
            ).astype(np.float32)

        model = artifacts["model"]
        device = artifacts["device"]
        outputs_probability = bool(artifacts.get("outputs_probability", False))
        calibration_state = artifacts.get("calibration")

        with torch.no_grad():
            tensor_x = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)
            outputs = model(tensor_x)
            raw_outputs = outputs.detach().cpu().numpy().ravel().astype(np.float64)

        probabilities = apply_calibration_state(
            raw_outputs,
            outputs_probability=outputs_probability,
            calibration_state=None,
        )
        calibrated_probabilities = apply_calibration_state(
            raw_outputs,
            outputs_probability=outputs_probability,
            calibration_state=calibration_state,
        )

        threshold_default = float(artifacts["threshold"])
        raw_quality = _build_quality_block(
            y_true=y_test,
            probabilities=probabilities,
            threshold_default=threshold_default,
            threshold_override=threshold_override,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        calibrated_quality = _build_quality_block(
            y_true=y_test,
            probabilities=calibrated_probabilities,
            threshold_default=threshold_default,
            threshold_override=threshold_override,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 100,
        )

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "dataset": {
                "consented_records_total": int(records.count()),
                "usable_records": int(len(features)),
                "evaluation_records": int(len(y_test)),
                "evaluation_positive": int((y_test == 1).sum()),
                "evaluation_negative": int((y_test == 0).sum()),
                "skipped_missing_label": int(skipped_missing_label),
                "skipped_missing_features": int(skipped_missing_features),
                "test_size": float(test_size),
                "seed": int(seed),
            },
            "core_model_quality": raw_quality,
            "calibrated_core_model_quality": calibrated_quality,
            "calibration": {
                "applied": bool(calibration_state and calibration_state.method != "identity"),
                "method": getattr(calibration_state, "method", "identity"),
                "source_model": getattr(calibration_state, "source_model", model_name),
                "calibration_size": getattr(calibration_state, "calibration_size", 0.0),
            },
            "evaluation_record_ids": [int(record_id) for record_id in ids_test.tolist()],
        }

        if output_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_dir = Path("reports") / "model_quality"
            output_dir.mkdir(parents=True, exist_ok=True)
            destination = output_dir / f"core_quality_{model_name}_{timestamp}.json"
        else:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)

        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")

        self.stdout.write(self.style.SUCCESS("Core Model Quality report generated"))
        self.stdout.write(f"Model: {model_name}")
        self.stdout.write(f"Records (eval): {len(y_test)}")
        self.stdout.write(f"ROC AUC (raw): {report['core_model_quality']['roc_auc']}")
        self.stdout.write(f"ROC AUC (calibrated): {report['calibrated_core_model_quality']['roc_auc']}")
        self.stdout.write(f"PR AUC (raw): {report['core_model_quality']['pr_auc']}")
        self.stdout.write(f"PR AUC (calibrated): {report['calibrated_core_model_quality']['pr_auc']}")
        self.stdout.write(f"Brier score (raw): {report['core_model_quality']['brier_score']}")
        self.stdout.write(f"Brier score (calibrated): {report['calibrated_core_model_quality']['brier_score']}")
        self.stdout.write(f"Output: {destination}")