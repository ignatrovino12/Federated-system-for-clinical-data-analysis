from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from django.core.management.base import BaseCommand, CommandError
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedShuffleSplit

from clinical_ai import services
from clinical_ai.calibration import (
    fit_bias_only_post_calibration,
    apply_calibration_state,
    fit_platt_calibration,
    fit_temperature_bias_calibration,
    fit_temperature_calibration,
    save_calibration_state,
)
from clinical_ai.models import PatientClinicalRecord


def _explicit_diabetes_label(record: PatientClinicalRecord) -> Optional[int]:
    if record.diabetes_status == PatientClinicalRecord.DiabetesStatus.HAS:
        return 1
    if record.diabetes_status == PatientClinicalRecord.DiabetesStatus.HAS_NOT:
        return 0
    return None


class Command(BaseCommand):
    help = "Fit a post-hoc Platt calibrator for a model and save it next to the artifacts."

    def add_arguments(self, parser):
        parser.add_argument("--model", choices=["alex5050", "mustafa"], required=True)
        parser.add_argument("--calibration-size", type=float, default=0.3)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--method",
            choices=["temperature_bias", "temperature", "platt"],
            default="temperature_bias",
            help="Calibration method to fit. temperature_bias is usually the safest default.",
        )
        parser.add_argument(
            "--target-recall",
            type=float,
            default=0.95,
            help="Recall target used when selecting the safety threshold.",
        )
        parser.add_argument("--output", type=str, default=None)

    def handle(self, *args, **options):
        model_name = options["model"]
        calibration_size = float(options["calibration_size"])
        seed = int(options["seed"])
        method = options["method"]
        target_recall = float(options["target_recall"])
        output_path = options["output"]

        if not (0.05 <= calibration_size <= 0.5):
            raise CommandError("--calibration-size must be between 0.05 and 0.5")

        if model_name == "alex5050":
            artifacts = services._load_alex_artifacts()
            feature_builder = lambda rec: rec.alex5050_features()
            default_output = Path("machinelearning") / "alex5050_model_NN" / "calibration.json"
        else:
            artifacts = services._load_mustafa_artifacts()
            feature_builder = lambda rec: rec.mustafa_features()
            default_output = Path("machinelearning") / "mustafa_model_NN" / "calibration.json"

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

        for record in records:
            label = _explicit_diabetes_label(record)
            if label is None:
                continue

            payload = feature_builder(record)
            row = [payload.get(column) for column in scaler_columns]
            if any(value is None for value in row):
                continue

            features.append([float(value) for value in row])
            labels.append(label)

        if len(features) < 20:
            raise CommandError(f"Not enough complete labeled samples to fit calibration: {len(features)}")

        x_all = np.asarray(features, dtype=np.float32)
        y_all = np.asarray(labels, dtype=np.int32)

        class_counts = np.bincount(y_all, minlength=2)
        if class_counts.min() < 2:
            raise CommandError("Need at least 2 samples in each class to fit a calibrator.")

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=calibration_size, random_state=seed)
        _, calibration_idx = next(splitter.split(x_all, y_all))

        x_calibration = x_all[calibration_idx]
        y_calibration = y_all[calibration_idx]

        if scaler_feature_names is None:
            x_scaled = artifacts["scaler"].transform(x_calibration).astype(np.float32)
        else:
            x_scaled = artifacts["scaler"].transform(pd.DataFrame(x_calibration, columns=scaler_columns)).astype(np.float32)

        model = artifacts["model"]
        device = artifacts["device"]
        outputs_probability = bool(artifacts.get("outputs_probability", False))

        with torch.no_grad():
            tensor_x = torch.tensor(x_scaled, dtype=torch.float32).to(device)
            outputs = model(tensor_x).detach().cpu().numpy().ravel().astype(np.float64)

        raw_probabilities = apply_calibration_state(
            outputs,
            outputs_probability=outputs_probability,
            calibration_state=None,
        )
        if method == "temperature":
            calibration_state = fit_temperature_calibration(
                y_calibration,
                outputs,
                source_model=model_name,
                calibration_size=calibration_size,
            )
        elif method == "temperature_bias":
            calibration_state = fit_temperature_bias_calibration(
                y_calibration,
                outputs,
                source_model=model_name,
                calibration_size=calibration_size,
                target_recall=target_recall,
            )
            calibration_state = fit_bias_only_post_calibration(
                y_calibration,
                outputs,
                base_state=calibration_state,
                source_model=model_name,
                calibration_size=calibration_size,
                intercept_bounds=(-0.5, 0.5),
                intercept_steps=201,
                bias_penalty=0.03,
            )
        else:
            calibration_state = fit_platt_calibration(
                y_calibration,
                outputs,
                source_model=model_name,
                calibration_size=calibration_size,
            )
        calibrated_probabilities = apply_calibration_state(
            outputs,
            outputs_probability=outputs_probability,
            calibration_state=calibration_state,
        )

        before_brier = float(brier_score_loss(y_calibration, raw_probabilities))
        after_brier = float(brier_score_loss(y_calibration, calibrated_probabilities))

        destination = Path(output_path) if output_path is not None else default_output
        save_calibration_state(destination, calibration_state)

        self.stdout.write(self.style.SUCCESS("Calibration fitted and saved"))
        self.stdout.write(f"Model: {model_name}")
        self.stdout.write(f"Method: {method}")
        self.stdout.write(f"Calibration samples: {len(y_calibration)}")
        self.stdout.write(f"Raw Brier: {before_brier:.6f}")
        self.stdout.write(f"Calibrated Brier: {after_brier:.6f}")
        self.stdout.write(f"Safety threshold: {calibration_state.decision_threshold:.6f}")
        self.stdout.write(f"Output: {destination}")
