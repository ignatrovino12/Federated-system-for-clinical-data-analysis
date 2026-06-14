import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from django.core.management.base import BaseCommand, CommandError

from clinical_ai.models import PatientClinicalRecord


def _sampling_strength() -> float:
    raw_value = os.getenv("FEDERATED_SAMPLING_STRENGTH", "1.0")
    try:
        value = float(raw_value)
    except ValueError:
        value = 1.0
    return max(0.0, min(4.0, value))


def _training_sample_fraction() -> float:
    raw_value = os.getenv("FEDERATED_TRAINING_SAMPLE_FRACTION", "0.5")
    try:
        value = float(raw_value)
    except ValueError:
        value = 0.5
    return max(0.1, min(1.0, value))


def _explicit_diabetes_label(record: PatientClinicalRecord) -> Optional[float]:
    """Use explicit diagnosis status as federated target label."""
    if record.diabetes_status == PatientClinicalRecord.DiabetesStatus.HAS:
        return 1.0
    if record.diabetes_status == PatientClinicalRecord.DiabetesStatus.HAS_NOT:
        return 0.0
    return None


def _build_feature_row(record: PatientClinicalRecord, model_type: str) -> Optional[List[float]]:
    payload: Dict[str, Optional[float]]
    if model_type == "alex5050":
        payload = record.alex5050_features()
    else:
        payload = record.mustafa_features()

    if any(value is None for value in payload.values()):
        return None

    return [float(value) for value in payload.values()]


def _training_sample_weight(record: PatientClinicalRecord) -> float:
    """Downweight frequently used records; strength controls how aggressively."""
    count = float(record.federated_train_count or 0)
    strength = _sampling_strength()
    return 1.0 / ((1.0 + count) ** strength)


def _load_feature_scaler(model_type: str):
    base_dir = Path(__file__).resolve().parents[3] / "machinelearning"
    if model_type == "alex5050":
        scaler_path = base_dir / "alex5050_model_NN" / "alex5050_scaler.joblib"
    else:
        candidate_paths = [
            base_dir / "mustafa_model_NN" / "scaler.joblib",
            base_dir / "mustafa_model" / "scaler.joblib",
        ]
        scaler_path = next((path for path in candidate_paths if path.exists()), candidate_paths[0])

    if not scaler_path.exists():
        raise CommandError(f"Could not find scaler for {model_type} at {scaler_path}")

    return joblib.load(scaler_path)


def _load_root_certificates(root_certificates: Optional[str]) -> Optional[bytes]:
    if not root_certificates:
        return None

    root_path = Path(root_certificates)
    if root_path.exists():
        return root_path.read_bytes()
    return root_certificates.encode("utf-8")


class Command(BaseCommand):
    help = "Train locally on clinic data and connect this Django instance as a Flower client."

    def add_arguments(self, parser):
        parser.add_argument(
            "--model",
            choices=["alex5050", "mustafa"],
            default="alex5050",
            help="Model architecture to train federated.",
        )
        parser.add_argument(
            "--server-address",
            default=os.getenv("FLOWER_SERVER_ADDRESS", "flower.vladar34.xyz:443"),
            help="Flower server address (host:port). Defaults to FLOWER_SERVER_ADDRESS or flower.vladar34.xyz:443.",
        )
        parser.add_argument(
            "--secure",
            dest="insecure",
            action="store_false",
            help="Use a secure TLS connection for Flower gRPC.",
        )
        parser.add_argument(
            "--insecure",
            dest="insecure",
            action="store_true",
            help="Use an insecure gRPC connection for Flower.",
        )
        parser.set_defaults(insecure=os.getenv("FLOWER_INSECURE", "false").lower() == "true")
        parser.add_argument(
            "--root-certificates",
            default=os.getenv("FLOWER_ROOT_CERTIFICATES"),
            help="Path to PEM root certificates for TLS validation.",
        )
        parser.add_argument(
            "--min-samples",
            type=int,
            default=1,
            help="Minimum local samples required to start federated training.",
        )
        parser.add_argument(
            "--test-split",
            type=float,
            default=0.2,
            help="Fraction of local samples used for evaluation.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Batch size for local training.",
        )
        parser.add_argument(
            "--sync-latest-model",
            action="store_true",
            help="Download the latest federated model from MinIO before training starts.",
        )
        parser.add_argument(
            "--minio-endpoint",
            default=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            help="MinIO endpoint used when syncing the latest model.",
        )
        parser.add_argument(
            "--minio-access-key",
            default=os.getenv("MINIO_ACCESS_KEY"),
            help="MinIO access key used when syncing the latest model.",
        )
        parser.add_argument(
            "--minio-secret-key",
            default=os.getenv("MINIO_SECRET_KEY"),
            help="MinIO secret key used when syncing the latest model.",
        )
        parser.add_argument(
            "--minio-use-ssl",
            action="store_true",
            default=os.getenv("MINIO_USE_SSL", "false").lower() == "true",
            help="Use SSL when connecting to MinIO.",
        )
        parser.add_argument(
            "--minio-bucket-name",
            default=os.getenv("MINIO_BUCKET_NAME", "models"),
            help="MinIO bucket that stores federated checkpoints.",
        )

    def handle(self, *args, **options):
        try:
            import flwr as fl
        except ImportError as exc:
            raise CommandError(
                "flwr is not installed in this environment. Install federated dependencies first."
            ) from exc

        from clinical_ai.federated_client import create_client

        model_type: str = options["model"]
        server_address: str = options["server_address"]
        min_samples: int = options["min_samples"]
        test_split: float = options["test_split"]
        batch_size: int = options["batch_size"]

        if not (0.05 <= test_split <= 0.5):
            raise CommandError("--test-split must be between 0.05 and 0.5")

        if options["sync_latest_model"]:
            try:
                from clinical_ai.federated_model_sync import download_latest_model

                synced_path = download_latest_model(
                    model_type=model_type,
                    bucket_name=options["minio_bucket_name"],
                    endpoint=options["minio_endpoint"],
                    access_key=options["minio_access_key"],
                    secret_key=options["minio_secret_key"],
                    use_ssl=options["minio_use_ssl"],
                )
                self.stdout.write(self.style.SUCCESS(f"Synced latest federated model to {synced_path}"))
            except FileNotFoundError as exc:
                self.stdout.write(
                    self.style.WARNING(
                        f"No existing federated model was found in MinIO for {model_type}; starting from scratch. {exc}"
                    )
                )
            except Exception as exc:
                raise CommandError(f"Failed to sync latest federated model: {exc}") from exc

        records = PatientClinicalRecord.objects.select_related("patient").all().order_by("id")

        features: List[List[float]] = []
        labels: List[float] = []
        sample_weights: List[float] = []
        record_ids: List[int] = []

        skipped_missing_features = 0
        skipped_missing_label = 0
        skipped_no_consent = 0

        for record in records:
            if not record.data_consent_for_training:
                skipped_no_consent += 1
                continue

            row = _build_feature_row(record, model_type=model_type)
            if row is None:
                skipped_missing_features += 1
                continue

            label = _explicit_diabetes_label(record)
            if label is None:
                skipped_missing_label += 1
                continue

            features.append(row)
            labels.append(label)
            sample_weights.append(_training_sample_weight(record))
            # Capture federated training counters for debugging per-clinic
            try:
                # store 0 when None for clearer stats
                counts_list.append(float(record.federated_train_count or 0))
            except NameError:
                counts_list = [float(record.federated_train_count or 0)]
            record_ids.append(record.id)

        if len(features) < min_samples:
            message = (
                f"Not enough local samples for federated training: {len(features)} < {min_samples}. "
                f"Skipped missing features: {skipped_missing_features}, "
                f"skipped missing labels: {skipped_missing_label}, "
                f"skipped no consent: {skipped_no_consent}."
            )
            if len(features) == 0:
                self.stdout.write(self.style.WARNING(message))
                self.stdout.write(
                    self.style.WARNING(
                        f"No complete {model_type} records were found. Add a clinical record with all required "
                        f"fields or switch to a model that matches the available data."
                    )
                )
                return

            raise CommandError(message)

        x = np.asarray(features, dtype=np.float32)
        scaler = _load_feature_scaler(model_type)
        x = scaler.transform(x).astype(np.float32)
        y = np.asarray(labels, dtype=np.float32)
        sample_weight_array = np.asarray(sample_weights, dtype=np.float32)

        # Emit debug info about federated_train_count distribution for this clinic
        try:
            counts = np.asarray(counts_list, dtype=np.float32)
            zeros = int((counts == 0).sum())
            total = counts.size
            uniq = np.unique(counts)
            self.stdout.write(
                self.style.NOTICE(
                    f"federated_train_count stats: min={float(counts.min())}, max={float(counts.max())}, zeros={zeros}/{total} ({zeros/total:.2%}), unique_sample={uniq[:10].tolist()}"
                )
            )
        except Exception:
            pass

        # Shuffle the dataset before splitting into train/test so the training split isn't deterministically biased by database id ordering.
        if len(x) > 1:
            perm = np.random.RandomState(seed=None).permutation(len(x))
            x = x[perm]
            y = y[perm]
            sample_weight_array = sample_weight_array[perm]
            record_ids = [record_ids[i] for i in perm]

        self.stdout.write(self.style.SUCCESS("Prepared local federated dataset"))
        self.stdout.write(
            f"Model: {model_type} | Samples: {len(x)} | "
            f"Skipped features: {skipped_missing_features} | "
            f"Skipped labels: {skipped_missing_label} | Skipped no consent: {skipped_no_consent}"
        )
        self.stdout.write(f"Sampling strength: {_sampling_strength():.2f}")
        self.stdout.write(f"Training sample fraction: {_training_sample_fraction():.2f}")
        self.stdout.write(f"Connecting to Flower server: {server_address}")

        client = create_client(
            model_type=model_type,
            local_data=(x, y),
            local_sample_weights=sample_weight_array,
            record_ids=record_ids,
            test_split=test_split,
            batch_size=batch_size,
            sampling_strength=_sampling_strength(),
            training_sample_fraction=_training_sample_fraction(),
        )

        fl.client.start_client(
            server_address=server_address,
            client=client.to_client(),
            insecure=options["insecure"],
            root_certificates=_load_root_certificates(options["root_certificates"]),
        )
