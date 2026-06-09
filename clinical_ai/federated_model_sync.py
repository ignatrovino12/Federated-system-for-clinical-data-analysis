import os
from pathlib import Path
from typing import Optional
from minio import Minio
from minio.error import S3Error

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_ROOT = BASE_DIR / "machinelearning"


def get_local_federated_model_path(model_type: str) -> Path:
    model_type = model_type.lower()
    if model_type == "alex5050":
        return MODEL_ROOT / "alex5050_model_NN" / "latest.pt"
    if model_type == "mustafa":
        return MODEL_ROOT / "mustafa_model_NN" / "latest.pt"
    raise ValueError(f"Unknown model type: {model_type}")


def create_minio_client(
    endpoint: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    use_ssl: Optional[bool] = None,
) -> object:

    endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = access_key or os.getenv("MINIO_ACCESS_KEY")
    secret_key = secret_key or os.getenv("MINIO_SECRET_KEY")
    if use_ssl is None:
        use_ssl = os.getenv("MINIO_USE_SSL", "false").lower() == "true"

    return Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=use_ssl,
    )


def download_latest_model(
    model_type: str,
    bucket_name: str = "models",
    endpoint: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    use_ssl: Optional[bool] = None,
) -> Path:

    client = create_minio_client(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        use_ssl=use_ssl,
    )

    remote_object = f"models/{model_type.lower()}/latest.pt"
    local_path = get_local_federated_model_path(model_type)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = client.get_object(bucket_name=bucket_name, object_name=remote_object)
        try:
            local_path.write_bytes(response.read())
        finally:
            response.close()
            response.release_conn()
    except S3Error as exc:
        raise FileNotFoundError(f"Could not download {remote_object} from bucket {bucket_name}: {exc}") from exc

    return local_path