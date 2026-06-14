import os
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from urllib.parse import urlparse

import urllib3

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    Minio = None
    S3Error = Exception

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_ROOT = BASE_DIR / "machinelearning"


def _candidate_minio_targets(endpoint: str, use_ssl: bool) -> List[Tuple[str, bool]]:
    candidates: List[Tuple[str, bool]] = []

    def add_candidate(raw_endpoint: Optional[str], secure: bool) -> None:
        if not raw_endpoint:
            return
        cleaned = raw_endpoint.strip()
        if not cleaned:
            return

        parsed = urlparse(cleaned)
        if parsed.scheme in {"http", "https"}:
            host = parsed.netloc
            secure = parsed.scheme == "https"
        else:
            host = cleaned

        key = (host, secure)
        if key not in candidates:
            candidates.append(key)

    api_endpoint_override = os.getenv("MINIO_API_ENDPOINT")
    api_use_ssl = os.getenv("MINIO_API_USE_SSL")
    if api_endpoint_override:
        resolved_ssl = use_ssl if api_use_ssl is None else api_use_ssl.lower() == "true"
        add_candidate(api_endpoint_override, resolved_ssl)

    add_candidate(endpoint, use_ssl)

    if ":" in endpoint:
        host, port = endpoint.rsplit(":", 1)
        if port == "443":
            # Common misconfiguration: endpoint points to MinIO console over 443.
            add_candidate(f"{host}:9000", False)
            add_candidate(f"{host}:9000", True)
    else:
        add_candidate(f"{endpoint}:9000", False)
        add_candidate(f"{endpoint}:9000", True)

    return candidates


def get_local_federated_model_path(model_type: str) -> Path:
    model_type = model_type.lower()
    if model_type == "alex5050":
        return MODEL_ROOT / "alex5050_model_NN" / "latest.npz"
    if model_type == "mustafa":
        return MODEL_ROOT / "mustafa_model_NN" / "latest.npz"
    raise ValueError(f"Unknown model type: {model_type}")


def create_minio_client(
    endpoint: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    use_ssl: Optional[bool] = None,
) -> object:

    if Minio is None:
        raise RuntimeError(
            "MinIO Python package is not installed. Install `minio` to enable federated model sync."
        )

    endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = access_key or os.getenv("MINIO_ACCESS_KEY")
    secret_key = secret_key or os.getenv("MINIO_SECRET_KEY")
    if use_ssl is None:
        use_ssl = os.getenv("MINIO_USE_SSL", "false").lower() == "true"

    connect_timeout = float(os.getenv("MINIO_CONNECT_TIMEOUT_SECONDS", "2.0"))
    read_timeout = float(os.getenv("MINIO_READ_TIMEOUT_SECONDS", "3.0"))
    total_retries = int(os.getenv("MINIO_RETRY_TOTAL", "0"))

    http_client = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=connect_timeout, read=read_timeout),
        retries=urllib3.Retry(total=total_retries, redirect=False),
    )

    return Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=use_ssl,
        http_client=http_client,
    )


def _download_object_bytes(
    *,
    bucket_name: str,
    object_name: str,
    endpoint: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    use_ssl: Optional[bool] = None,
) -> bytes:
    resolved_endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
    resolved_use_ssl = (os.getenv("MINIO_USE_SSL", "false").lower() == "true") if use_ssl is None else use_ssl
    resolved_access_key = access_key or os.getenv("MINIO_ACCESS_KEY")
    resolved_secret_key = secret_key or os.getenv("MINIO_SECRET_KEY")

    errors: List[str] = []
    for candidate_endpoint, candidate_ssl in _candidate_minio_targets(resolved_endpoint, resolved_use_ssl):
        try:
            client = create_minio_client(
                endpoint=candidate_endpoint,
                access_key=resolved_access_key,
                secret_key=resolved_secret_key,
                use_ssl=candidate_ssl,
            )
            response = client.get_object(bucket_name=bucket_name, object_name=object_name)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except S3Error as exc:
            errors.append(f"{candidate_endpoint} (ssl={candidate_ssl}): {exc}")
        except Exception as exc:
            errors.append(f"{candidate_endpoint} (ssl={candidate_ssl}): {exc}")

    raise RuntimeError(
        f"Could not download {object_name} from bucket {bucket_name}. Tried endpoints: " + " | ".join(errors)
    )


def load_json_object(
    *,
    bucket_name: str,
    object_name: str,
    endpoint: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    use_ssl: Optional[bool] = None,
) -> Dict[str, Any]:
    raw_bytes = _download_object_bytes(
        bucket_name=bucket_name,
        object_name=object_name,
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        use_ssl=use_ssl,
    )
    return json.loads(raw_bytes.decode("utf-8"))


def download_latest_model(
    model_type: str,
    bucket_name: str = "models",
    endpoint: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    use_ssl: Optional[bool] = None,
) -> Path:
    remote_object = f"models/{model_type.lower()}/latest.npz"
    local_path = get_local_federated_model_path(model_type)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        payload = _download_object_bytes(
            bucket_name=bucket_name,
            object_name=remote_object,
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            use_ssl=use_ssl,
        )
        local_path.write_bytes(payload)
    except RuntimeError as exc:
        raise FileNotFoundError(f"Could not download {remote_object} from bucket {bucket_name}: {exc}") from exc

    return local_path