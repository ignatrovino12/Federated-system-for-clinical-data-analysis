from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import requests
from django.core.management.base import BaseCommand, CommandError

PrometheusSample = Dict[str, object]
QueryFetcher = Callable[[str], List[PrometheusSample]]

DEFAULT_PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:30990")


def _safe_float(value: object) -> Optional[float]:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if numeric_value != numeric_value or numeric_value in (float("inf"), float("-inf")):
        return None
    return float(numeric_value)


def _parse_window_to_seconds(window: str) -> Optional[int]:
    match = re.fullmatch(r"(\d+)([smhdw])", window.strip())
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    return amount * multipliers[unit]


def _build_label_selector(clinics: Optional[Sequence[str]] = None, models: Optional[Sequence[str]] = None) -> str:
    parts: List[str] = []
    if clinics:
        clinic_pattern = "|".join(re.escape(item.strip()) for item in clinics if item and item.strip())
        if clinic_pattern:
            parts.append(f'clinic=~"{clinic_pattern}"')
    if models:
        model_pattern = "|".join(re.escape(item.strip()) for item in models if item and item.strip())
        if model_pattern:
            parts.append(f'model=~"{model_pattern}"')
    return f"{{{','.join(parts)}}}" if parts else ""


def _query_prometheus(prometheus_url: str, query: str, timeout_seconds: int = 30) -> List[PrometheusSample]:
    response = requests.get(
        f"{prometheus_url.rstrip('/')}/api/v1/query",
        params={"query": query},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
        raise CommandError(f"Prometheus query failed for {query!r}: {payload!r}")
    data = payload.get("data", {})
    if not isinstance(data, dict):
        return []
    result = data.get("result", [])
    return result if isinstance(result, list) else []


def _sample_value(sample: PrometheusSample) -> Optional[float]:
    value = sample.get("value")
    if not isinstance(value, list) or len(value) < 2:
        return None
    return _safe_float(value[1])


def _labels(sample: PrometheusSample, keys: Sequence[str]) -> Tuple[str, ...]:
    metric = sample.get("metric", {})
    if not isinstance(metric, dict):
        return tuple("" for _ in keys)
    return tuple(str(metric.get(key, "")) for key in keys)


def _vector_to_map(samples: Sequence[PrometheusSample], keys: Sequence[str]) -> Dict[Tuple[str, ...], float]:
    output: Dict[Tuple[str, ...], float] = {}
    for sample in samples:
        value = _sample_value(sample)
        if value is None:
            continue
        output[_labels(sample, keys)] = value
    return output


def _vector_to_status_map(samples: Sequence[PrometheusSample], keys: Sequence[str]) -> Dict[Tuple[str, ...], Dict[str, float]]:
    output: Dict[Tuple[str, ...], Dict[str, float]] = {}
    for sample in samples:
        value = _sample_value(sample)
        if value is None:
            continue
        metric = sample.get("metric", {})
        if not isinstance(metric, dict):
            continue
        key = tuple(str(metric.get(item, "")) for item in keys)
        status = str(metric.get("status", "unknown"))
        output.setdefault(key, {})[status] = value
    return output


def _sum_statuses(values_by_status: Mapping[str, float]) -> float:
    return float(sum(values_by_status.values()))


def _average_from_sum_and_count(sum_value: Optional[float], count_value: Optional[float]) -> Optional[float]:
    if sum_value is None or count_value is None or count_value == 0:
        return None
    return float(sum_value / count_value)


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(mean(filtered))


def _clinic_row(
    clinic: str,
    model: str,
    fit_rounds: Mapping[str, float],
    evaluate_rounds: Mapping[str, float],
    fit_duration_sum: Optional[float],
    fit_duration_count: Optional[float],
    evaluate_duration_sum: Optional[float],
    evaluate_duration_count: Optional[float],
    bytes_sent: Optional[float],
    bytes_received: Optional[float],
    training_loss: Optional[float],
    evaluation_accuracy: Optional[float],
    evaluation_loss: Optional[float],
) -> Dict[str, Any]:
    fit_failures = sum(value for status, value in fit_rounds.items() if status != "success")
    evaluate_failures = sum(value for status, value in evaluate_rounds.items() if status != "success")
    return {
        "clinic": clinic,
        "model": model,
        "fit_rounds": {
            "by_status": dict(sorted(fit_rounds.items())),
            "success": float(fit_rounds.get("success", 0.0)),
            "failures": float(fit_failures),
            "total": float(_sum_statuses(fit_rounds)),
        },
        "evaluate_rounds": {
            "by_status": dict(sorted(evaluate_rounds.items())),
            "success": float(evaluate_rounds.get("success", 0.0)),
            "failures": float(evaluate_failures),
            "total": float(_sum_statuses(evaluate_rounds)),
        },
        "fit_duration_seconds": {
            "sum": fit_duration_sum,
            "count": fit_duration_count,
            "average": _average_from_sum_and_count(fit_duration_sum, fit_duration_count),
        },
        "evaluate_duration_seconds": {
            "sum": evaluate_duration_sum,
            "count": evaluate_duration_count,
            "average": _average_from_sum_and_count(evaluate_duration_sum, evaluate_duration_count),
        },
        "bytes_sent": bytes_sent,
        "bytes_received": bytes_received,
        "communication_bytes": None if bytes_sent is None or bytes_received is None else float(bytes_sent + bytes_received),
        "training_loss": training_loss,
        "evaluation_accuracy": evaluation_accuracy,
        "evaluation_loss": evaluation_loss,
    }


def _flower_row(
    model: str,
    rounds: Mapping[str, float],
    duration_sum: Optional[float],
    duration_count: Optional[float],
    clients_sum: Optional[float],
    clients_count: Optional[float],
    current_round: Optional[float],
    last_checkpoint_round: Optional[float],
    checkpoint_saves: Optional[float],
    checkpoint_failures: Optional[float],
) -> Dict[str, Any]:
    failed_rounds = sum(value for status, value in rounds.items() if status != "success")
    return {
        "model": model,
        "rounds": {
            "by_status": dict(sorted(rounds.items())),
            "success": float(rounds.get("success", 0.0)),
            "failures": float(failed_rounds),
            "total": float(_sum_statuses(rounds)),
        },
        "round_duration_seconds": {
            "sum": duration_sum,
            "count": duration_count,
            "average": _average_from_sum_and_count(duration_sum, duration_count),
        },
        "clients_per_round": {
            "sum": clients_sum,
            "count": clients_count,
            "average": _average_from_sum_and_count(clients_sum, clients_count),
        },
        "current_round": current_round,
        "last_checkpoint_round": last_checkpoint_round,
        "checkpoint_saves": checkpoint_saves,
        "checkpoint_failures": checkpoint_failures,
    }


def build_system_efficiency_report(
    fetch: QueryFetcher,
    *,
    window: str,
    clinics: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    selector = _build_label_selector(clinics=clinics, models=models)
    model_selector = _build_label_selector(models=models)

    fit_rounds = _vector_to_status_map(
        fetch(f"sum by (clinic, model, status) (increase(healthcheck_client_fit_rounds_total{selector}[{window}]))"),
        ["clinic", "model"],
    )
    evaluate_rounds = _vector_to_status_map(
        fetch(f"sum by (clinic, model, status) (increase(healthcheck_client_evaluate_rounds_total{selector}[{window}]))"),
        ["clinic", "model"],
    )
    fit_duration_sum = _vector_to_map(
        fetch(f"sum by (clinic, model) (increase(healthcheck_client_fit_duration_seconds_sum{selector}[{window}]))"),
        ["clinic", "model"],
    )
    fit_duration_count = _vector_to_map(
        fetch(f"sum by (clinic, model) (increase(healthcheck_client_fit_duration_seconds_count{selector}[{window}]))"),
        ["clinic", "model"],
    )
    evaluate_duration_sum = _vector_to_map(
        fetch(f"sum by (clinic, model) (increase(healthcheck_client_evaluate_duration_seconds_sum{selector}[{window}]))"),
        ["clinic", "model"],
    )
    evaluate_duration_count = _vector_to_map(
        fetch(f"sum by (clinic, model) (increase(healthcheck_client_evaluate_duration_seconds_count{selector}[{window}]))"),
        ["clinic", "model"],
    )
    bytes_sent = _vector_to_map(
        fetch(f"sum by (clinic, model) (increase(healthcheck_client_parameters_sent_bytes_total{selector}[{window}]))"),
        ["clinic", "model"],
    )
    bytes_received = _vector_to_map(
        fetch(f"sum by (clinic, model) (increase(healthcheck_client_parameters_received_bytes_total{selector}[{window}]))"),
        ["clinic", "model"],
    )
    training_loss = _vector_to_map(fetch(f"healthcheck_client_training_loss{selector}"), ["clinic", "model"])
    evaluation_accuracy = _vector_to_map(fetch(f"healthcheck_client_evaluation_accuracy{selector}"), ["clinic", "model"])
    evaluation_loss = _vector_to_map(fetch(f"healthcheck_client_evaluation_loss{selector}"), ["clinic", "model"])

    flower_rounds = _vector_to_status_map(
        fetch(f"sum by (model, status) (increase(healthcheck_flower_rounds_total{model_selector}[{window}]))"),
        ["model"],
    )
    flower_duration_sum = _vector_to_map(
        fetch(f"sum by (model) (increase(healthcheck_flower_round_duration_seconds_sum{model_selector}[{window}]))"),
        ["model"],
    )
    flower_duration_count = _vector_to_map(
        fetch(f"sum by (model) (increase(healthcheck_flower_round_duration_seconds_count{model_selector}[{window}]))"),
        ["model"],
    )
    flower_clients_sum = _vector_to_map(
        fetch(f"sum by (model) (increase(healthcheck_flower_clients_per_round_sum{model_selector}[{window}]))"),
        ["model"],
    )
    flower_clients_count = _vector_to_map(
        fetch(f"sum by (model) (increase(healthcheck_flower_clients_per_round_count{model_selector}[{window}]))"),
        ["model"],
    )
    current_round = _vector_to_map(fetch(f"healthcheck_flower_current_round{model_selector}"), ["model"])
    last_checkpoint_round = _vector_to_map(fetch(f"healthcheck_flower_last_checkpoint_round{model_selector}"), ["model"])
    checkpoint_saves = _vector_to_map(
        fetch(f"sum by (model) (increase(healthcheck_flower_checkpoint_saves_total{model_selector}[{window}]))"),
        ["model"],
    )
    checkpoint_failures = _vector_to_map(
        fetch(f"sum by (model, reason) (increase(healthcheck_flower_checkpoint_failures_total{model_selector}[{window}]))"),
        ["model"],
    )
    flower_up = _vector_to_map(fetch('up{job="healthcheck-flower"}'), ["instance", "job"])
    pushgateway_up = _vector_to_map(fetch('up{job="healthcheck-pushgateway"}'), ["instance", "job"])

    clinic_keys = sorted(
        set(fit_rounds.keys())
        | set(evaluate_rounds.keys())
        | set(fit_duration_sum.keys())
        | set(evaluate_duration_sum.keys())
        | set(bytes_sent.keys())
        | set(bytes_received.keys())
        | set(training_loss.keys())
        | set(evaluation_accuracy.keys())
        | set(evaluation_loss.keys())
    )

    flower_models = sorted(
        {key[0] for key in flower_rounds.keys()}
        | {key[0] for key in flower_duration_sum.keys()}
        | {key[0] for key in flower_duration_count.keys()}
        | {key[0] for key in flower_clients_sum.keys()}
        | {key[0] for key in flower_clients_count.keys()}
        | {key[0] for key in current_round.keys()}
        | {key[0] for key in last_checkpoint_round.keys()}
        | {key[0] for key in checkpoint_saves.keys()}
        | {key[0] for key in checkpoint_failures.keys()}
    )

    clinic_rows = [
        _clinic_row(
            clinic=clinic,
            model=model,
            fit_rounds=fit_rounds.get((clinic, model), {}),
            evaluate_rounds=evaluate_rounds.get((clinic, model), {}),
            fit_duration_sum=fit_duration_sum.get((clinic, model)),
            fit_duration_count=fit_duration_count.get((clinic, model)),
            evaluate_duration_sum=evaluate_duration_sum.get((clinic, model)),
            evaluate_duration_count=evaluate_duration_count.get((clinic, model)),
            bytes_sent=bytes_sent.get((clinic, model)),
            bytes_received=bytes_received.get((clinic, model)),
            training_loss=training_loss.get((clinic, model)),
            evaluation_accuracy=evaluation_accuracy.get((clinic, model)),
            evaluation_loss=evaluation_loss.get((clinic, model)),
        )
        for clinic, model in clinic_keys
    ]

    flower_rows = [
        _flower_row(
            model=model,
            rounds=flower_rounds.get((model,), {}),
            duration_sum=flower_duration_sum.get((model,)),
            duration_count=flower_duration_count.get((model,)),
            clients_sum=flower_clients_sum.get((model,)),
            clients_count=flower_clients_count.get((model,)),
            current_round=current_round.get((model,)),
            last_checkpoint_round=last_checkpoint_round.get((model,)),
            checkpoint_saves=checkpoint_saves.get((model,)),
            checkpoint_failures=sum(value for (label_model, _reason), value in checkpoint_failures.items() if label_model == model),
        )
        for model in flower_models
    ]

    summary = {
        "window": window,
        "window_seconds": _parse_window_to_seconds(window),
        "clinic_count": len(clinic_rows),
        "flower_model_count": len(flower_rows),
        "clinics": {
            "fit_rounds_success": float(sum(row["fit_rounds"]["success"] for row in clinic_rows)),
            "fit_rounds_failed": float(sum(row["fit_rounds"]["failures"] for row in clinic_rows)),
            "evaluate_rounds_success": float(sum(row["evaluate_rounds"]["success"] for row in clinic_rows)),
            "evaluate_rounds_failed": float(sum(row["evaluate_rounds"]["failures"] for row in clinic_rows)),
            "avg_fit_duration_seconds": _mean([row["fit_duration_seconds"]["average"] for row in clinic_rows if row["fit_duration_seconds"]["average"] is not None]),
            "avg_evaluate_duration_seconds": _mean([row["evaluate_duration_seconds"]["average"] for row in clinic_rows if row["evaluate_duration_seconds"]["average"] is not None]),
            "bytes_sent": float(sum(row["bytes_sent"] or 0.0 for row in clinic_rows)),
            "bytes_received": float(sum(row["bytes_received"] or 0.0 for row in clinic_rows)),
        },
        "flower": {
            "rounds_success": float(sum(row["rounds"]["success"] for row in flower_rows)),
            "rounds_failed": float(sum(row["rounds"]["failures"] for row in flower_rows)),
            "avg_round_duration_seconds": _mean([row["round_duration_seconds"]["average"] for row in flower_rows if row["round_duration_seconds"]["average"] is not None]),
            "avg_clients_per_round": _mean([row["clients_per_round"]["average"] for row in flower_rows if row["clients_per_round"]["average"] is not None]),
            "flower_up": bool(flower_up),
            "pushgateway_up": bool(pushgateway_up),
        },
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prometheus_url": DEFAULT_PROMETHEUS_URL,
        "window": window,
        "filters": {
            "clinics": list(clinics) if clinics else [],
            "models": list(models) if models else [],
        },
        "summary": summary,
        "clinics": clinic_rows,
        "flower": flower_rows,
    }


class Command(BaseCommand):
    help = "Generate a JSON system efficiency report from Prometheus metrics."

    def add_arguments(self, parser):
        parser.add_argument(
            "--prometheus-url",
            default=DEFAULT_PROMETHEUS_URL,
            help="Prometheus base URL, for example http://localhost:30990.",
        )
        parser.add_argument(
            "--window",
            default="24h",
            help="Prometheus range window, for example 24h or 7d.",
        )
        parser.add_argument(
            "--clinic",
            action="append",
            help="Optional clinic filter. Repeat for multiple clinics.",
        )
        parser.add_argument(
            "--model",
            action="append",
            help="Optional model filter. Repeat for multiple models.",
        )
        parser.add_argument(
            "--output",
            default=None,
            help="Optional path for the JSON report.",
        )

    def handle(self, *args, **options):
        prometheus_url = str(options["prometheus_url"]).rstrip("/")
        window = str(options["window"]).strip()
        if not window:
            raise CommandError("--window cannot be empty")

        timeout_seconds = int(os.getenv("PROMETHEUS_QUERY_TIMEOUT_SECONDS", "30"))

        def fetch(query: str) -> List[PrometheusSample]:
            response = requests.get(
                f"{prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") != "success":
                raise CommandError(f"Prometheus query failed: {payload!r}")
            data = payload.get("data", {})
            if not isinstance(data, dict):
                return []
            result = data.get("result", [])
            return result if isinstance(result, list) else []

        report = build_system_efficiency_report(
            fetch,
            window=window,
            clinics=options.get("clinic"),
            models=options.get("model"),
        )
        report["prometheus_url"] = prometheus_url

        if options["output"]:
            destination = Path(options["output"])
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            destination = Path("reports") / "system_efficiency" / f"system_efficiency_report_{timestamp}.json"

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")

        self.stdout.write(self.style.SUCCESS("System efficiency report generated"))
        self.stdout.write(f"Prometheus URL: {prometheus_url}")
        self.stdout.write(f"Window: {window}")
        self.stdout.write(f"JSON: {destination}")
