from prometheus_client import Counter, Gauge, Histogram


HTTP_REQUESTS_TOTAL = Counter(
    "healthcheck_http_requests_total",
    "Total Django HTTP requests processed by view and status code.",
    ["app", "view", "method", "status"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "healthcheck_http_request_duration_seconds",
    "Django request latency by view and method.",
    ["app", "view", "method"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

HTTP_REQUEST_EXCEPTIONS_TOTAL = Counter(
    "healthcheck_http_request_exceptions_total",
    "Unhandled Django request exceptions by view and exception type.",
    ["app", "view", "exception"],
)

CLINICAL_ANALYSIS_REQUESTS_TOTAL = Counter(
    "healthcheck_clinical_analysis_requests_total",
    "Clinical AI analysis requests by model, mode, and outcome.",
    ["model", "mode", "status"],
)

FLOWER_ROUNDS_TOTAL = Counter(
    "healthcheck_flower_rounds_total",
    "Completed Flower aggregation rounds by model and outcome.",
    ["model", "status"],
)

FLOWER_ROUND_DURATION_SECONDS = Histogram(
    "healthcheck_flower_round_duration_seconds",
    "Flower aggregation duration by model.",
    ["model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
)

FLOWER_CLIENTS_PER_ROUND = Histogram(
    "healthcheck_flower_clients_per_round",
    "Number of clients contributing to each Flower aggregation round.",
    ["model"],
    buckets=(0, 1, 2, 3, 5, 10, 20, 50),
)

FLOWER_CURRENT_ROUND = Gauge(
    "healthcheck_flower_current_round",
    "Last Flower round number processed for a model.",
    ["model"],
)

FLOWER_LAST_CHECKPOINT_ROUND = Gauge(
    "healthcheck_flower_last_checkpoint_round",
    "Last saved checkpoint round for a model.",
    ["model"],
)

FLOWER_CHECKPOINT_SAVES_TOTAL = Counter(
    "healthcheck_flower_checkpoint_saves_total",
    "Successful Flower checkpoint saves by model.",
    ["model"],
)

FLOWER_CHECKPOINT_FAILURES_TOTAL = Counter(
    "healthcheck_flower_checkpoint_failures_total",
    "Failed Flower checkpoint saves by model and reason.",
    ["model", "reason"],
)


def record_http_request(view_name: str, method: str, status_code: int, duration_seconds: float) -> None:
    labels = {
        "app": "django",
        "view": view_name or "unknown",
        "method": method,
        "status": str(status_code),
    }
    HTTP_REQUESTS_TOTAL.labels(**labels).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(
        app=labels["app"],
        view=labels["view"],
        method=labels["method"],
    ).observe(duration_seconds)


def record_http_exception(view_name: str, method: str, exception_name: str) -> None:
    HTTP_REQUEST_EXCEPTIONS_TOTAL.labels(
        app="django",
        view=view_name or "unknown",
        exception=exception_name or "Exception",
    ).inc()


def record_analysis_request(model: str, mode: str, status: str) -> None:
    CLINICAL_ANALYSIS_REQUESTS_TOTAL.labels(
        model=model,
        mode=mode,
        status=status,
    ).inc()


def record_flower_round(model: str, round_number: int, clients: int, duration_seconds: float, status: str) -> None:
    FLOWER_CURRENT_ROUND.labels(model=model).set(round_number)
    FLOWER_CLIENTS_PER_ROUND.labels(model=model).observe(clients)
    FLOWER_ROUND_DURATION_SECONDS.labels(model=model).observe(duration_seconds)
    FLOWER_ROUNDS_TOTAL.labels(model=model, status=status).inc()


def record_flower_checkpoint_saved(model: str, round_number: int) -> None:
    FLOWER_CHECKPOINT_SAVES_TOTAL.labels(model=model).inc()
    FLOWER_LAST_CHECKPOINT_ROUND.labels(model=model).set(round_number)


def record_flower_checkpoint_failure(model: str, reason: str) -> None:
    FLOWER_CHECKPOINT_FAILURES_TOTAL.labels(model=model, reason=reason).inc()