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

CLIENT_FIT_ROUNDS_TOTAL = Counter(
    "healthcheck_client_fit_rounds_total",
    "Federated training fit calls completed by clinic, model, and status.",
    ["clinic", "model", "status"],
)

CLIENT_EVALUATE_ROUNDS_TOTAL = Counter(
    "healthcheck_client_evaluate_rounds_total",
    "Federated evaluation calls completed by clinic, model, and status.",
    ["clinic", "model", "status"],
)

CLIENT_FIT_DURATION_SECONDS = Histogram(
    "healthcheck_client_fit_duration_seconds",
    "Local federated training duration by clinic and model.",
    ["clinic", "model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
)

CLIENT_EVALUATE_DURATION_SECONDS = Histogram(
    "healthcheck_client_evaluate_duration_seconds",
    "Local federated evaluation duration by clinic and model.",
    ["clinic", "model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
)

CLIENT_TRAINING_EXAMPLES = Histogram(
    "healthcheck_client_training_examples",
    "Number of local training samples used per clinic and model.",
    ["clinic", "model"],
    buckets=(0, 1, 5, 10, 25, 50, 100, 250, 500, 1000),
)

CLIENT_EVALUATION_EXAMPLES = Histogram(
    "healthcheck_client_evaluation_examples",
    "Number of local evaluation samples used per clinic and model.",
    ["clinic", "model"],
    buckets=(0, 1, 5, 10, 25, 50, 100, 250, 500, 1000),
)

CLIENT_PARAMETERS_SENT_BYTES = Counter(
    "healthcheck_client_parameters_sent_bytes_total",
    "Estimated number of parameter bytes sent from each clinic.",
    ["clinic", "model"],
)

CLIENT_PARAMETERS_RECEIVED_BYTES = Counter(
    "healthcheck_client_parameters_received_bytes_total",
    "Estimated number of parameter bytes received by each clinic.",
    ["clinic", "model"],
)

CLIENT_TRAINING_LOSS = Gauge(
    "healthcheck_client_training_loss",
    "Last local training loss reported by each clinic.",
    ["clinic", "model"],
)

CLIENT_EVALUATION_LOSS = Gauge(
    "healthcheck_client_evaluation_loss",
    "Last local evaluation loss reported by each clinic.",
    ["clinic", "model"],
)

CLIENT_EVALUATION_ACCURACY = Gauge(
    "healthcheck_client_evaluation_accuracy",
    "Last local evaluation accuracy reported by each clinic.",
    ["clinic", "model"],
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


def record_client_fit(
    clinic: str,
    model: str,
    duration_seconds: float,
    num_examples: int,
    loss: float,
    parameter_bytes_sent: int,
    status: str = "success",
) -> None:
    CLIENT_FIT_DURATION_SECONDS.labels(clinic=clinic, model=model).observe(duration_seconds)
    CLIENT_TRAINING_EXAMPLES.labels(clinic=clinic, model=model).observe(num_examples)
    CLIENT_TRAINING_LOSS.labels(clinic=clinic, model=model).set(loss)
    CLIENT_PARAMETERS_SENT_BYTES.labels(clinic=clinic, model=model).inc(parameter_bytes_sent)
    CLIENT_FIT_ROUNDS_TOTAL.labels(clinic=clinic, model=model, status=status).inc()


def record_client_evaluate(
    clinic: str,
    model: str,
    duration_seconds: float,
    num_examples: int,
    loss: float,
    accuracy: float,
    parameter_bytes_received: int,
    status: str = "success",
) -> None:
    CLIENT_EVALUATE_DURATION_SECONDS.labels(clinic=clinic, model=model).observe(duration_seconds)
    CLIENT_EVALUATION_EXAMPLES.labels(clinic=clinic, model=model).observe(num_examples)
    CLIENT_EVALUATION_LOSS.labels(clinic=clinic, model=model).set(loss)
    CLIENT_EVALUATION_ACCURACY.labels(clinic=clinic, model=model).set(accuracy)
    CLIENT_PARAMETERS_RECEIVED_BYTES.labels(clinic=clinic, model=model).inc(parameter_bytes_received)
    CLIENT_EVALUATE_ROUNDS_TOTAL.labels(clinic=clinic, model=model, status=status).inc()