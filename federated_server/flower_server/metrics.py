from prometheus_client import Counter, Gauge, Histogram


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