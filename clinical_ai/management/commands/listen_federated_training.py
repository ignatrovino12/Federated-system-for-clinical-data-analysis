import logging
import os
import threading
import time

import requests
from django.core.management.base import BaseCommand, CommandError
from prometheus_client import REGISTRY, push_to_gateway, start_http_server


logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Poll the Flower control API and start local federated clients when a new training request appears."

    def add_arguments(self, parser):
        parser.add_argument(
            "--control-url",
            default="http://flower-server:8081",
            help="Base URL of the Flower control API.",
        )
        parser.add_argument(
            "--poll-interval",
            type=int,
            default=30,
            help="Seconds between polling attempts.",
        )
        parser.add_argument(
            "--once",
            action="store_true",
            help="Process a single request and exit.",
        )

    def handle(self, *args, **options):
        from django.core.management import call_command

        control_url = options["control_url"].rstrip("/")
        poll_interval = max(5, int(options["poll_interval"]))
        once = options["once"]
        last_request_id = None

        metrics_port = int(os.getenv("FLOWER_CLIENT_PROMETHEUS_PORT", "8110"))
        try:
            start_http_server(metrics_port, addr="0.0.0.0")
            self.stdout.write(self.style.SUCCESS(f"Prometheus metrics available on 0.0.0.0:{metrics_port}/metrics"))
        except OSError as exc:
            logger.warning("Client metrics endpoint already bound on port %s: %s", metrics_port, exc)

        pushgateway_url = os.getenv("PROMETHEUS_PUSHGATEWAY_URL", "").strip()
        if pushgateway_url:
            push_interval = max(5, int(os.getenv("PROMETHEUS_PUSH_INTERVAL_SECONDS", "15")))
            push_job = os.getenv("PROMETHEUS_PUSH_JOB", "healthcheck-clinic")
            clinic_key = os.getenv("CLINIC_ID", os.getenv("HOSTNAME", "clinic"))

            def _push_metrics_loop() -> None:
                while True:
                    try:
                        push_to_gateway(
                            pushgateway_url,
                            job=push_job,
                            registry=REGISTRY,
                            grouping_key={"clinic": clinic_key},
                        )
                    except Exception as exc:
                        logger.warning("Could not push metrics to Pushgateway %s: %s", pushgateway_url, exc)
                    time.sleep(push_interval)

            threading.Thread(target=_push_metrics_loop, daemon=True).start()
            self.stdout.write(
                self.style.SUCCESS(
                    f"Pushing metrics to Pushgateway at {pushgateway_url} every {push_interval}s (clinic={clinic_key})."
                )
            )

        self.stdout.write(self.style.SUCCESS(f"Listening for training requests at {control_url}/training/latest"))

        while True:
            try:
                response = requests.get(f"{control_url}/training/latest", timeout=10)
                if response.status_code == 404:
                    time.sleep(poll_interval)
                    continue

                response.raise_for_status()
                payload = response.json()

                request_id = payload.get("request_id")
                if not request_id or request_id == last_request_id:
                    time.sleep(poll_interval)
                    continue

                models = payload.get("models") or ["alex5050"]
                server_address = payload.get("server_address")
                min_samples = int(payload.get("min_samples", 1))
                test_split = float(payload.get("test_split", 0.2))
                batch_size = int(payload.get("batch_size", 32))
                sync_default = os.getenv("FLOWER_SYNC_LATEST_MODEL", "false").lower() == "true"
                sync_latest_model = bool(payload.get("sync_latest_model", sync_default))
                model_gap_seconds = max(0, int(payload.get("model_gap_seconds", 0)))

                self.stdout.write(
                    self.style.SUCCESS(
                        f"New federated request {request_id} detected for models: {', '.join(models)}"
                    )
                )

                for index, model in enumerate(models):
                    call_kwargs = {
                        "model": model,
                        "min_samples": min_samples,
                        "test_split": test_split,
                        "batch_size": batch_size,
                    }
                    if server_address:
                        call_kwargs["server_address"] = server_address
                    if sync_latest_model:
                        call_kwargs["sync_latest_model"] = True

                    self.stdout.write(f"Starting federated client for {model}...")
                    try:
                        call_command("run_flower_client", **call_kwargs)
                    except Exception as exc:
                        logger.exception("Federated client run failed for model %s: %s", model, exc)
                        self.stderr.write(self.style.ERROR(f"Federated client failed for {model}: {exc}"))

                    is_last_model = index == len(models) - 1
                    if not is_last_model and model_gap_seconds > 0:
                        self.stdout.write(
                            f"Waiting {model_gap_seconds}s before starting next model..."
                        )
                        time.sleep(model_gap_seconds)

                last_request_id = request_id
                if once:
                    return

            except requests.RequestException as exc:
                logger.warning("Could not reach Flower control API: %s", exc)
            except Exception as exc:
                raise CommandError(f"Federated training listener failed: {exc}") from exc

            time.sleep(poll_interval)