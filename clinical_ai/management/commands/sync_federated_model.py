import os

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Download the latest federated model from MinIO and replace the local synced snapshot."

    def add_arguments(self, parser):
        parser.add_argument(
            "--model",
            choices=["alex5050", "mustafa"],
            default="alex5050",
            help="Model family to sync.",
        )
        parser.add_argument(
            "--bucket-name",
            default=os.getenv("MINIO_BUCKET_NAME", "models"),
            help="MinIO bucket name containing federated checkpoints.",
        )
        parser.add_argument(
            "--endpoint",
            default=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            help="MinIO endpoint.",
        )
        parser.add_argument(
            "--access-key",
            default=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            help="MinIO access key.",
        )
        parser.add_argument(
            "--secret-key",
            default=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            help="MinIO secret key.",
        )
        parser.add_argument(
            "--use-ssl",
            action="store_true",
            default=os.getenv("MINIO_USE_SSL", "false").lower() == "true",
            help="Use SSL when connecting to MinIO.",
        )

    def handle(self, *args, **options):
        try:
            from clinical_ai.federated_model_sync import download_latest_model

            synced_path = download_latest_model(
                model_type=options["model"],
                bucket_name=options["bucket_name"],
                endpoint=options["endpoint"],
                access_key=options["access_key"],
                secret_key=options["secret_key"],
                use_ssl=options["use_ssl"],
            )
        except Exception as exc:
            raise CommandError(f"Failed to sync federated model: {exc}") from exc

        self.stdout.write(self.style.SUCCESS(f"Latest federated model saved to {synced_path}"))