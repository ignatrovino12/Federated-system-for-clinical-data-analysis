from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Clear the model artifact cache so fresh synced models are loaded on next prediction."

    def add_arguments(self, parser):
        parser.add_argument(
            "--model",
            choices=["alex5050", "mustafa", "all"],
            default="all",
            help="Which model cache to clear.",
        )

    def handle(self, *args, **options):
        from clinical_ai.services import _load_alex_artifacts_cached, _load_mustafa_artifacts_cached

        model = options["model"]

        if model in {"alex5050", "all"}:
            _load_alex_artifacts_cached.cache_clear()
            self.stdout.write(self.style.SUCCESS("Cleared Alex5050 model cache"))

        if model in {"mustafa", "all"}:
            _load_mustafa_artifacts_cached.cache_clear()
            self.stdout.write(self.style.SUCCESS("Cleared Mustafa model cache"))

        self.stdout.write(
            self.style.SUCCESS("✓ Model cache cleared. Next predictions will use fresh models.")
        )
