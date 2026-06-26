from django.apps import AppConfig


class HealthcheckConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "healthcheck"

    def ready(self):
        from .metrics import start_pushgateway_exporter

        start_pushgateway_exporter()