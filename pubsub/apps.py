from django.apps import AppConfig


class PubsubConfig(AppConfig):
    name = "pubsub"

    def ready(self):
        from . import signals  # noqa: F401
