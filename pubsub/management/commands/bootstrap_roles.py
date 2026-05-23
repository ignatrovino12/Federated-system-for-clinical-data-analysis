from django.core.management.base import BaseCommand
from pubsub.roles import ensure_default_roles


class Command(BaseCommand):
    help = "Create basic role groups and assign model permissions for pubsub app."

    def handle(self, *args, **options):
        ensure_default_roles()
        self.stdout.write(self.style.SUCCESS("Roles and permissions bootstrapped successfully."))
