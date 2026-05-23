from django.db.models.signals import post_migrate
from django.dispatch import receiver

from .roles import ensure_default_roles


@receiver(post_migrate)
def bootstrap_default_roles(sender, **kwargs):
	if getattr(sender, "label", None) != "pubsub":
		return
	try:
		ensure_default_roles()
	except Exception:
		pass
