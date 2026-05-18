from __future__ import annotations

import logging

from celery import shared_task

from .appointment_lifecycle import expire_completed_appointments

logger = logging.getLogger(__name__)


@shared_task(name="pubsub.expire_completed_appointments")
def expire_completed_appointments_task() -> int:
    expired = expire_completed_appointments()
    if expired:
        logger.info("Auto-expired completed appointments: %s", expired)
    return expired
