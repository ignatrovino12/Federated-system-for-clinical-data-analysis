from __future__ import annotations

import os
from datetime import timedelta

from django.utils import timezone

from .models import Appointment


def get_appointment_expiry_delta() -> timedelta:
    hours = int(os.getenv("APPOINTMENT_COMPLETED_EXPIRY_HOURS", "24"))
    return timedelta(hours=max(1, hours))


def expire_completed_appointments() -> int:
    expiry_before = timezone.now() - get_appointment_expiry_delta()
    return Appointment.objects.filter(
        status="completed",
        completed_at__isnull=False,
        completed_at__lte=expiry_before,
    ).update(status="expired")
