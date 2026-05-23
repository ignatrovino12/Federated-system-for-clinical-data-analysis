"""Security helpers for pseudonymization and field encryption."""
from django.conf import settings
import hmac
import hashlib


def make_pseudonym(identifier: str) -> str:
    key = getattr(settings, 'PATIENT_HMAC_KEY', None) or settings.SECRET_KEY
    if not isinstance(key, (bytes, bytearray)):
        key = str(key).encode('utf-8')
    return hmac.new(key, str(identifier).encode('utf-8'), hashlib.sha256).hexdigest()
