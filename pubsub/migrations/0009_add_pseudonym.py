from django.db import migrations, models
import hashlib
import hmac
from django.conf import settings


def compute_pseudonym(cnp):
    key = getattr(settings, 'PATIENT_HMAC_KEY', None) or settings.SECRET_KEY
    if not isinstance(key, (bytes, bytearray)):
        key = str(key).encode('utf-8')
    return hmac.new(key, str(cnp).encode('utf-8'), hashlib.sha256).hexdigest()


def forwards(apps, schema_editor):
    Patient = apps.get_model('pubsub', 'Patient')
    for p in Patient.objects.all():
        try:
            if not p.pseudonym and p.CNP:
                p.pseudonym = compute_pseudonym(p.CNP)
                p.save(update_fields=['pseudonym'])
        except Exception:
            # skip any problematic records
            continue


def backwards(apps, schema_editor):
    Patient = apps.get_model('pubsub', 'Patient')
    for p in Patient.objects.all():
        p.pseudonym = None
        p.save(update_fields=['pseudonym'])


class Migration(migrations.Migration):

    dependencies = [
        ('pubsub', '0008_appointment_completed_at_alter_appointment_status_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='patient',
            name='pseudonym',
            field=models.CharField(max_length=64, unique=True, null=True, blank=True, db_index=True, help_text='HMAC-based pseudonym for patient identifier (deterministic)'),
        ),
        migrations.RunPython(forwards, backwards),
    ]
