from django.db import models
from django.core.validators import RegexValidator
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import date
from django.conf import settings
import hmac
import hashlib
import logging


class AccessLog(models.Model):
    """Audit log for tracking access to patient data"""
    
    ACTION_CHOICES = [
        ('view', 'Viewed'),
        ('edit', 'Edited'),
        ('delete', 'Deleted'),
        ('create', 'Created'),
        ('search', 'Searched'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='access_logs')
    patient = models.ForeignKey('Patient', on_delete=models.SET_NULL, null=True, related_name='access_logs')
    action = models.CharField(max_length=20, choices=ACTION_CHOICES)
    timestamp = models.DateTimeField(default=timezone.now)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    details = models.TextField(blank=True, help_text="Additional details about the action")
    
    class Meta:
        db_table = 'access_logs'
        ordering = ['-timestamp']
        verbose_name = 'Access Log'
        verbose_name_plural = 'Access Logs'
        indexes = [
            models.Index(fields=['user', 'timestamp']),
            models.Index(fields=['patient', 'timestamp']),
        ]
    
    def __str__(self):
        patient_name = self.patient.get_full_name() if self.patient else "N/A"
        user_name = self.user.username if self.user else "N/A"
        return f"{user_name} {self.get_action_display()} {patient_name} at {self.timestamp}"


class Patient(models.Model):
    """Model for patient data in medical clinics"""
    
    # Validators for personal identification fields
    cnp_validator = RegexValidator(
        regex=r'^\d{13}$',
        message='CNP-ul trebuie să aibă exact 13 cifre'
    )
    
    serie_ci_validator = RegexValidator(
        regex=r'^[A-Z]{2}$',
        message='Serie CI trebuie să aibă exact 2 litere majuscule'
    )
    
    numar_ci_validator = RegexValidator(
        regex=r'^\d{6}$',
        message='Număr CI trebuie să aibă exact 6 cifre'
    )
    
    phone_validator = RegexValidator(
        regex=r'^(\+?[1-9]\d{1,14}|0\d{9,14})$',
        message='Introduceți un număr de telefon valid în format internațional (ex: +40712345678)'
    )
    
    email_validator = RegexValidator(
        regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        message='Introduceți o adresă de email validă'
    )
    
    # Primary key 
    id = models.AutoField(primary_key=True)

    # Ownership / assignment
    assigned_doctor = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='assigned_patients',
        limit_choices_to={'groups__name': 'doctor'},
        help_text='Doctor responsible for this patient',
    )
    
    # Personal identification
    CNP = models.CharField(
        max_length=13, 
        unique=True, 
        validators=[cnp_validator],
        help_text="Cod Numeric Personal (13 digits)"
    )
    nume = models.CharField(max_length=100, help_text="Last name")
    prenume = models.CharField(max_length=100, help_text="First name")
    data_nasterii = models.DateField(help_text="Date of birth")
    
    # Identity document
    serie_ci = models.CharField(
        max_length=2, 
        validators=[serie_ci_validator],
        help_text="ID card series (2 uppercase letters)"
    )
    numar_ci = models.CharField(
        max_length=6, 
        validators=[numar_ci_validator],
        help_text="ID card number (6 digits)"
    )
    nationalitate = models.CharField(max_length=50, default="Română", help_text="Nationality")
    
    # Contact information
    telefon = models.CharField(
        max_length=20, 
        blank=True, 
        null=True, 
        validators=[phone_validator],
        help_text="Phone number in international format (e.g., +40712345678)"
    )
    email = models.EmailField(
        blank=True, 
        null=True, 
        validators=[email_validator],
        help_text="Email address"
    )
    
    # Address
    oras = models.CharField(max_length=100, help_text="City")
    judet = models.CharField(max_length=100, help_text="County")
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    # Pseudonym for safe lookup (HMAC of CNP)
    pseudonym = models.CharField(
        max_length=64,
        unique=True,
        null=True,
        blank=True,
        db_index=True,
        help_text='HMAC-based pseudonym for patient identifier (deterministic)'
    )
    
    class Meta:
        db_table = 'patients'
        ordering = ['-created_at']
        verbose_name = 'Patient'
        verbose_name_plural = 'Patients'
        indexes = [
            models.Index(fields=['CNP']),
            models.Index(fields=['nume', 'prenume']),
            models.Index(fields=['assigned_doctor', 'created_at']),
                models.Index(fields=['pseudonym']),
        ]
    
    def __str__(self):
        return f"{self.nume} {self.prenume} (CNP: {self.CNP})"
    
    def get_full_name(self):
        return f"{self.prenume} {self.nume}"
    
    def get_initials(self):
        """Return patient initials for privacy in lists"""
        return f"{self.prenume[0]}. {self.nume[0]}."
    
    def get_masked_cnp(self):
        """Mask CNP showing only last 4 digits"""
        if len(self.CNP) >= 4:
            return '*' * 9 + self.CNP[-4:]
        return self.CNP
    
    def get_age(self):
        """Calculate age from date of birth"""
        today = date.today()
        age = today.year - self.data_nasterii.year
        if today.month < self.data_nasterii.month or (today.month == self.data_nasterii.month and today.day < self.data_nasterii.day):
            age -= 1
        return age
    
    def get_masked_phone(self):
        """Mask phone number showing only first and last 3 digits"""
        if self.telefon and len(self.telefon) >= 6:
            return self.telefon[:3] + '*' * (len(self.telefon) - 6) + self.telefon[-3:]
        return self.telefon or "—"
    
    def get_masked_email(self):
        """Mask email showing only first char and domain"""
        if self.email:
            parts = self.email.split('@')
            if len(parts) == 2:
                return f"{parts[0][0]}***@{parts[1]}"
        return "—"
    
    def get_masked_identity(self):
        """Return masked identity document info"""
        return f"{self.serie_ci} ******"

    def _compute_pseudonym(self, identifier: str) -> str:
        """Compute HMAC-SHA256 hex digest for a given identifier using a secret key.

        Falls back to `settings.SECRET_KEY` if `settings.PATIENT_HMAC_KEY` is not set.
        """
        key = getattr(settings, 'PATIENT_HMAC_KEY', None) or settings.SECRET_KEY
        if not isinstance(key, (bytes, bytearray)):
            key = str(key).encode('utf-8')
        return hmac.new(key, identifier.encode('utf-8'), hashlib.sha256).hexdigest()

    def save(self, *args, **kwargs):
        # Ensure pseudonym exists and is deterministic from CNP
        try:
            if not self.pseudonym and self.CNP:
                self.pseudonym = self._compute_pseudonym(self.CNP)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to compute pseudonym: {e}")

        # proceed with normal validation and save
        return super().save(*args, **kwargs)


class Appointment(models.Model):
    """Model for patient appointments with doctors"""
    
    STATUS_CHOICES = [
        ('scheduled', 'Scheduled'),
        ('confirmed', 'Confirmed'),
        ('completed', 'Completed'),
        ('expired', 'Expired'),
        ('cancelled', 'Cancelled'),
        ('no_show', 'No Show'),
    ]
    
    # Appointment details
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='appointments')
    doctor = models.ForeignKey(User, on_delete=models.CASCADE, related_name='doctor_appointments', 
                               limit_choices_to={'groups__name': 'doctor'})
    
    # Schedule
    appointment_date = models.DateField(help_text="Appointment date")
    appointment_time = models.TimeField(help_text="Appointment time")
    duration_minutes = models.IntegerField(default=30, help_text="Duration in minutes")
    
    # Status and details
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='scheduled')
    reason = models.TextField(help_text="Reason for appointment")
    notes = models.TextField(blank=True, help_text="Additional notes")
    
    # Tracking
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='created_appointments')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Redis notification tracking
    notification_sent = models.BooleanField(default=False)
    notification_read = models.BooleanField(default=False)
    
    class Meta:
        db_table = 'appointments'
        ordering = ['appointment_date', 'appointment_time']
        verbose_name = 'Appointment'
        verbose_name_plural = 'Appointments'
        indexes = [
            models.Index(fields=['appointment_date', 'appointment_time']),
            models.Index(fields=['patient', 'appointment_date']),
            models.Index(fields=['doctor', 'appointment_date']),
            models.Index(fields=['status']),
            models.Index(fields=['status', 'completed_at']),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['doctor', 'appointment_date', 'appointment_time'],
                name='unique_doctor_appointment_slot'
            )
        ]
    
    def __str__(self):
        return f"{self.patient.get_full_name()} with Dr. {self.doctor.get_full_name() or self.doctor.username} on {self.appointment_date} at {self.appointment_time}"
    
    def get_end_time(self):
        """Calculate appointment end time"""
        from datetime import datetime, timedelta
        start = datetime.combine(self.appointment_date, self.appointment_time)
        end = start + timedelta(minutes=self.duration_minutes)
        return end.time()
    
    def is_past(self):
        """Check if appointment is in the past"""
        from datetime import datetime
        appointment_datetime = datetime.combine(self.appointment_date, self.appointment_time)
        return appointment_datetime < datetime.now()
    
    def can_cancel(self):
        """Check if appointment can be cancelled"""
        return self.status in ['scheduled', 'confirmed'] and not self.is_past()

    def clean(self):
        """Extra validation for appointments (time grid + overlap for same doctor)."""
        from datetime import datetime, timedelta
        from django.core.exceptions import ValidationError

        super().clean()

        if not self.doctor_id or not self.appointment_date or not self.appointment_time:
            return

        # Enforce fixed 15-minute grid
        allowed_minutes = {0, 15, 30, 45}
        if self.appointment_time.minute not in allowed_minutes:
            raise ValidationError({
                'appointment_time': (
                    "Appointments can only be scheduled at fixed times: 00, 15, 30, or 45 minutes."
                )
            })

        # Compute this appointment's interval
        start_dt = datetime.combine(self.appointment_date, self.appointment_time)
        duration = self.duration_minutes or 30
        end_dt = start_dt + timedelta(minutes=duration)

        # Find other appointments for the same doctor/date that are not cancelled/no_show
        overlapping_qs = Appointment.objects.filter(
            doctor=self.doctor,
            appointment_date=self.appointment_date,
        ).exclude(
            id=self.id
        ).exclude(
            status__in=['cancelled', 'no_show', 'completed', 'expired']
        )

        for other in overlapping_qs:
            other_start = datetime.combine(other.appointment_date, other.appointment_time)
            other_duration = other.duration_minutes or 30
            other_end = other_start + timedelta(minutes=other_duration)

            # Intervals overlap if start < other_end and end > other_start
            if start_dt < other_end and end_dt > other_start:
                raise ValidationError({
                    'appointment_time': (
                        "The appointment overlaps with another appointment for the same doctor "
                        f"({other_start.strftime('%H:%M')} - {other_end.strftime('%H:%M')})."
                    )
                })

    def save(self, *args, **kwargs):
        # Ensure validation (including overlap check) always runs before saving
        self.full_clean()
        return super().save(*args, **kwargs)
