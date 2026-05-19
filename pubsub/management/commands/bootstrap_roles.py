from django.core.management.base import BaseCommand
from django.contrib.auth.models import Group, Permission
from django.contrib.contenttypes.models import ContentType


class Command(BaseCommand):
    help = "Create basic role groups and assign model permissions for pubsub app."

    def handle(self, *args, **options):
        # Import models here to avoid app-loading issues at import time
        from pubsub.models import Patient, Appointment

        self.stdout.write("Bootstrapping roles and permissions...")

        # Ensure groups exist
        groups = {}
        for name in ["admin", "doctor", "receptionist"]:
            group, created = Group.objects.get_or_create(name=name)
            groups[name] = group
            if created:
                self.stdout.write(f"Created group: {name}")

        # Content types
        patient_ct = ContentType.objects.get_for_model(Patient)
        appointment_ct = ContentType.objects.get_for_model(Appointment)

        # All permissions for admin
        admin_perms = Permission.objects.filter(content_type__in=[patient_ct, appointment_ct])
        groups["admin"].permissions.set(admin_perms)
        self.stdout.write("Assigned all pubsub perms to 'admin' group")

        # Doctor: view/change patients, view/add/change appointments
        doctor_perms = []
        doctor_perms += list(Permission.objects.filter(content_type=patient_ct, codename__in=["view_patient", "change_patient"]))
        doctor_perms += list(Permission.objects.filter(content_type=appointment_ct, codename__in=["view_appointment", "add_appointment", "change_appointment"]))
        groups["doctor"].permissions.set(doctor_perms)
        self.stdout.write("Assigned patient/appointment perms to 'doctor' group")

        # Receptionist: can view patients and create/view appointments
        recep_perms = []
        recep_perms += list(Permission.objects.filter(content_type=patient_ct, codename__in=["view_patient"]))
        recep_perms += list(Permission.objects.filter(content_type=appointment_ct, codename__in=["view_appointment", "add_appointment"]))
        groups["receptionist"].permissions.set(recep_perms)
        self.stdout.write("Assigned appointment perms to 'receptionist' group")
        # If django-guardian is available, grant object permissions to already-assigned doctors
        try:
            from guardian.shortcuts import assign_perm
            assigned_count = 0
            for p in Patient.objects.filter(assigned_doctor__isnull=False).select_related('assigned_doctor'):
                doctor = p.assigned_doctor
                assign_perm('view_patient', doctor, p)
                assign_perm('change_patient', doctor, p)
                assign_perm('delete_patient', doctor, p)
                assigned_count += 1
            if assigned_count:
                self.stdout.write(f"Assigned object perms to {assigned_count} existing patients' doctors")
        except Exception:
            # guardian not installed or failed; skip object perms
            self.stdout.write("django-guardian not installed; skipping object-level grants for existing patients.")

        self.stdout.write(self.style.SUCCESS("Roles and permissions bootstrapped successfully."))
