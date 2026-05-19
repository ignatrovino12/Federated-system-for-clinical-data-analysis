from django.test import TestCase, Client
from django.contrib.auth.models import User, Group, Permission
from django.urls import reverse
from pubsub.models import Patient, Appointment
from django.utils import timezone


class PermissionMatrixTests(TestCase):
    def setUp(self):
        # Create groups
        admin_group, _ = Group.objects.get_or_create(name='admin')
        doctor_group, _ = Group.objects.get_or_create(name='doctor')
        reception_group, _ = Group.objects.get_or_create(name='receptionist')

        # Create users
        self.admin = User.objects.create_user('admin', password='pass')
        self.doctor = User.objects.create_user('doctor', password='pass')
        self.receptionist = User.objects.create_user('reception', password='pass')

        # Assign groups
        self.admin.groups.add(admin_group)
        self.doctor.groups.add(doctor_group)
        self.receptionist.groups.add(reception_group)

        # Give model perms (simplified)
        perms = Permission.objects.filter(codename__in=['add_patient', 'change_patient', 'delete_patient', 'add_appointment', 'view_appointment', 'change_appointment', 'delete_appointment'])
        for p in perms:
            admin_group.permissions.add(p)
        # Doctors get view/change appointment and patient change
        for codename in ['view_appointment', 'change_appointment', 'change_patient']:
            perm = Permission.objects.get(codename=codename)
            doctor_group.permissions.add(perm)
        # Receptionists can add appointments and view patients
        for codename in ['add_appointment', 'view_appointment']:
            perm = Permission.objects.get(codename=codename)
            reception_group.permissions.add(perm)

        # Create a patient assigned to the doctor
        self.patient = Patient.objects.create(nume='Popescu', prenume='Ion', CNP='1234567890123', assigned_doctor=self.doctor)

        # Create an appointment for the doctor
        self.appointment = Appointment.objects.create(
            patient=self.patient,
            doctor=self.doctor,
            appointment_date=timezone.now().date(),
            appointment_time=timezone.now().time().replace(microsecond=0),
            duration_minutes=30,
            reason='Test',
            created_by=self.receptionist,
        )

        self.client = Client()

    def test_doctor_can_view_assigned_patient(self):
        self.client.force_login(self.doctor)
        url = reverse('pubsub:patient_detail', kwargs={'patient_id': self.patient.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)

    def test_doctor_cannot_view_unassigned_patient(self):
        other_patient = Patient.objects.create(nume='Ionescu', prenume='Ana', CNP='9876543210987')
        self.client.force_login(self.doctor)
        url = reverse('pubsub:patient_detail', kwargs={'patient_id': other_patient.id})
        response = self.client.get(url)
        # Should redirect due to permission denied
        self.assertNotEqual(response.status_code, 200)

    def test_receptionist_can_create_appointment(self):
        self.client.force_login(self.receptionist)
        url = reverse('pubsub:appointment_create')
        resp = self.client.post(url, data={
            'patient_id': self.patient.id,
            'doctor_id': self.doctor.id,
            'appointment_date': timezone.now().date().isoformat(),
            'appointment_time': timezone.now().time().replace(microsecond=0).isoformat(),
            'duration_minutes': 30,
            'reason': 'Follow-up'
        }, follow=True)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(Appointment.objects.filter(reason='Follow-up').exists())

    def test_admin_can_delete_patient(self):
        self.client.force_login(self.admin)
        url = reverse('pubsub:patient_delete', kwargs={'patient_id': self.patient.id})
        resp = self.client.post(url, follow=True)
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(Patient.objects.filter(id=self.patient.id).exists())
