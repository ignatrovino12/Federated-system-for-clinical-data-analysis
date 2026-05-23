from dataclasses import dataclass

from django.contrib.auth.models import Group, Permission
from django.contrib.contenttypes.models import ContentType
from django.db.utils import OperationalError, ProgrammingError


ROLE_DISPLAY_NAMES = {
	"admin": "Administrator",
	"doctor": "Doctor",
	"receptionist": "Receptionist",
	"unassigned": "Unassigned",
}


_ROLES_BOOTSTRAPPED = False


@dataclass(frozen=True)
class RoleInfo:
	role: str

	def get_role_display(self):
		return ROLE_DISPLAY_NAMES.get(self.role, self.role.title())

	def can_view_full_data(self):
		return self.role in {"admin", "doctor"}

	def can_edit_data(self):
		return self.role in {"admin", "doctor"}

	def can_delete_data(self):
		return self.role == "admin"


def is_in_group(user, group_name: str) -> bool:
	maybe_bootstrap_default_roles()
	if not getattr(user, "is_authenticated", False):
		return False
	if group_name == "admin" and getattr(user, "is_superuser", False):
		return True
	try:
		return user.groups.filter(name=group_name).exists()
	except Exception:
		return False


def get_primary_role(user):
	maybe_bootstrap_default_roles()
	if not getattr(user, "is_authenticated", False):
		return None
	if is_in_group(user, "admin"):
		return "admin"
	if is_in_group(user, "doctor"):
		return "doctor"
	if is_in_group(user, "receptionist"):
		return "receptionist"
	return None


def get_role_info(user) -> RoleInfo:
	maybe_bootstrap_default_roles()
	return RoleInfo(role=get_primary_role(user) or "unassigned")


def has_any_role(user, allowed_roles) -> bool:
	maybe_bootstrap_default_roles()
	return any(is_in_group(user, role) for role in allowed_roles)


def maybe_bootstrap_default_roles():
	global _ROLES_BOOTSTRAPPED
	if _ROLES_BOOTSTRAPPED:
		return
	try:
		ensure_default_roles()
		_ROLES_BOOTSTRAPPED = True
	except (OperationalError, ProgrammingError):
		return
	except Exception:
		return


def ensure_default_roles():
	"""Create default role groups and attach baseline permissions."""
	from clinical_ai.models import PatientClinicalRecord
	from pubsub.models import Appointment, Patient

	groups = {name: Group.objects.get_or_create(name=name)[0] for name in ("admin", "doctor", "receptionist")}

	patient_ct = ContentType.objects.get_for_model(Patient)
	appointment_ct = ContentType.objects.get_for_model(Appointment)
	record_ct = ContentType.objects.get_for_model(PatientClinicalRecord)

	groups["admin"].permissions.set(Permission.objects.filter(content_type__app_label__in=("pubsub", "clinical_ai")))

	doctor_perms = list(Permission.objects.filter(content_type=patient_ct, codename__in=("view_patient", "change_patient")))
	doctor_perms += list(Permission.objects.filter(content_type=appointment_ct, codename__in=("view_appointment", "add_appointment", "change_appointment")))
	doctor_perms += list(Permission.objects.filter(content_type=record_ct, codename__in=("view_patientclinicalrecord", "add_patientclinicalrecord", "change_patientclinicalrecord")))
	groups["doctor"].permissions.set(doctor_perms)

	receptionist_perms = list(Permission.objects.filter(content_type=patient_ct, codename__in=("view_patient", "add_patient")))
	receptionist_perms += list(Permission.objects.filter(content_type=appointment_ct, codename__in=("view_appointment", "add_appointment")))
	groups["receptionist"].permissions.set(receptionist_perms)

	global _ROLES_BOOTSTRAPPED
	_ROLES_BOOTSTRAPPED = True
