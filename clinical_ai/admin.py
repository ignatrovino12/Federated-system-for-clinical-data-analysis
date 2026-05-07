import os
import shlex
import shutil
import threading
import subprocess
import logging

from django.contrib import admin, messages
from django.http import HttpRequest, HttpResponseRedirect
from django.urls import path, reverse

from .models import PatientClinicalRecord

logger = logging.getLogger(__name__)


@admin.register(PatientClinicalRecord)
class PatientClinicalRecordAdmin(admin.ModelAdmin):
	list_display = (
		"patient",
		"diabetes_status",
		"data_consent_for_training",
		"gender",
		"age_years",
		"bmi",
		"hba1c_level",
		"blood_glucose_level",
		"recorded_at",
	)
	list_filter = ("diabetes_status", "data_consent_for_training")
	search_fields = ("patient__nume", "patient__prenume", "patient__CNP")
	list_select_related = ("patient",)
	change_list_template = "admin/clinical_ai/patientclinicalrecord/change_list.html"

	def get_urls(self):
		urls = super().get_urls()
		custom_urls = [
			path(
				"run-federated-transfer/",
				self.admin_site.admin_view(self.run_federated_transfer_view),
				name="clinical_ai_patientclinicalrecord_run_federated_transfer",
			),
		]
		return custom_urls + urls

	def run_federated_transfer_view(self, request: HttpRequest):
		model = request.GET.get("model", "alex5050")
		server_address = request.GET.get("server_address") or os.getenv(
			"FLOWER_SERVER_ADDRESS", "flower-server:8080"
		)
		min_samples = int(request.GET.get("min_samples", 1))

		if model not in {"alex5050", "mustafa"}:
			self.message_user(request, "Invalid model type.", level=messages.ERROR)
			return HttpResponseRedirect(self._changelist_url())

		command = (
			"docker compose -f docker/docker-compose.yml --profile federated "
			f"run --rm federated-client python manage.py run_flower_client --model {model} "
			f"--server-address {server_address} --min-samples {min_samples}"
		)

		# If docker CLI is available, run the docker compose command in background
		if shutil.which("docker"):
			def _run_cmd(cmd: str):
				try:
					logger.info("Starting federated transfer command: %s", cmd)
					# Use shlex.split to avoid shell=True where possible
					proc = subprocess.run(shlex.split(cmd), cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), capture_output=True, text=True)
					if proc.returncode == 0:
						logger.info("Federated transfer finished successfully: %s", proc.stdout)
					else:
						logger.error("Federated transfer failed (rc=%s): %s", proc.returncode, proc.stderr)
				except Exception as exc:
					logger.exception("Error running federated transfer: %s", exc)

			threading.Thread(target=_run_cmd, args=(command,), daemon=True).start()
			self.message_user(request, "Federated transfer started in background.", level=messages.SUCCESS)
			self.message_user(request, f"Command: {command}", level=messages.INFO)
		else:
			# Docker CLI not available in this runtime; fall back to showing the command
			self.message_user(
				request,
				"Docker CLI not available in this environment; cannot run the transfer automatically.",
				level=messages.WARNING,
			)
			self.message_user(
				request,
				f"Run this command from project root: {command}",
				level=messages.INFO,
			)

		return HttpResponseRedirect(self._changelist_url())

	def _changelist_url(self) -> str:
		return reverse("admin:clinical_ai_patientclinicalrecord_changelist")
