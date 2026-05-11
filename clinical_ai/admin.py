import os
import threading
import logging

from django.contrib import admin, messages
from django.core.management import call_command
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
			path(
				"sync-model/<str:model>/",
				self.admin_site.admin_view(self.sync_model_view),
				name="clinical_ai_patientclinicalrecord_sync_model",
			),
			path(
				"clear-cache/<str:model>/",
				self.admin_site.admin_view(self.clear_cache_view),
				name="clinical_ai_patientclinicalrecord_clear_cache",
			),
		]
		return custom_urls + urls

	def _start_background_task(self, request: HttpRequest, description: str, task):
		threading.Thread(target=task, daemon=True).start()
		self.message_user(request, f"{description} started in background.", level=messages.SUCCESS)

	def _background_command(self, command_name: str, **kwargs):
		def _run():
			try:
				call_command(command_name, **kwargs)
				logger.info("%s completed successfully with args=%s", command_name, kwargs)
			except Exception as exc:
				logger.exception("%s failed with args=%s: %s", command_name, kwargs, exc)
		return _run

	def run_federated_transfer_view(self, request: HttpRequest):
		model = request.GET.get("model", "alex5050")
		server_address = request.GET.get("server_address") or os.getenv(
			"FLOWER_SERVER_ADDRESS", "flower-server:8080"
		)
		min_samples = int(request.GET.get("min_samples", 1))
		test_split = float(request.GET.get("test_split", 0.2))
		batch_size = int(request.GET.get("batch_size", 32))

		if model not in {"alex5050", "mustafa"}:
			self.message_user(request, "Invalid model type.", level=messages.ERROR)
			return HttpResponseRedirect(self._changelist_url())

		def _run_training():
			try:
				logger.info(
					"Starting federated training via management command: model=%s server=%s min_samples=%s test_split=%s batch_size=%s",
					model,
					server_address,
					min_samples,
					test_split,
					batch_size,
				)
				call_command(
					"run_flower_client",
					model=model,
					server_address=server_address,
					min_samples=min_samples,
					test_split=test_split,
					batch_size=batch_size,
				)
				logger.info("Federated training finished successfully for model=%s", model)
			except Exception as exc:
				logger.exception("Federated training failed for model=%s: %s", model, exc)

		threading.Thread(target=_run_training, daemon=True).start()
		self.message_user(request, f"Federated training for {model} started in background.", level=messages.SUCCESS)

		return HttpResponseRedirect(self._changelist_url())

	def sync_model_view(self, request: HttpRequest, model: str):
		if model not in {"alex5050", "mustafa"}:
			self.message_user(request, "Invalid model type.", level=messages.ERROR)
			return HttpResponseRedirect(self._changelist_url())

		self._start_background_task(
			request,
			f"Sync for {model}",
			self._background_command(
				"sync_federated_model",
				model=model,
				endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"),
				access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
				secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
			),
		)
		return HttpResponseRedirect(self._changelist_url())

	def clear_cache_view(self, request: HttpRequest, model: str):
		if model not in {"alex5050", "mustafa"}:
			self.message_user(request, "Invalid model type.", level=messages.ERROR)
			return HttpResponseRedirect(self._changelist_url())

		self._start_background_task(
			request,
			f"Cache clear for {model}",
			self._background_command("clear_model_cache", model=model),
		)
		return HttpResponseRedirect(self._changelist_url())

	def _changelist_url(self) -> str:
		return reverse("admin:clinical_ai_patientclinicalrecord_changelist")
