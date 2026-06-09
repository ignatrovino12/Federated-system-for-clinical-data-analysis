import os
import threading
import logging
import json
from datetime import datetime, timezone

from django.contrib import admin, messages
from django.core.management import call_command
from django.http import HttpRequest, HttpResponseRedirect
from django.urls import path, reverse

try:
	from minio import Minio
	from minio.error import S3Error
except ImportError:
	Minio = None
	S3Error = Exception

from .federated_model_sync import get_local_federated_model_path
from .models import PatientClinicalRecord

logger = logging.getLogger(__name__)


@admin.register(PatientClinicalRecord)
class PatientClinicalRecordAdmin(admin.ModelAdmin):
	list_display = (
		"patient",
		"diabetes_status",
		"data_consent_for_training",
		"federated_train_count",
		"last_federated_train_at",
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
	ordering = ("-federated_train_count", "-last_federated_train_at", "-recorded_at")
	change_list_template = "admin/clinical_ai/patientclinicalrecord/change_list.html"

	def _minio_client(self):
		if Minio is None:
			logger.error(
				"MinIO is not installed in this environment; model update notices and sync actions are disabled until the package is available."
			)
			return None
		endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
		access_key = os.getenv("MINIO_ACCESS_KEY")
		secret_key = os.getenv("MINIO_SECRET_KEY")
		use_ssl = os.getenv("MINIO_USE_SSL", "false").lower() == "true"
		if not access_key or not secret_key:
			return None
		return Minio(endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=use_ssl)

	def _load_model_update_notice(self, model: str):
		client = self._minio_client()
		if client is None:
			return None

		bucket_name = os.getenv("MINIO_BUCKET_NAME", "models")
		object_name = f"control/model-updates/{model}.json"
		try:
			response = client.get_object(bucket_name, object_name)
			try:
				payload = json.loads(response.read().decode("utf-8"))
			finally:
				response.close()
				response.release_conn()
		except S3Error:
			return None
		except Exception:
			logger.exception("Failed to load model update notice for %s", model)
			return None

		updated_at_raw = str(payload.get("updated_at", ""))
		try:
			updated_at = datetime.fromisoformat(updated_at_raw.replace("Z", "+00:00"))
		except ValueError:
			return None

		local_path = get_local_federated_model_path(model)
		if local_path.exists():
			local_mtime = datetime.fromtimestamp(local_path.stat().st_mtime, tz=timezone.utc)
			if local_mtime >= updated_at:
				return None

		payload["model"] = model
		payload["updated_at"] = updated_at_raw
		return payload

	def get_urls(self):
		urls = super().get_urls()
		custom_urls = [
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

	def changelist_view(self, request, extra_context=None):
		extra_context = extra_context or {}
		pending_updates = [
			update
			for update in (
				self._load_model_update_notice("alex5050"),
				self._load_model_update_notice("mustafa"),
			)
			if update is not None
		]
		extra_context["pending_model_updates"] = pending_updates
		return super().changelist_view(request, extra_context=extra_context)

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
				access_key=os.getenv("MINIO_ACCESS_KEY"),
				secret_key=os.getenv("MINIO_SECRET_KEY"),
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
