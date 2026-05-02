from django.contrib import admin

from .models import PatientClinicalRecord


@admin.register(PatientClinicalRecord)
class PatientClinicalRecordAdmin(admin.ModelAdmin):
	list_display = ("patient", "gender", "age_years", "bmi", "hba1c_level", "blood_glucose_level", "recorded_at")
	search_fields = ("patient__nume", "patient__prenume", "patient__CNP")
	list_select_related = ("patient",)
