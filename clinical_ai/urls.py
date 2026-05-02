from django.urls import path

from . import views

app_name = "clinical_ai"

urlpatterns = [
    path("analysis/", views.analysis_dashboard_view, name="analysis_dashboard"),
    path("patients/<int:patient_id>/medical-data/", views.patient_medical_data_view, name="patient_medical_data"),
]