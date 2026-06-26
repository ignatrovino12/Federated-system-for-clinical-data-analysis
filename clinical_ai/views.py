from functools import wraps

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.shortcuts import get_object_or_404, redirect, render

from pubsub.models import AccessLog, Appointment, Patient
from pubsub.roles import get_role_info, has_any_role, is_in_group
from pubsub.views import permission_required_or_role, is_doctor_like, is_group_doctor

from healthcheck.metrics import record_analysis_request

from .forms import (
    AlexManualAnalysisForm,
    ClinicalAnalysisSelectionForm,
    MustafaManualAnalysisForm,
    PatientClinicalRecordForm,
)
from .models import PatientClinicalRecord


def get_user_profile(user):
    return get_role_info(user)


def role_required(allowed_roles):
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            profile = get_user_profile(request.user)
            if not has_any_role(request.user, allowed_roles):
                messages.error(request, "You do not have permission to access this resource.")
                return redirect("pubsub:dashboard")
            return view_func(request, *args, **kwargs)

        return wrapper

    return decorator


def get_client_ip(request):
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0]
    return request.META.get("REMOTE_ADDR")


def log_access(user, patient, action, request, details=""):
    AccessLog.objects.create(
        user=user,
        patient=patient,
        action=action,
        ip_address=get_client_ip(request),
        details=details,
    )


def can_access_patient(request_user, patient):
    profile = get_user_profile(request_user)
    # Admins by group membership have full access
    try:
        if is_in_group(request_user, 'admin'):
            return True, profile
    except Exception:
        pass

    # Doctors may access patients assigned to them or where they have appointments
    if is_group_doctor(request_user) or is_doctor_like(request_user):
        allowed = (
            patient.assigned_doctor_id == request_user.id
            or Appointment.objects.filter(doctor=request_user, patient=patient).exists()
        )
        return allowed, profile

    return False, profile


def get_accessible_patients(request_user, profile):
    try:
        if request_user.groups.filter(name='admin').exists():
            return Patient.objects.all().order_by("nume", "prenume")
    except Exception:
        pass

    return (
        Patient.objects.filter(
            Q(appointments__doctor=request_user) | Q(assigned_doctor=request_user)
        )
        .distinct()
        .order_by("nume", "prenume")
    )


def get_unread_appointments_count(user):
    return Appointment.objects.filter(doctor=user, notification_read=False).count()


@login_required
@permission_required_or_role('pubsub.view_patient', allowed_roles=['admin', 'doctor'])
def analysis_dashboard_view(request):
    profile = get_user_profile(request.user)
    accessible_patients = get_accessible_patients(request.user, profile)

    alex_form = AlexManualAnalysisForm(prefix="alex")
    mustafa_form = MustafaManualAnalysisForm(prefix="mustafa")
    selection_form = ClinicalAnalysisSelectionForm(prefix="selection", patient_queryset=accessible_patients)

    manual_result = None
    quick_result = None
    manual_error = None
    quick_error = None
    quick_patient = None

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "manual_alex":
            alex_form = AlexManualAnalysisForm(request.POST, prefix="alex")
            if alex_form.is_valid():
                try:
                    from .services import predict_alex_probability

                    manual_result = {
                        "model_label": "Lifestyle-Based Diabetes Risk Model",
                        "patient_label": None,
                    }
                    manual_result.update(predict_alex_probability(alex_form.to_feature_payload()))
                    log_access(request.user, None, "search", request, details="Manual Lifestyle-Based Diabetes Risk Model analysis")
                    record_analysis_request("alex5050", "manual", "success")
                except ValueError as exc:
                    manual_error = str(exc)
                    record_analysis_request("alex5050", "manual", "error")
                else:
                    manual_error = "Please complete all fields in the Lifestyle-Based Diabetes Risk Model section."
                record_analysis_request("alex5050", "manual", "invalid_form")

        elif action == "manual_mustafa":
            mustafa_form = MustafaManualAnalysisForm(request.POST, prefix="mustafa")
            if mustafa_form.is_valid():
                try:
                    from .services import predict_mustafa_probability

                    manual_result = {
                        "model_label": "Clinical Diabetes Risk Model",
                        "patient_label": None,
                    }
                    manual_result.update(predict_mustafa_probability(mustafa_form.to_feature_payload()))
                    log_access(request.user, None, "search", request, details="Manual Clinical Diabetes Risk Model analysis")
                    record_analysis_request("mustafa", "manual", "success")
                except ValueError as exc:
                    manual_error = str(exc)
                    record_analysis_request("mustafa", "manual", "error")
            else:
                manual_error = "Please complete all fields in the Clinical Diabetes Risk Model section."
                record_analysis_request("mustafa", "manual", "invalid_form")

        elif action == "quick_patient":
            selection_form = ClinicalAnalysisSelectionForm(request.POST, prefix="selection", patient_queryset=accessible_patients)
            if selection_form.is_valid():
                quick_patient = selection_form.cleaned_data["patient"]
                selected_model = selection_form.cleaned_data["model"]
                selected_model_name = "alex5050" if selected_model == "alex" else "mustafa"
                clinical_record = PatientClinicalRecord.objects.filter(patient=quick_patient).first()

                if clinical_record is None:
                    quick_error = "This patient does not have clinical data saved yet."
                    record_analysis_request(selected_model_name, "quick", "missing_clinical_record")
                else:
                    try:
                        if selected_model == "alex":
                            from .services import predict_alex_probability

                            quick_result = predict_alex_probability(clinical_record.alex5050_features())
                            quick_result["model_label"] = "Lifestyle-Based Diabetes Risk Model"
                        else:
                            from .services import predict_mustafa_probability

                            quick_result = predict_mustafa_probability(clinical_record.mustafa_features())
                            quick_result["model_label"] = "Clinical Diabetes Risk Model"

                        quick_result["patient_label"] = quick_patient.get_full_name()
                        log_access(
                            request.user,
                            quick_patient,
                            "view",
                            request,
                            details=f"Ran {selected_model} analysis on stored clinical data",
                        )
                        record_analysis_request(selected_model_name, "quick", "success")
                    except ValueError as exc:
                        quick_error = str(exc)
                        record_analysis_request(selected_model_name, "quick", "error")
            else:
                quick_error = "Please choose both a patient and a model."
                record_analysis_request("unknown", "quick", "invalid_form")

    context = {
        "user_profile": profile,
        "alex_form": alex_form,
        "mustafa_form": mustafa_form,
        "selection_form": selection_form,
        "manual_result": manual_result,
        "quick_result": quick_result,
        "manual_error": manual_error,
        "quick_error": quick_error,
        "quick_patient": quick_patient,
        "unread_appointments_count": get_unread_appointments_count(request.user),
    }
    return render(request, "clinical_ai/analysis_dashboard.html", context)


@login_required
@permission_required_or_role('pubsub.change_patient', allowed_roles=['admin', 'doctor'], obj_getter=lambda request, *a, **kw: get_object_or_404(Patient, id=kw.get('patient_id')))
def patient_medical_data_view(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id)
    allowed, profile = can_access_patient(request.user, patient)

    if not allowed:
        messages.error(request, "You do not have permission to access this patient's medical data.")
        return redirect("pubsub:patient_list")

    clinical_record, _ = PatientClinicalRecord.objects.get_or_create(patient=patient)

    if request.method == "POST":
        form = PatientClinicalRecordForm(request.POST, instance=clinical_record)
        if form.is_valid():
            form.save()
            log_access(request.user, patient, "edit", request, details="Updated clinical medical data")
            messages.success(request, f"Medical data for {patient.get_full_name()} updated successfully!")
            return redirect("pubsub:patient_detail", patient_id=patient.id)
        messages.error(request, "Please correct the errors below.")
    else:
        form = PatientClinicalRecordForm(instance=clinical_record)

    log_access(request.user, patient, "view", request, details="Viewed medical data section")

    return render(
        request,
        "clinical_ai/patient_clinical_record_form.html",
        {
            "patient": patient,
            "form": form,
            "user_profile": profile,
            "clinical_record": clinical_record,
            "can_edit": (request.user.has_perm('pubsub.change_patient') or is_doctor_like(request.user)),
            "unread_appointments_count": Appointment.objects.filter(doctor=request.user, notification_read=False).count(),
        },
    )

