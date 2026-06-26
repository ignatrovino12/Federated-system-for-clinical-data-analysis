import importlib

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from django.db.models import Q
from django.http import JsonResponse
from django.core.exceptions import ValidationError
from django.utils import timezone
from functools import wraps
from datetime import datetime, date
from .models import Patient, AccessLog, Appointment
from .roles import get_role_info, has_any_role, is_in_group
from .forms import PatientForm
from .redis_pubsub import AppointmentNotifier
from .appointment_lifecycle import expire_completed_appointments

try:
    assign_perm = importlib.import_module("guardian.shortcuts").assign_perm
except Exception: 
    def assign_perm(perm, user, obj=None):
        return None


# UTILITY FUNCTIONS

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


def log_access(user, patient, action, request, details=""):
    AccessLog.objects.create(
        user=user,
        patient=patient,
        action=action,
        ip_address=get_client_ip(request),
        details=details
    )


def get_user_profile(user):
    return get_role_info(user)


def get_unread_appointments_count(user):
    # Count unread appointments where the user is the doctor (works regardless of role flags)
    return Appointment.objects.filter(doctor=user, notification_read=False).count()


def doctor_has_patient_access(patient, doctor):
    return (
        patient.assigned_doctor_id == doctor.id
        or Appointment.objects.filter(doctor=doctor, patient=patient).exists()
    )


def is_doctor_like(user):
    return Patient.objects.filter(assigned_doctor=user).exists() or Appointment.objects.filter(doctor=user).exists()


def is_group_doctor(user):
    return is_in_group(user, 'doctor')


def role_required(allowed_roles):
    # Decorator to check if user has required role
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            profile = get_user_profile(request.user)
            if not has_any_role(request.user, allowed_roles):
                messages.error(request, 'You do not have permission to access this resource.')
                return redirect('pubsub:dashboard')
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator


def permission_required_or_role(perm, allowed_roles=None, obj_getter=None):
    # Decorator to allow when user has `perm` (model or object) or when user's role is in allowed_roles.


    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            user = request.user
            # model-level permission
            try:
                if user.has_perm(perm):
                    return view_func(request, *args, **kwargs)
            except Exception:
                # in case auth backend not ready
                pass

            # object-level permission if getter provided
            if obj_getter is not None:
                try:
                    obj = obj_getter(request, *args, **kwargs)
                    if user.has_perm(perm, obj):
                        return view_func(request, *args, **kwargs)
                except Exception:
                    # lookup failed or guardian not installed — fallthrough to role check
                    pass

            if allowed_roles and has_any_role(request.user, allowed_roles):
                return view_func(request, *args, **kwargs)

            messages.error(request, 'You do not have permission to access this resource.')
            return redirect('pubsub:dashboard')

        return wrapper
    return decorator


# AUTHENTICATION AND DASHBOARD VIEWS

def login_view(request):
  
    if request.user.is_authenticated:
        return redirect('pubsub:dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, f'Welcome back, {user.username}!')
            next_url = request.GET.get('next', 'pubsub:dashboard')
            return redirect(next_url)
        else:
            messages.error(request, 'Invalid username or password.')
    
    return render(request, 'main/login.html')


def logout_view(request):
   
    logout(request)
    messages.info(request, 'You have been logged out successfully.')
    return redirect('pubsub:login')


@login_required
def dashboard_view(request):
 
    from .models import Patient
    
    profile = get_user_profile(request.user)

    # Get appointment statistics
    today = date.today()
    # If the user is a practicing doctor (has appointments or assigned patients) show only own appointments
    if is_doctor_like(request.user):
        appointments_query = Appointment.objects.filter(doctor=request.user)
    else:
        appointments_query = Appointment.objects.all()
    
    today_appointments = appointments_query.filter(appointment_date=today).count()
    upcoming_appointments = appointments_query.filter(
        appointment_date__gte=today,
        status__in=['scheduled', 'confirmed']
    ).count()
    
    context = {
        'user': request.user,
        'user_profile': profile,
        'total_patients': Patient.objects.count(),
        'today_appointments': today_appointments,
        'upcoming_appointments': upcoming_appointments,
        'unread_appointments_count': get_unread_appointments_count(request.user),
    }
    return render(request, 'main/dashboard.html', context)


@login_required
def profile_view(request):

    profile = get_user_profile(request.user)
    
    # Get user's recent access logs
    recent_logs = AccessLog.objects.filter(user=request.user).select_related('patient')[:20]
    
    context = {
        'user': request.user,
        'user_profile': profile,
        'recent_logs': recent_logs,
        'unread_appointments_count': get_unread_appointments_count(request.user),
    }
    return render(request, 'main/profile.html', context)


# PATIENT VIEWS

@login_required
def patient_list_view(request):
    
    profile = get_user_profile(request.user)
    min_search_length = 3

    # Get search query
    search_query = request.GET.get('search', '').strip()
    search_too_short = bool(search_query) and len(search_query) < min_search_length

    if is_doctor_like(request.user):
        # Doctors should see patients assigned to them or linked through appointments
        patients = Patient.objects.none()
        if search_query and not search_too_short:
            log_access(request.user, None, 'search', request, details=f"Searched: {search_query}")
            patients = (
                Patient.objects.filter(
                    Q(appointments__doctor=request.user) |
                    Q(assigned_doctor=request.user)
                )
                .filter(
                    Q(nume__icontains=search_query) |
                    Q(prenume__icontains=search_query)
                )
                .distinct()
                .order_by('prenume', 'nume')
            )
        elif not search_query:
            patients = (
                Patient.objects.filter(
                    Q(appointments__doctor=request.user) |
                    Q(assigned_doctor=request.user)
                )
                .distinct()
                .order_by('prenume', 'nume')
            )
        total_patients = patients.count()
    else:
        # Receptionists/admins keep the existing search-first workflow
        if search_query and not search_too_short:
            log_access(request.user, None, 'search', request, details=f"Searched: {search_query}")
            patients = Patient.objects.filter(
                Q(nume__icontains=search_query) |
                Q(prenume__icontains=search_query)
            ).order_by('-created_at')
        else:
            patients = Patient.objects.none()
        total_patients = Patient.objects.count()
    
    context = {
        'patients': patients,
        'total_patients': total_patients,
        'search_query': search_query,
        'user_profile': profile,
        'show_full_data': profile.can_view_full_data(),
        'is_doctor_patient_list': is_doctor_like(request.user),
        'search_too_short': search_too_short,
        'min_search_length': min_search_length,
        'unread_appointments_count': get_unread_appointments_count(request.user),
    }
    return render(request, 'patient/patient_list.html', context)


@login_required
@permission_required_or_role(
    'pubsub.view_patient',
    allowed_roles=['admin', 'doctor', 'receptionist'],
    obj_getter=lambda request, *a, **kw: get_object_or_404(Patient, id=kw.get('patient_id')),
)
def patient_detail_view(request, patient_id):
    
    patient = get_object_or_404(Patient, id=patient_id)
    profile = get_user_profile(request.user)
    
    # Doctors can only view patients they have appointments with
    if is_doctor_like(request.user):
        if not doctor_has_patient_access(patient, request.user):
            messages.error(request, 'You do not have permission to view this patient. You can only view patients assigned to you or with whom you have appointments.')
            return redirect('pubsub:patient_list')
    
    # Log view action with patient details 
    log_access(request.user, patient, 'view', request, details="Viewed full patient details") 
    
    # Get recent access logs for this patient
    recent_logs = AccessLog.objects.filter(patient=patient).select_related('user')[:10]
    
    context = {
        'patient': patient,
        'user_profile': profile,
        'can_edit': profile.can_edit_data(),
        'can_delete': profile.can_delete_data(),
        'show_full_data': profile.can_view_full_data(),
        'recent_logs': recent_logs,
        'unread_appointments_count': get_unread_appointments_count(request.user),
    }
    return render(request, 'patient/patient_detail.html', context)


@login_required
@permission_required_or_role('pubsub.add_patient', allowed_roles=['admin', 'doctor', 'receptionist'])
def patient_add_view(request):
    
    profile = get_user_profile(request.user)
    
    if request.method == 'POST':
        form = PatientForm(request.POST)
        if form.is_valid():
            patient = form.save(commit=False)
            if is_group_doctor(request.user):
                patient.assigned_doctor = request.user
            else:
                patient.assigned_doctor = None
            patient.save()
            # Grant object-level permissions to assigned doctor
            if patient.assigned_doctor_id is not None:
                try:
                    assign_perm('view_patient', patient.assigned_doctor, patient)
                    assign_perm('change_patient', patient.assigned_doctor, patient)
                    assign_perm('delete_patient', patient.assigned_doctor, patient)
                except Exception:
                    pass
            # Log patient creation
            if is_group_doctor(request.user):
                details = f"Created patient {patient.get_full_name()} and assigned to Dr. {request.user.get_full_name() or request.user.username}"
                success_message = f'Patient {patient.get_full_name()} added and assigned to you successfully!'
            else:
                details = f"Created patient {patient.get_full_name()}"
                success_message = f'Patient {patient.get_full_name()} added successfully!'

            log_access(request.user, patient, 'create', request, details=details)
            messages.success(request, success_message)
            return redirect('pubsub:patient_list')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = PatientForm()
    
    return render(request, 'patient/patient_add.html', {
        'form': form,
        'user_profile': profile,
        'unread_appointments_count': get_unread_appointments_count(request.user),
    })


@login_required
@permission_required_or_role(
    'pubsub.change_patient',
    allowed_roles=['admin', 'doctor'],
    obj_getter=lambda request, *a, **kw: get_object_or_404(Patient, id=kw.get('patient_id')),
)
def patient_edit_view(request, patient_id):

    patient = get_object_or_404(Patient, id=patient_id)
    profile = get_user_profile(request.user)
    
    # Doctors can only edit patients they have appointments with
    if is_doctor_like(request.user):
        if not doctor_has_patient_access(patient, request.user):
            messages.error(request, 'You do not have permission to edit this patient. You can only edit patients assigned to you or with whom you have appointments.')
            return redirect('pubsub:patient_list')
    
    if request.method == 'POST':
        form = PatientForm(request.POST, instance=patient)
        if form.is_valid():
            patient = form.save()
            # Log patient edit
            log_access(request.user, patient, 'edit', request, details=f"Updated patient {patient.get_full_name()}")
            messages.success(request, f'Patient {patient.get_full_name()} updated successfully!')
            return redirect('pubsub:patient_detail', patient_id=patient.id)
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = PatientForm(instance=patient)
    
    return render(request, 'patient/patient_edit.html', {
        'form': form,
        'patient': patient,
        'user_profile': profile,
        'unread_appointments_count': get_unread_appointments_count(request.user),
    })


@login_required
@permission_required_or_role('pubsub.delete_patient', allowed_roles=['admin'])
def patient_delete_view(request, patient_id):
  
    patient = get_object_or_404(Patient, id=patient_id)
    profile = get_user_profile(request.user)
    
    if request.method == 'POST':
        patient_name = patient.get_full_name()
        patient_cnp = patient.CNP

        # Log patient delete
        log_access(request.user, patient, 'delete', request, details=f"Deleted patient {patient_name} (CNP: {patient_cnp})")
        patient.delete()
        messages.success(request, f'Patient {patient_name} deleted successfully!')
        return redirect('pubsub:patient_list')
    
    return render(request, 'patient/patient_delete.html', {
        'patient': patient,
        'user_profile': profile,
        'unread_appointments_count': get_unread_appointments_count(request.user),
    })


# APPOINTMENT VIEWS

@login_required
@permission_required_or_role('pubsub.view_appointment', allowed_roles=['admin', 'doctor', 'receptionist'])
def appointment_list_view(request):

    profile = get_user_profile(request.user)
    expire_completed_appointments()
    
    # Get base queryset based on whether user is a practicing doctor
    if is_doctor_like(request.user):
        # Doctors only see their own appointments
        appointments = Appointment.objects.filter(doctor=request.user).select_related('patient', 'doctor', 'created_by')
    else:
        # Admins and receptionists see all appointments
        appointments = Appointment.objects.all().select_related('patient', 'doctor', 'created_by')
    
    # Apply filters
    status_filter = request.GET.get('status', '')
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    
    if status_filter:
        appointments = appointments.filter(status=status_filter)
    if date_from:
        appointments = appointments.filter(appointment_date__gte=date_from)
    if date_to:
        appointments = appointments.filter(appointment_date__lte=date_to)
    
    appointments = appointments.order_by('-appointment_date', '-appointment_time')
    
    # Calculate statistics
    today = date.today()
    total_count = appointments.count()
    today_count = appointments.filter(appointment_date=today).count()
    upcoming_count = appointments.filter(appointment_date__gte=today, status__in=['scheduled', 'confirmed']).count()
    
    # Unread notifications: count unread notifications where current user is the doctor
    unread_count = appointments.filter(doctor=request.user, notification_read=False).count()
    
    context = {
        'appointments': appointments,
        'user_profile': profile,
        'status_filter': status_filter,
        'start_date': date_from,
        'end_date': date_to,
        'stats': {
            'total': total_count,
            'today': today_count,
            'upcoming': upcoming_count,
            'unread': unread_count,
        },
        'status_choices': Appointment.STATUS_CHOICES,
        'unread_appointments_count': unread_count,
        'is_doctor_like': is_doctor_like(request.user),
    }
    
    return render(request, 'appointment/appointment_list.html', context)


@login_required
@permission_required_or_role('pubsub.add_appointment', allowed_roles=['admin', 'receptionist', 'doctor'])
def appointment_create_view(request):
    
    profile = get_user_profile(request.user)
    
    if request.method == 'POST':
        patient_id = request.POST.get('patient_id')
        doctor_id = request.POST.get('doctor_id')
        appointment_date = request.POST.get('appointment_date')
        appointment_time = request.POST.get('appointment_time')
        duration_minutes = request.POST.get('duration_minutes', 30)
        reason = request.POST.get('reason')
        notes = request.POST.get('notes', '')
        
        # Doctors can only create appointments for themselves
        if is_group_doctor(request.user):
            doctor_id = str(request.user.id)
        
        # Validate required fields
        if not patient_id or not doctor_id:
            messages.error(request, 'Please select both a patient and a doctor.')
            return redirect('pubsub:appointment_create')
        
        try:
            patient = Patient.objects.get(id=patient_id)
            doctor = User.objects.get(id=doctor_id)
            
            # Check for conflicts
            conflicts = Appointment.objects.filter(
                doctor=doctor,
                appointment_date=appointment_date,
                appointment_time=appointment_time
            ).exists()
            
            if conflicts:
                messages.error(request, 'This time slot is already booked for the selected doctor.')
                return redirect('pubsub:appointment_create')
            
            # Create appointment
            appointment = Appointment.objects.create(
                patient=patient,
                doctor=doctor,
                appointment_date=appointment_date,
                appointment_time=appointment_time,
                duration_minutes=duration_minutes,
                reason=reason,
                notes=notes,
                created_by=request.user
            )

            if patient.assigned_doctor_id is None:
                patient.assigned_doctor = doctor
                patient.save(update_fields=['assigned_doctor'])
                # assign object perms to doctor when auto-assigned during appointment creation
                try:
                    assign_perm('view_patient', doctor, patient)
                    assign_perm('change_patient', doctor, patient)
                    assign_perm('delete_patient', doctor, patient)
                except Exception:
                    pass
            
            # Send Redis notification when appointment is created, but only if doctor is different from creator
            if appointment.created_by_id == appointment.doctor_id:
                appointment.notification_sent = True
                appointment.notification_read = True
                appointment.save(update_fields=['notification_sent', 'notification_read'])
            else:
                try:
                    notifier = AppointmentNotifier()
                    notifier.publish_appointment_created(appointment)
                    appointment.notification_sent = True
                    appointment.save(update_fields=['notification_sent'])
                except Exception as e:
                    print(f"Redis notification failed: {e}")
            
            # Log action
            log_access(request.user, patient, 'appointment_create', request, 
                      details=f"Created appointment with Dr. {doctor.get_full_name()} on {appointment_date}")
            
            messages.success(request, 'Appointment created successfully!')
            return redirect('pubsub:appointment_list')
            
        except Patient.DoesNotExist:
            messages.error(request, f'Patient with ID {patient_id} not found.')
            return redirect('pubsub:appointment_create')
        except User.DoesNotExist:
            messages.error(request, f'Doctor with ID {doctor_id} not found.')
            return redirect('pubsub:appointment_create')
        except ValidationError as e:
            # Surface model validation errors (including overlapping interval)
            if hasattr(e, 'message_dict'):
                for field_errors in e.message_dict.values():
                    for err in field_errors:
                        messages.error(request, err)
            else:
                messages.error(request, str(e))
            return redirect('pubsub:appointment_create')
        except Exception as e:
            messages.error(request, f'Error creating appointment: {str(e)}')
            return redirect('pubsub:appointment_create')
    
    # Get patients and doctors for selection
    if is_doctor_like(request.user):
        patients = Patient.objects.filter(
            Q(appointments__doctor=request.user) | Q(assigned_doctor=request.user)
        ).distinct().order_by('nume', 'prenume')[:50]
    else:
        patients = Patient.objects.all().order_by('nume', 'prenume')[:50]  # Limit for performance
    
    doctors = User.objects.filter(groups__name='doctor').order_by('first_name', 'last_name')
    
    context = {
        'patients': patients,
        'doctors': doctors,
        'user_profile': profile,
        'unread_appointments_count': get_unread_appointments_count(request.user),
        'is_doctor_creator': is_group_doctor(request.user),
    }
    
    return render(request, 'appointment/appointment_create.html', context)


@login_required
@role_required(['admin', 'doctor', 'receptionist'])
def appointment_detail_view(request, appointment_id):

    expire_completed_appointments()
    appointment = get_object_or_404(Appointment, id=appointment_id)
    profile = get_user_profile(request.user)
    
    # Check permissions
    if is_doctor_like(request.user) and appointment.doctor != request.user:
        messages.error(request, 'You do not have permission to view this appointment.')
        return redirect('pubsub:appointment_list')
    
    # Mark as read for doctors
    if is_doctor_like(request.user) and not appointment.notification_read:
        appointment.notification_read = True
        appointment.save(update_fields=['notification_read'])
    
    # Log access
    log_access(request.user, appointment.patient, 'appointment_view', request,
              details=f"Viewed appointment #{appointment.id}")
    
    context = {
        'appointment': appointment,
        'user_profile': profile,
        'can_edit': (request.user.has_perm('pubsub.change_appointment') or is_doctor_like(request.user)),
        'unread_appointments_count': get_unread_appointments_count(request.user),
    }
    
    return render(request, 'appointment/appointment_detail.html', context)


@login_required
@permission_required_or_role(
    'pubsub.change_appointment',
    allowed_roles=['admin', 'doctor'],
    obj_getter=lambda request, *a, **kw: get_object_or_404(Appointment, id=kw.get('appointment_id')),
)
def appointment_update_status_view(request, appointment_id):

    expire_completed_appointments()
    appointment = get_object_or_404(Appointment, id=appointment_id)
    profile = get_user_profile(request.user)
    
    # Check permissions - preserve doctor business rule using doctor-like heuristic
    if is_doctor_like(request.user) and appointment.doctor != request.user:
        messages.error(request, 'You do not have permission to modify this appointment.')
        return redirect('pubsub:appointment_list')
    
    if request.method == 'POST':
        new_status = request.POST.get('status')
        notes = request.POST.get('notes', '')

        if new_status == 'expired':
            messages.error(request, 'Expired status is set automatically by the system.')
            return redirect('pubsub:appointment_detail', appointment_id=appointment.id)
        
        if new_status in dict(Appointment.STATUS_CHOICES):
            old_status = appointment.status
            appointment.status = new_status
            if new_status == 'completed':
                appointment.completed_at = timezone.now()
            elif old_status == 'completed' and new_status != 'completed':
                appointment.completed_at = None
            if notes:
                appointment.notes = appointment.notes + f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {notes}"
            appointment.save()
            
            # Send Redis notification about status update
            try:
                notifier = AppointmentNotifier()
                notifier.publish_appointment_updated(appointment, old_status, new_status)
            except Exception as e:
                print(f"Redis notification failed: {e}")
            
            # Log action
            log_access(request.user, appointment.patient, 'appointment_update', request,
                      details=f"Updated appointment #{appointment.id} status from {old_status} to {new_status}")
            
            messages.success(request, f'Appointment status updated to {appointment.get_status_display()}')
        else:
            messages.error(request, 'Invalid status selected.')
    
    return redirect('pubsub:appointment_detail', appointment_id=appointment.id)


@login_required
@permission_required_or_role('pubsub.delete_appointment', allowed_roles=['admin', 'doctor'])
def appointment_delete_view(request, appointment_id):

    expire_completed_appointments()
    appointment = get_object_or_404(Appointment, id=appointment_id)
    profile = get_user_profile(request.user)

    if is_doctor_like(request.user) and appointment.doctor != request.user:
        messages.error(request, 'You do not have permission to delete this appointment.')
        return redirect('pubsub:appointment_list')

    if request.method != 'POST':
        messages.error(request, 'Invalid request method.')
        return redirect('pubsub:appointment_detail', appointment_id=appointment.id)

    if appointment.status not in {'completed', 'expired', 'cancelled', 'no_show'}:
        messages.error(request, 'Only completed, expired, cancelled, or no-show appointments can be deleted.')
        return redirect('pubsub:appointment_detail', appointment_id=appointment.id)

    appointment_id_label = appointment.id
    patient = appointment.patient
    appointment.delete()
    log_access(
        request.user,
        patient,
        'delete',
        request,
        details=f"Deleted appointment #{appointment_id_label}",
    )
    messages.success(request, f'Appointment #{appointment_id_label} deleted successfully.')
    return redirect('pubsub:appointment_list')


@login_required
def appointment_search_patients_ajax(request):
  
    query = request.GET.get('q', '')
    
    if len(query) < 2:
        return JsonResponse({'results': []})
    
    patients = Patient.objects.filter(
        Q(nume__icontains=query) | 
        Q(prenume__icontains=query) |
        Q(CNP__icontains=query)
    )[:10]
    
    results = [{
        'id': p.id,
        'text': f"{p.get_full_name()} - CNP: {p.get_masked_cnp()}"
    } for p in patients]
    
    return JsonResponse({'results': results})
