from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from django.db.models import Q
from django.http import JsonResponse
from functools import wraps
from .models import Patient, UserProfile, AccessLog
from .forms import PatientForm


def get_client_ip(request):
    """Get client IP address from request"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


def log_access(user, patient, action, request, details=""):
    """Create an access log entry"""
    AccessLog.objects.create(
        user=user,
        patient=patient,
        action=action,
        ip_address=get_client_ip(request),
        details=details
    )


def get_user_profile(user):
    """Get or create user profile"""
    profile, created = UserProfile.objects.get_or_create(
        user=user,
        defaults={'role': 'receptionist'}
    )
    return profile


def role_required(allowed_roles):
    """Decorator to check if user has required role"""
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            profile = get_user_profile(request.user)
            if profile.role not in allowed_roles:
                messages.error(request, 'You do not have permission to access this resource.')
                return redirect('pubsub:dashboard')
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator


def login_view(request):
    """Login view for clinic personnel"""
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
    """Logout view for clinic personnel"""
    logout(request)
    messages.info(request, 'You have been logged out successfully.')
    return redirect('pubsub:login')


@login_required
def dashboard_view(request):
    """Dashboard view for authenticated clinic personnel"""
    from .models import Patient
    
    profile = get_user_profile(request.user)
    
    context = {
        'user': request.user,
        'user_profile': profile,
        'total_patients': Patient.objects.count(),
    }
    return render(request, 'main/dashboard.html', context)


@login_required
def profile_view(request):
    """Profile view for clinic personnel"""
    profile = get_user_profile(request.user)
    
    # Get user's recent access logs
    recent_logs = AccessLog.objects.filter(user=request.user).select_related('patient')[:20]
    
    context = {
        'user': request.user,
        'user_profile': profile,
        'recent_logs': recent_logs,
    }
    return render(request, 'main/profile.html', context)


@login_required
def patient_list_view(request):
    """List patients with search functionality and masked data"""
    profile = get_user_profile(request.user)
    
    # Get search query
    search_query = request.GET.get('search', '').strip()
    
    # Start with no patients (search-first approach)
    if search_query:
        # Log search action
        log_access(request.user, None, 'search', request, details=f"Searched: {search_query}")
        
        # Search by name, CNP (partial), city, or county
        patients = Patient.objects.filter(
            Q(nume__icontains=search_query) |
            Q(prenume__icontains=search_query) |
            Q(CNP__icontains=search_query) |
            Q(oras__icontains=search_query) |
            Q(judet__icontains=search_query)
        ).order_by('-created_at')
    else:
        patients = Patient.objects.none()
    
    context = {
        'patients': patients,
        'total_patients': Patient.objects.count(),
        'search_query': search_query,
        'user_profile': profile,
        'show_full_data': profile.can_view_full_data(),
    }
    return render(request, 'patient/patient_list.html', context)


@login_required
@role_required(['admin', 'doctor', 'receptionist'])
def patient_detail_view(request, patient_id):
    """View full patient details with audit logging"""
    patient = get_object_or_404(Patient, id=patient_id)
    profile = get_user_profile(request.user)
    
    # Log access to patient data
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
    }
    return render(request, 'patient/patient_detail.html', context)


@login_required
@role_required(['admin', 'doctor', 'receptionist'])
def patient_add_view(request):
    """Add a new patient"""
    profile = get_user_profile(request.user)
    
    if request.method == 'POST':
        form = PatientForm(request.POST)
        if form.is_valid():
            patient = form.save()
            # Log patient creation
            log_access(request.user, patient, 'create', request, details=f"Created patient {patient.get_full_name()}")
            messages.success(request, f'Patient {patient.get_full_name()} added successfully!')
            return redirect('pubsub:patient_list')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = PatientForm()
    
    return render(request, 'patient/patient_add.html', {'form': form, 'user_profile': profile})


@login_required
@role_required(['admin', 'doctor'])
def patient_edit_view(request, patient_id):
    """Edit an existing patient (doctors and admins only)"""
    patient = get_object_or_404(Patient, id=patient_id)
    profile = get_user_profile(request.user)
    
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
    
    return render(request, 'patient/patient_edit.html', {'form': form, 'patient': patient, 'user_profile': profile})


@login_required
@role_required(['admin'])
def patient_delete_view(request, patient_id):
    """Delete a patient (admins only)"""
    patient = get_object_or_404(Patient, id=patient_id)
    profile = get_user_profile(request.user)
    
    if request.method == 'POST':
        patient_name = patient.get_full_name()
        patient_cnp = patient.CNP
        # Log patient deletion before deleting
        log_access(request.user, patient, 'delete', request, details=f"Deleted patient {patient_name} (CNP: {patient_cnp})")
        patient.delete()
        messages.success(request, f'Patient {patient_name} deleted successfully!')
        return redirect('pubsub:patient_list')
    
    return render(request, 'patient/patient_delete.html', {'patient': patient, 'user_profile': profile})


