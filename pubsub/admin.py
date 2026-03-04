from django.contrib import admin
from .models import Patient, UserProfile, AccessLog, Appointment


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'role', 'created_at')
    list_filter = ('role',)
    search_fields = ('user__username', 'user__email')
    ordering = ('-created_at',)


@admin.register(AccessLog)
class AccessLogAdmin(admin.ModelAdmin):
    list_display = ('user', 'patient', 'action', 'timestamp', 'ip_address')
    list_filter = ('action', 'timestamp')
    search_fields = ('user__username', 'patient__nume', 'patient__prenume', 'details')
    ordering = ('-timestamp',)
    readonly_fields = ('user', 'patient', 'action', 'timestamp', 'ip_address', 'details')
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ('CNP', 'nume', 'prenume', 'data_nasterii', 'telefon', 'email', 'oras', 'judet', 'created_at')
    list_filter = ('judet', 'nationalitate', 'data_nasterii')
    search_fields = ('CNP', 'nume', 'prenume', 'telefon', 'email')
    ordering = ('-created_at',)
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        ('Informatii Personale', {
            'fields': ('CNP', 'nume', 'prenume', 'data_nasterii', 'nationalitate')
        }),
        ('Document Identitate', {
            'fields': ('serie_ci', 'numar_ci')
        }),
        ('Contact', {
            'fields': ('telefon', 'email')
        }),
        ('Adresa', {
            'fields': ('oras', 'judet')
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(Appointment)
class AppointmentAdmin(admin.ModelAdmin):
    list_display = ('id', 'patient', 'doctor', 'appointment_date', 'appointment_time', 'status', 'notification_sent', 'created_by')
    list_filter = ('status', 'appointment_date', 'doctor', 'notification_sent')
    search_fields = ('patient__nume', 'patient__prenume', 'doctor__username', 'reason')
    ordering = ('-appointment_date', '-appointment_time')
    readonly_fields = ('created_by', 'created_at', 'updated_at', 'notification_sent')
    
    fieldsets = (
        ('Appointment Details', {
            'fields': ('patient', 'doctor', 'appointment_date', 'appointment_time', 'duration_minutes')
        }),
        ('Status & Notes', {
            'fields': ('status', 'reason', 'notes')
        }),
        ('Notifications', {
            'fields': ('notification_sent', 'notification_read')
        }),
        ('Metadata', {
            'fields': ('created_by', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def save_model(self, request, obj, form, change):
        if not change:  # If creating new appointment
            obj.created_by = request.user
        super().save_model(request, obj, form, change)
