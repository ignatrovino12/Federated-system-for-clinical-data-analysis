from django.contrib import admin
from .models import Patient, UserProfile, AccessLog


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

