from django.contrib import admin
from .models import Patient


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

