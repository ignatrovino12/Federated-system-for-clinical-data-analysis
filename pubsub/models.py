from django.db import models


class Patient(models.Model):
    """Model for patient data in medical clinics"""
    
    # Primary key 
    id = models.AutoField(primary_key=True)
    
    # Personal identification
    CNP = models.CharField(max_length=13, unique=True, help_text="Cod Numeric Personal")
    nume = models.CharField(max_length=100, help_text="Last name")
    prenume = models.CharField(max_length=100, help_text="First name")
    data_nasterii = models.DateField(help_text="Date of birth")
    
    # Identity document
    serie_ci = models.CharField(max_length=10, help_text="ID card series")
    numar_ci = models.CharField(max_length=20, help_text="ID card number")
    nationalitate = models.CharField(max_length=50, default="Română", help_text="Nationality")
    
    # Contact information
    telefon = models.CharField(max_length=20, blank=True, null=True, help_text="Phone number")
    email = models.EmailField(blank=True, null=True, help_text="Email address")
    
    # Address
    oras = models.CharField(max_length=100, help_text="City")
    judet = models.CharField(max_length=100, help_text="County")
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'patients'
        ordering = ['-created_at']
        verbose_name = 'Patient'
        verbose_name_plural = 'Patients'
        indexes = [
            models.Index(fields=['CNP']),
            models.Index(fields=['nume', 'prenume']),
        ]
    
    def __str__(self):
        return f"{self.nume} {self.prenume} (CNP: {self.CNP})"
    
    def get_full_name(self):
        return f"{self.prenume} {self.nume}"
