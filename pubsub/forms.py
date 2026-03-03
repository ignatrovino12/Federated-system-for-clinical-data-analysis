from django import forms
from django.core.validators import RegexValidator
from .models import Patient


class PatientForm(forms.ModelForm):
    """Form for adding/editing patients"""
    
    # Validators for Romanian identity documents
    cnp_validator = RegexValidator(
        regex=r'^\d{13}$',
        message='CNP-ul trebuie să aibă exact 13 cifre'
    )
    
    serie_ci_validator = RegexValidator(
        regex=r'^[A-Z]{2}$',
        message='Serie CI trebuie să aibă exact 2 litere majuscule'
    )
    
    numar_ci_validator = RegexValidator(
        regex=r'^\d{6}$',
        message='Număr CI trebuie să aibă exact 6 cifre'
    )
    
    phone_validator = RegexValidator(
        regex=r'^(\+?[1-9]\d{1,14}|0\d{9,14})$',
        message='Introduceți un număr de telefon valid (ex: +40712345678 sau 0712345678)'
    )
    
    email_validator = RegexValidator(
        regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        message='Introduceți o adresă de email validă'
    )
    
    CNP = forms.CharField(
        validators=[cnp_validator],
        max_length=13,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-primary transition-colors',
            'placeholder': '1234567890123',
            'maxlength': '13'
        })
    )
    
    serie_ci = forms.CharField(
        validators=[serie_ci_validator],
        max_length=2,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-primary transition-colors uppercase',
            'placeholder': 'AB',
            'maxlength': '2',
            'style': 'text-transform: uppercase;'
        })
    )
    
    numar_ci = forms.CharField(
        validators=[numar_ci_validator],
        max_length=6,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-primary transition-colors',
            'placeholder': '123456',
            'maxlength': '6'
        })
    )
    
    telefon = forms.CharField(
        required=False,
        validators=[phone_validator],
        max_length=20,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-primary transition-colors',
            'placeholder': '+40712345678',
            'type': 'tel'
        })
    )
    
    email = forms.EmailField(
        required=False,
        validators=[email_validator],
        widget=forms.EmailInput(attrs={
            'class': 'w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-primary transition-colors',
            'placeholder': 'alexandrescu.ion@example.com'
        })
    )
    
    class Meta:
        model = Patient
        fields = [
            'nume', 'prenume', 'data_nasterii', 'nationalitate',
            'CNP', 'serie_ci', 'numar_ci',
            'telefon', 'email', 'oras', 'judet'
        ]
        widgets = {
            'nume': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-primary transition-colors',
                'placeholder': 'Nume'
            }),
            'prenume': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-primary transition-colors',
                'placeholder': 'Prenume'
            }),
            'data_nasterii': forms.DateInput(attrs={
                'class': 'w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-primary transition-colors',
                'type': 'date'
            }),
            'nationalitate': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-primary transition-colors',
                'placeholder': 'Română'
            }),
            'oras': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-primary transition-colors',
                'placeholder': 'Iași'
            }),
            'judet': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-primary transition-colors',
                'placeholder': 'Iași'
            }),
        }
        labels = {
            'nume': 'Nume',
            'prenume': 'Prenume',
            'data_nasterii': 'Data Nașterii',
            'nationalitate': 'Naționalitate',
            'CNP': 'CNP (Cod Numeric Personal)',
            'serie_ci': 'Serie CI',
            'numar_ci': 'Număr CI',
            'telefon': 'Telefon',
            'email': 'Email',
            'oras': 'Oraș',
            'judet': 'Județ'
        }
