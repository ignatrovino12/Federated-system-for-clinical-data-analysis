from django import forms
from django.core.validators import RegexValidator
from .models import Patient
from django.utils.translation import gettext_lazy as _


class PatientForm(forms.ModelForm):

    # Validators for identity documents
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
        label=_("CNP"),
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-primary transition-colors',
            'placeholder': '1234567890123',
            'maxlength': '13'
        })
    )
    
    serie_ci = forms.CharField(
        validators=[serie_ci_validator],
        max_length=2,
        label=_("CI Series"),
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
        label=_("CI Number"),
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
                'placeholder': _('Last Name')
            }),
            'prenume': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-primary transition-colors',
                'placeholder': _('First Name')
            }),
            'data_nasterii': forms.DateInput(attrs={
                'class': 'w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-primary transition-colors',
                'type': 'date'
            }),
            'nationalitate': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-primary transition-colors',
                'placeholder': _('Nationality')
            }),
            'oras': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-primary transition-colors',
                'placeholder': _('City')
            }),
            'judet': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-primary transition-colors',
                'placeholder': _('County')
            }),
        }
        labels = {
            'nume': _('Last Name'),
            'prenume': _('First Name'),
            'data_nasterii': _('Date of Birth'),
            'nationalitate': _('Nationality'),
            'CNP': _('CNP'),
            'serie_ci': _('CI Series'),
            'numar_ci': _('CI Number'),
            'telefon': _('Phone Number'),
            'email': _('Email Address'),
            'oras': _('City'),
            'judet': _('County')
        }
