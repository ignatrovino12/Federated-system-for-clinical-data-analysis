from django import forms
from django.utils.translation import gettext_lazy as _

from pubsub.models import Patient

from .models import PatientClinicalRecord


YES_NO_CHOICES = [
    ("", _("Select one")),
    ("1", _("Yes")),
    ("0", _("No")),
]

GENDER_CHOICES = [
    ("", _("Select one")),
    ("0", _("Female")),
    ("1", _("Male")),
]

SMOKING_CHOICES = [
    ("", _("Select one")),
    ("0", _("Never")),
    ("1", _("No info")),
    ("2", _("Former")),
    ("3", _("Current")),
]

GENERAL_HEALTH_CHOICES = [
    ("", _("Select one")),
    ("1", _("Excellent")),
    ("2", _("Very good")),
    ("3", _("Good")),
    ("4", _("Fair")),
    ("5", _("Poor")),
]

EDUCATION_CHOICES = [
    ("", _("Select one")),
    ("1", _("Less than 9th grade")),
    ("2", _("High school graduate")),
    ("3", _("Attended college (no degree)")),
    ("4", _("College graduate")),
    ("5", _("Post-graduate")),
    ("6", _("Other / Unknown")),
]

INCOME_CHOICES = [
    ("", _("Select one")),
    ("1", _("Less than $10,000")),
    ("2", _("$10,000 to $14,999")),
    ("3", _("$15,000 to $19,999")),
    ("4", _("$20,000 to $24,999")),
    ("5", _("$25,000 to $34,999")),
    ("6", _("$35,000 to $49,999")),
    ("7", _("$50,000 to $74,999")),
    ("8", _("$75,000 or more")),
]


class PatientClinicalRecordForm(forms.ModelForm):
    patient_birth_date = forms.DateField(
        required=True,
        widget=forms.DateInput(
            attrs={
                "class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg",
                "type": "date",
            }
        ),
        label=_("Date of birth"),
    )

    class Meta:
        model = PatientClinicalRecord
        fields = [
            "patient_birth_date",
            "gender",
            "age_years",
            "hypertension",
            "heart_disease",
            "smoking_history",
            "bmi",
            "hba1c_level",
            "blood_glucose_level",
            "high_bp",
            "high_chol",
            "chol_check",
            "smoker",
            "stroke",
            "phys_activity",
            "fruits",
            "veggies",
            "heavy_alcohol_consumption",
            "any_healthcare",
            "no_docbc_cost",
            "general_health",
            "mental_health_days",
            "physical_health_days",
            "diff_walk",
            "education",
            "income",
            "diabetes_status",
            "data_consent_for_training",
            "notes",
        ]
        widgets = {
            "gender": forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}),
            "age_years": forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg bg-gray-100", "readonly": "readonly"}),
            "hypertension": forms.CheckboxInput(attrs={"class": "h-4 w-4 text-primary rounded"}),
            "heart_disease": forms.CheckboxInput(attrs={"class": "h-4 w-4 text-primary rounded"}),
            "smoking_history": forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}),
            "bmi": forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "step": "0.01"}),
            "hba1c_level": forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "step": "0.01"}),
            "blood_glucose_level": forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}),
            "high_bp": forms.CheckboxInput(attrs={"class": "h-4 w-4 text-primary rounded"}),
            "high_chol": forms.CheckboxInput(attrs={"class": "h-4 w-4 text-primary rounded"}),
            "chol_check": forms.CheckboxInput(attrs={"class": "h-4 w-4 text-primary rounded"}),
            "smoker": forms.CheckboxInput(attrs={"class": "h-4 w-4 text-primary rounded"}),
            "stroke": forms.CheckboxInput(attrs={"class": "h-4 w-4 text-primary rounded"}),
            "phys_activity": forms.CheckboxInput(attrs={"class": "h-4 w-4 text-primary rounded"}),
            "fruits": forms.CheckboxInput(attrs={"class": "h-4 w-4 text-primary rounded"}),
            "veggies": forms.CheckboxInput(attrs={"class": "h-4 w-4 text-primary rounded"}),
            "heavy_alcohol_consumption": forms.CheckboxInput(attrs={"class": "h-4 w-4 text-primary rounded"}),
            "any_healthcare": forms.CheckboxInput(attrs={"class": "h-4 w-4 text-primary rounded"}),
            "no_docbc_cost": forms.CheckboxInput(attrs={"class": "h-4 w-4 text-primary rounded"}),
            "general_health": forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}),
            "mental_health_days": forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "min": 0, "max": 30}),
            "physical_health_days": forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "min": 0, "max": 30}),
            "diff_walk": forms.CheckboxInput(attrs={"class": "h-4 w-4 text-primary rounded"}),
            "education": forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}),
            "income": forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}),
            "diabetes_status": forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}),
            "data_consent_for_training": forms.CheckboxInput(attrs={"class": "h-4 w-4 text-primary rounded"}),
            "notes": forms.Textarea(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "rows": 4}),
        }
        labels = {
            "patient_birth_date": _("Date of birth"),
            "gender": _("Sex"),
            "age_years": _("Age (auto-calculated)"),
            "hypertension": _("High blood pressure"),
            "heart_disease": _("Heart disease"),
            "smoking_history": _("Smoking history"),
            "bmi": _("Body mass index (BMI)"),
            "hba1c_level": _("HbA1c level"),
            "blood_glucose_level": _("Blood glucose level"),
            "high_bp": _("High blood pressure"),
            "high_chol": _("High cholesterol"),
            "chol_check": _("Cholesterol screening completed"),
            "smoker": _("Current smoker"),
            "stroke": _("History of stroke"),
            "phys_activity": _("Regular physical activity"),
            "fruits": _("Eats fruit regularly"),
            "veggies": _("Eats vegetables regularly"),
            "heavy_alcohol_consumption": _("Heavy alcohol consumption"),
            "any_healthcare": _("Has healthcare coverage"),
            "no_docbc_cost": _("Did not see a doctor because of cost"),
            "general_health": _("Overall health"),
            "mental_health_days": _("Mental health days in the past 30 days"),
            "physical_health_days": _("Physical health days in the past 30 days"),
            "diff_walk": _("Difficulty walking"),
            "education": _("Education level"),
            "income": _("Income level"),
            "diabetes_status": _("Diabetes status"),
            "data_consent_for_training": _("Consent to use data for federated training"),
            "notes": _("Clinical notes"),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        patient = getattr(self.instance, "patient", None)
        if patient and patient.data_nasterii:
            self.fields["patient_birth_date"].initial = patient.data_nasterii
            self.fields["age_years"].initial = patient.get_age()
        elif self.instance and self.instance.age_years is not None:
            self.fields["age_years"].initial = self.instance.age_years

        self.fields["age_years"].required = False

    def save(self, commit=True):
        patient_birth_date = self.cleaned_data.get("patient_birth_date")
        record = super().save(commit=False)

        if record.patient_id and patient_birth_date:
            patient = record.patient
            if patient.data_nasterii != patient_birth_date:
                patient.data_nasterii = patient_birth_date
                patient.save(update_fields=["data_nasterii", "updated_at"])
            record.age_years = patient.get_age()

        if commit:
            record.save()

        return record


class AlexManualAnalysisForm(forms.Form):
    gender = forms.TypedChoiceField(
        choices=GENDER_CHOICES,
        coerce=int,
        widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}),
        label=_("Sex"),
    )
    age_years = forms.IntegerField(
        min_value=0,
        max_value=120,
        widget=forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "min": 0, "max": 120}),
        label=_("Age (years)"),
    )
    high_bp = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label=_("High blood pressure"))
    high_chol = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label=_("High cholesterol"))
    chol_check = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label=_("Cholesterol screening completed"))
    bmi = forms.FloatField(
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "step": "0.01"}),
        label=_("Body mass index (BMI)"),
    )
    smoker = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label=_("Current smoker"))
    stroke = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label=_("History of stroke"))
    heart_disease = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label=_("Heart disease or heart attack"))
    phys_activity = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label=_("Regular physical activity"))
    fruits = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label=_("Eats fruit regularly"))
    veggies = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label=_("Eats vegetables regularly"))
    heavy_alcohol_consumption = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label=_("Heavy alcohol consumption"))
    any_healthcare = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label=_("Has healthcare coverage"))
    no_docbc_cost = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label=_("Did not see a doctor because of cost"))
    general_health = forms.TypedChoiceField(choices=GENERAL_HEALTH_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label=_("Overall health"))
    mental_health_days = forms.IntegerField(min_value=0, max_value=30, widget=forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "min": 0, "max": 30}), label=_("Mental health days in the past 30 days"))
    physical_health_days = forms.IntegerField(min_value=0, max_value=30, widget=forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "min": 0, "max": 30}), label=_("Physical health days in the past 30 days"))
    diff_walk = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label=_("Difficulty walking"))
    education = forms.TypedChoiceField(choices=EDUCATION_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label=_("Education level"))
    income = forms.TypedChoiceField(choices=INCOME_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label=_("Income level"))

    def to_feature_payload(self):
        cleaned = self.cleaned_data
        age_bucket = PatientClinicalRecord(age_years=cleaned["age_years"])._alex_age_category()
        return {
            "HighBP": cleaned["high_bp"],
            "HighChol": cleaned["high_chol"],
            "CholCheck": cleaned["chol_check"],
            "BMI": cleaned["bmi"],
            "Smoker": cleaned["smoker"],
            "Stroke": cleaned["stroke"],
            "HeartDiseaseorAttack": cleaned["heart_disease"],
            "PhysActivity": cleaned["phys_activity"],
            "Fruits": cleaned["fruits"],
            "Veggies": cleaned["veggies"],
            "HvyAlcoholConsump": cleaned["heavy_alcohol_consumption"],
            "AnyHealthcare": cleaned["any_healthcare"],
            "NoDocbcCost": cleaned["no_docbc_cost"],
            "GenHlth": cleaned["general_health"],
            "MentHlth": cleaned["mental_health_days"],
            "PhysHlth": cleaned["physical_health_days"],
            "DiffWalk": cleaned["diff_walk"],
            "Sex": cleaned["gender"],
            "Age": age_bucket,
            "Education": cleaned["education"],
            "Income": cleaned["income"],
        }


class MustafaManualAnalysisForm(forms.Form):
    gender = forms.TypedChoiceField(
        choices=GENDER_CHOICES,
        coerce=int,
        widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}),
        label=_("Sex"),
    )
    age_years = forms.IntegerField(
        min_value=0,
        max_value=120,
        widget=forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "min": 0, "max": 120}),
        label=_("Age (years)"),
    )
    hypertension = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label=_("High blood pressure"))
    heart_disease = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label=_("Heart disease"))
    smoking_history = forms.TypedChoiceField(
        choices=SMOKING_CHOICES,
        coerce=int,
        widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}),
        label=_("Smoking history"),
    )
    bmi = forms.FloatField(
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "step": "0.01"}),
        label=_("Body mass index (BMI)"),
    )
    hba1c_level = forms.FloatField(
        min_value=0,
        max_value=20,
        widget=forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "step": "0.01"}),
        label=_("HbA1c level"),
    )
    blood_glucose_level = forms.IntegerField(
        min_value=0,
        max_value=1000,
        widget=forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}),
        label=_("Blood glucose level"),
    )

    def to_feature_payload(self):
        cleaned = self.cleaned_data
        return {
            "gender": cleaned["gender"],
            "age": cleaned["age_years"],
            "hypertension": cleaned["hypertension"],
            "heart_disease": cleaned["heart_disease"],
            "smoking_history": cleaned["smoking_history"],
            "bmi": cleaned["bmi"],
            "HbA1c_level": cleaned["hba1c_level"],
            "blood_glucose_level": cleaned["blood_glucose_level"],
        }


class ClinicalAnalysisSelectionForm(forms.Form):
    MODEL_CHOICES = [
        ("", _("Select one")),
        ("alex", _("Lifestyle-Based Diabetes Risk Model")),
        ("mustafa", _("Clinical Diabetes Risk Model")),
    ]

    model = forms.ChoiceField(
        choices=MODEL_CHOICES,
        widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}),
        label=_("Analysis model"),
    )
    patient = forms.ModelChoiceField(
        queryset=Patient.objects.none(),
        widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}),
        label=_("Patient"),
    )

    def __init__(self, *args, patient_queryset=None, **kwargs):
        super().__init__(*args, **kwargs)
        if patient_queryset is not None:
            self.fields["patient"].queryset = patient_queryset