from django import forms

from pubsub.models import Patient

from .models import PatientClinicalRecord


YES_NO_CHOICES = [
    ("", "Select one"),
    ("1", "Yes"),
    ("0", "No"),
]

GENDER_CHOICES = [
    ("", "Select one"),
    ("0", "Female"),
    ("1", "Male"),
]

SMOKING_CHOICES = [
    ("", "Select one"),
    ("0", "Never"),
    ("1", "No info"),
    ("2", "Former"),
    ("3", "Current"),
]

GENERAL_HEALTH_CHOICES = [
    ("", "Select one"),
    ("1", "Excellent"),
    ("2", "Very good"),
    ("3", "Good"),
    ("4", "Fair"),
    ("5", "Poor"),
]

EDUCATION_CHOICES = [
    ("", "Select one"),
    ("1", "Less than 9th grade"),
    ("2", "High school graduate"),
    ("3", "Attended college (no degree)"),
    ("4", "College graduate"),
    ("5", "Post-graduate"),
    ("6", "Other / Unknown"),
]

INCOME_CHOICES = [
    ("", "Select one"),
    ("1", "Less than $10,000"),
    ("2", "$10,000 to $14,999"),
    ("3", "$15,000 to $19,999"),
    ("4", "$20,000 to $24,999"),
    ("5", "$25,000 to $34,999"),
    ("6", "$35,000 to $49,999"),
    ("7", "$50,000 to $74,999"),
    ("8", "$75,000 or more"),
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
        label="Date of birth",
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
            "patient_birth_date": "Date of birth",
            "gender": "Sex",
            "age_years": "Age (auto-calculated)",
            "hypertension": "High blood pressure",
            "heart_disease": "Heart disease",
            "smoking_history": "Smoking history",
            "bmi": "Body mass index (BMI)",
            "hba1c_level": "HbA1c level",
            "blood_glucose_level": "Blood glucose level",
            "high_bp": "High blood pressure",
            "high_chol": "High cholesterol",
            "chol_check": "Cholesterol screening completed",
            "smoker": "Current smoker",
            "stroke": "History of stroke",
            "phys_activity": "Regular physical activity",
            "fruits": "Eats fruit regularly",
            "veggies": "Eats vegetables regularly",
            "heavy_alcohol_consumption": "Heavy alcohol consumption",
            "any_healthcare": "Has healthcare coverage",
            "no_docbc_cost": "Did not see a doctor because of cost",
            "general_health": "Overall health",
            "mental_health_days": "Mental health days in the past 30 days",
            "physical_health_days": "Physical health days in the past 30 days",
            "diff_walk": "Difficulty walking",
            "education": "Education level",
            "income": "Income level",
            "diabetes_status": "Diabetes status",
            "data_consent_for_training": "Consent to use data for federated training",
            "notes": "Clinical notes",
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
        label="Sex",
    )
    age_years = forms.IntegerField(
        min_value=0,
        max_value=120,
        widget=forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "min": 0, "max": 120}),
        label="Age (years)",
    )
    high_bp = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label="High blood pressure")
    high_chol = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label="High cholesterol")
    chol_check = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label="Cholesterol screening completed")
    bmi = forms.FloatField(
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "step": "0.01"}),
        label="Body mass index (BMI)",
    )
    smoker = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label="Current smoker")
    stroke = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label="History of stroke")
    heart_disease = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label="Heart disease or heart attack")
    phys_activity = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label="Regular physical activity")
    fruits = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label="Eats fruit regularly")
    veggies = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label="Eats vegetables regularly")
    heavy_alcohol_consumption = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label="Heavy alcohol consumption")
    any_healthcare = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label="Has healthcare coverage")
    no_docbc_cost = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label="Did not see a doctor because of cost")
    general_health = forms.TypedChoiceField(choices=GENERAL_HEALTH_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label="Overall health")
    mental_health_days = forms.IntegerField(min_value=0, max_value=30, widget=forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "min": 0, "max": 30}), label="Mental health days in the past 30 days")
    physical_health_days = forms.IntegerField(min_value=0, max_value=30, widget=forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "min": 0, "max": 30}), label="Physical health days in the past 30 days")
    diff_walk = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label="Difficulty walking")
    education = forms.TypedChoiceField(choices=EDUCATION_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label="Education level")
    income = forms.TypedChoiceField(choices=INCOME_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label="Income level")

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
        label="Sex",
    )
    age_years = forms.IntegerField(
        min_value=0,
        max_value=120,
        widget=forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "min": 0, "max": 120}),
        label="Age (years)",
    )
    hypertension = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label="High blood pressure")
    heart_disease = forms.TypedChoiceField(choices=YES_NO_CHOICES, coerce=int, widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}), label="Heart disease")
    smoking_history = forms.TypedChoiceField(
        choices=SMOKING_CHOICES,
        coerce=int,
        widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}),
        label="Smoking history",
    )
    bmi = forms.FloatField(
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "step": "0.01"}),
        label="Body mass index (BMI)",
    )
    hba1c_level = forms.FloatField(
        min_value=0,
        max_value=20,
        widget=forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg", "step": "0.01"}),
        label="HbA1c level",
    )
    blood_glucose_level = forms.IntegerField(
        min_value=0,
        max_value=1000,
        widget=forms.NumberInput(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}),
        label="Blood glucose level",
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
        ("", "Select one"),
        ("alex", "Alex 5050 model"),
        ("mustafa", "Mustafa model"),
    ]

    model = forms.ChoiceField(
        choices=MODEL_CHOICES,
        widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}),
        label="Analysis model",
    )
    patient = forms.ModelChoiceField(
        queryset=Patient.objects.none(),
        widget=forms.Select(attrs={"class": "w-full px-4 py-3 border-2 border-gray-300 rounded-lg"}),
        label="Patient",
    )

    def __init__(self, *args, patient_queryset=None, **kwargs):
        super().__init__(*args, **kwargs)
        if patient_queryset is not None:
            self.fields["patient"].queryset = patient_queryset