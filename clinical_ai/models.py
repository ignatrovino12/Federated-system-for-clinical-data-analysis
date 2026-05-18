from decimal import Decimal

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone

from pubsub.models import Patient


class PatientClinicalRecord(models.Model):
	"""Clinical data used by the Alex5050 and Mustafa ML models."""

	class DiabetesStatus(models.TextChoices):
		HAS = "has", "Confirmed diabetes"
		HAS_NOT = "has_not", "Confirmed no diabetes"
		NOT_CONFIRMED = "not_confirmed", "Not confirmed"

	class Gender(models.IntegerChoices):
		FEMALE = 0, "Female"
		MALE = 1, "Male"

	class SmokingHistory(models.IntegerChoices):
		NEVER = 0, "Never"
		NO_INFO = 1, "No info"
		FORMER = 2, "Former"
		CURRENT = 3, "Current"

	class GeneralHealth(models.IntegerChoices):
		EXCELLENT = 1, "Excellent"
		VERY_GOOD = 2, "Very good"
		GOOD = 3, "Good"
		FAIR = 4, "Fair"
		POOR = 5, "Poor"

	class Education(models.IntegerChoices):
		LESS_THAN_9TH = 1, "Less than 9th grade"
		HIGH_SCHOOL = 2, "High school graduate"
		SOME_COLLEGE = 3, "Attended college (no degree)"
		COLLEGE_GRAD = 4, "College graduate"
		POST_GRAD = 5, "Post-graduate"
		OTHER = 6, "Other / Unknown"

	class Income(models.IntegerChoices):
		LESS_THAN_10K = 1, "Less than $10,000"
		TEN_TO_15K = 2, "$10,000 to $14,999"
		FIFTEEN_TO_20K = 3, "$15,000 to $19,999"
		TWENTY_TO_25K = 4, "$20,000 to $24,999"
		TWENTYFIVE_TO_35K = 5, "$25,000 to $34,999"
		THIRTYFIVE_TO_50K = 6, "$35,000 to $49,999"
		FIFTY_TO_75K = 7, "$50,000 to $74,999"
		SEVENTYFIVE_PLUS = 8, "$75,000 or more"

	patient = models.OneToOneField(
		Patient,
		on_delete=models.CASCADE,
		related_name="clinical_record",
	)

	# Mustafa model factors
	gender = models.PositiveSmallIntegerField(
		choices=Gender.choices,
		null=True,
		blank=True,
		help_text="Gender encoded as 0 = Female, 1 = Male",
	)
	age_years = models.PositiveSmallIntegerField(
		null=True,
		blank=True,
		validators=[MinValueValidator(0), MaxValueValidator(120)],
		help_text="Age in years",
	)
	hypertension = models.BooleanField(null=True, blank=True)
	heart_disease = models.BooleanField(null=True, blank=True)
	smoking_history = models.PositiveSmallIntegerField(
		choices=SmokingHistory.choices,
		null=True,
		blank=True,
		help_text="Ordinal smoking history used by the Mustafa model",
	)
	bmi = models.DecimalField(
		max_digits=5,
		decimal_places=2,
		null=True,
		blank=True,
		validators=[MinValueValidator(Decimal("0")), MaxValueValidator(Decimal("100"))],
	)
	hba1c_level = models.DecimalField(
		max_digits=4,
		decimal_places=2,
		null=True,
		blank=True,
		validators=[MinValueValidator(Decimal("0")), MaxValueValidator(Decimal("20"))],
		help_text="HbA1c level",
	)
	blood_glucose_level = models.PositiveIntegerField(
		null=True,
		blank=True,
		validators=[MinValueValidator(0), MaxValueValidator(1000)],
	)

	# Alex5050 model factors
	high_bp = models.BooleanField(null=True, blank=True)
	high_chol = models.BooleanField(null=True, blank=True)
	chol_check = models.BooleanField(null=True, blank=True)
	smoker = models.BooleanField(null=True, blank=True)
	stroke = models.BooleanField(null=True, blank=True)
	phys_activity = models.BooleanField(null=True, blank=True)
	fruits = models.BooleanField(null=True, blank=True)
	veggies = models.BooleanField(null=True, blank=True)
	heavy_alcohol_consumption = models.BooleanField(null=True, blank=True)
	any_healthcare = models.BooleanField(null=True, blank=True)
	no_docbc_cost = models.BooleanField(null=True, blank=True)
	general_health = models.PositiveSmallIntegerField(
		choices=GeneralHealth.choices,
		null=True,
		blank=True,
	)
	mental_health_days = models.PositiveSmallIntegerField(
		null=True,
		blank=True,
		validators=[MinValueValidator(0), MaxValueValidator(30)],
	)
	physical_health_days = models.PositiveSmallIntegerField(
		null=True,
		blank=True,
		validators=[MinValueValidator(0), MaxValueValidator(30)],
	)
	diff_walk = models.BooleanField(null=True, blank=True)
	education = models.PositiveSmallIntegerField(
		choices=Education.choices,
		null=True,
		blank=True,
	)
	income = models.PositiveSmallIntegerField(
		choices=Income.choices,
		null=True,
		blank=True,
	)

	diabetes_status = models.CharField(
		max_length=20,
		choices=DiabetesStatus.choices,
		default=DiabetesStatus.NOT_CONFIRMED,
		help_text="Confirmed diabetes diagnosis status used as the federated label.",
	)
	data_consent_for_training = models.BooleanField(
		default=False,
		help_text="Whether the patient consented to use this data in federated training.",
	)

	notes = models.TextField(blank=True)
	recorded_at = models.DateTimeField(default=timezone.now)
	updated_at = models.DateTimeField(auto_now=True)

	class Meta:
		db_table = "patient_clinical_records"
		ordering = ["-recorded_at", "-id"]
		verbose_name = "Patient Clinical Record"
		verbose_name_plural = "Patient Clinical Records"
		indexes = [
			models.Index(fields=["patient", "recorded_at"]),
		]

	def __str__(self):
		return f"Clinical record for {self.patient.get_full_name()} at {self.recorded_at:%Y-%m-%d %H:%M}"

	def _binary_value(self, value):
		if value is None:
			return None
		return int(bool(value))

	def _resolved_age_years(self):
		if self.patient_id:
			return self.patient.get_age()
		if self.age_years is not None:
			return self.age_years
		return None

	def _alex_age_category(self):
		age_years = self._resolved_age_years()
		if age_years is None:
			return None

		bands = [
			(18, 24),
			(25, 29),
			(30, 34),
			(35, 39),
			(40, 44),
			(45, 49),
			(50, 54),
			(55, 59),
			(60, 64),
			(65, 69),
			(70, 74),
			(75, 79),
			(80, 200),
		]

		for index, (lower, upper) in enumerate(bands, start=1):
			if lower <= age_years <= upper:
				return index

		return 1 if age_years < 18 else 13

	def _smoker_flag(self):
		if self.smoking_history is None:
			return None
		return 1 if self.smoking_history in {self.SmokingHistory.FORMER, self.SmokingHistory.CURRENT} else 0

	def alex5050_features(self):
		"""Return a feature dictionary in the exact Alex5050 column order."""
		return {
			"HighBP": self._binary_value(self.high_bp if self.high_bp is not None else self.hypertension),
			"HighChol": self._binary_value(self.high_chol),
			"CholCheck": self._binary_value(self.chol_check),
			"BMI": float(self.bmi) if self.bmi is not None else None,
			"Smoker": self._binary_value(self.smoker if self.smoker is not None else self._smoker_flag()),
			"Stroke": self._binary_value(self.stroke),
			"HeartDiseaseorAttack": self._binary_value(self.heart_disease),
			"PhysActivity": self._binary_value(self.phys_activity),
			"Fruits": self._binary_value(self.fruits),
			"Veggies": self._binary_value(self.veggies),
			"HvyAlcoholConsump": self._binary_value(self.heavy_alcohol_consumption),
			"AnyHealthcare": self._binary_value(self.any_healthcare),
			"NoDocbcCost": self._binary_value(self.no_docbc_cost),
			"GenHlth": self.general_health,
			"MentHlth": self.mental_health_days,
			"PhysHlth": self.physical_health_days,
			"DiffWalk": self._binary_value(self.diff_walk),
			"Sex": self.gender,
			"Age": self._alex_age_category(),
			"Education": self.education,
			"Income": self.income,
		}

	def mustafa_features(self):
		"""Return a feature dictionary in the exact Mustafa column order."""
		return {
			"gender": self.gender,
			"age": self._resolved_age_years(),
			"hypertension": self._binary_value(self.hypertension if self.hypertension is not None else self.high_bp),
			"heart_disease": self._binary_value(self.heart_disease),
			"smoking_history": self.smoking_history,
			"bmi": float(self.bmi) if self.bmi is not None else None,
			"HbA1c_level": float(self.hba1c_level) if self.hba1c_level is not None else None,
			"blood_glucose_level": self.blood_glucose_level,
		}
