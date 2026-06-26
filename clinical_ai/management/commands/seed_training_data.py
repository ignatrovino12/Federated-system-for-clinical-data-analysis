import random
from datetime import date, timedelta
from decimal import Decimal

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from clinical_ai.models import PatientClinicalRecord
from pubsub.models import Patient


FIRST_NAMES = [
	"Andrei", "Maria", "Ioana", "Alexandru", "Elena", "Mihai", "Ana", "David",
	"Gabriela", "George", "Cristina", "Rares", "Teodora", "Vlad", "Diana", "Paul",
	"Bianca", "Sorin", "Andreea", "Cătălin",
]

LAST_NAMES = [
	"Popescu", "Ionescu", "Stan", "Dumitrescu", "Nistor", "Marin", "Dima", "Munteanu",
	"Ilie", "Tudor", "Gheorghe", "Pavel", "Radu", "Lazar", "Stoica", "Enache",
	"Petrescu", "Neagu", "Matei", "Preda",
]

ORAS = ["București", "Cluj-Napoca", "Iași", "Timișoara", "Brașov", "Constanța", "Craiova", "Sibiu"]
JUDET = ["București", "Cluj", "Iași", "Timiș", "Brașov", "Constanța", "Dolj", "Sibiu"]


def _rand_bool(rng: random.Random, probability: float) -> bool:
	return rng.random() < probability


def _make_cnp(index: int) -> str:
	base = 4000000000000 + index
	return f"{base:013d}"


def _make_ci_number(index: int) -> str:
	return f"{index % 1000000:06d}"


def _age_to_birthdate(age_years: int, rng: random.Random) -> date:
	today = timezone.localdate()
	day_offset = rng.randint(0, 364)
	return today.replace(year=today.year - age_years) - timedelta(days=day_offset)


def _sigmoid(value: float) -> float:
	return 1.0 / (1.0 + pow(2.718281828459045, -value))


def _sample_diabetes_status(rng: random.Random, risk_score: int) -> str:
	# Convert a risk score into a noisy label so synthetic data is not perfectly separable
	# Add a little uncertainty so the same risk profile is not always labeled identically
	noisy_score = risk_score + rng.gauss(0.0, 1.35)

	if noisy_score <= 2:
		weights = [0.86, 0.10, 0.04]
	elif noisy_score <= 4:
		weights = [0.64, 0.22, 0.14]
	elif noisy_score <= 6:
		weights = [0.28, 0.42, 0.30]
	elif noisy_score <= 8:
		weights = [0.16, 0.28, 0.56]
	else:
		weights = [0.05, 0.12, 0.83]

	return rng.choices(
		[
			PatientClinicalRecord.DiabetesStatus.HAS_NOT,
			PatientClinicalRecord.DiabetesStatus.NOT_CONFIRMED,
			PatientClinicalRecord.DiabetesStatus.HAS,
		],
		weights=weights,
		k=1,
	)[0]


def _build_synthetic_profile(rng: random.Random):
	# Build one coherent synthetic clinical profile for both models
	risk_band = rng.choices(["low", "moderate", "high"], weights=[0.45, 0.35, 0.20], k=1)[0]

	if risk_band == "low":
		age_years = rng.randint(24, 44)
		hypertension = _rand_bool(rng, 0.08)
		heart_disease = _rand_bool(rng, 0.03)
		smoking_history = rng.choices(
			[
				PatientClinicalRecord.SmokingHistory.NEVER,
				PatientClinicalRecord.SmokingHistory.NO_INFO,
				PatientClinicalRecord.SmokingHistory.FORMER,
				PatientClinicalRecord.SmokingHistory.CURRENT,
			],
			weights=[0.72, 0.10, 0.12, 0.06],
			k=1,
		)[0]
		bmi = round(rng.uniform(20.0, 27.4), 2)
		hba1c = round(rng.uniform(4.6, 5.7), 2)
		blood_glucose = rng.randint(78, 128)
		high_bp = hypertension or _rand_bool(rng, 0.04)
		high_chol = _rand_bool(rng, 0.10)
		chol_check = _rand_bool(rng, 0.90)
		stroke = _rand_bool(rng, 0.01)
		phys_activity = _rand_bool(rng, 0.78)
		fruits = _rand_bool(rng, 0.74)
		veggies = _rand_bool(rng, 0.80)
		heavy_alcohol = _rand_bool(rng, 0.05)
		any_healthcare = True
		no_docbc_cost = _rand_bool(rng, 0.08)
		general_health = rng.choices(
			[
				PatientClinicalRecord.GeneralHealth.EXCELLENT,
				PatientClinicalRecord.GeneralHealth.VERY_GOOD,
				PatientClinicalRecord.GeneralHealth.GOOD,
				PatientClinicalRecord.GeneralHealth.FAIR,
			],
			weights=[0.28, 0.34, 0.28, 0.10],
			k=1,
		)[0]
		mental_days = rng.randint(0, 5)
		physical_days = rng.randint(0, 6)
		diff_walk = _rand_bool(rng, 0.08)
		education = rng.choices(
			list(PatientClinicalRecord.Education.values),
			weights=[1, 1, 2, 3, 3, 1],
			k=1,
		)[0]
		income = rng.choices(
			list(PatientClinicalRecord.Income.values),
			weights=[1, 1, 2, 3, 3, 2, 1, 1],
			k=1,
		)[0]
	elif risk_band == "high":
		age_years = rng.randint(55, 82)
		hypertension = _rand_bool(rng, 0.72)
		heart_disease = _rand_bool(rng, 0.18)
		smoking_history = rng.choices(
			[
				PatientClinicalRecord.SmokingHistory.NEVER,
				PatientClinicalRecord.SmokingHistory.NO_INFO,
				PatientClinicalRecord.SmokingHistory.FORMER,
				PatientClinicalRecord.SmokingHistory.CURRENT,
			],
			weights=[0.18, 0.05, 0.30, 0.47],
			k=1,
		)[0]
		bmi = round(rng.uniform(28.0, 43.0), 2)
		hba1c = round(rng.uniform(6.3, 10.8), 2)
		blood_glucose = rng.randint(160, 320)
		high_bp = True if hypertension else _rand_bool(rng, 0.18)
		high_chol = _rand_bool(rng, 0.46)
		chol_check = _rand_bool(rng, 0.88)
		stroke = _rand_bool(rng, 0.07)
		phys_activity = _rand_bool(rng, 0.32)
		fruits = _rand_bool(rng, 0.42)
		veggies = _rand_bool(rng, 0.48)
		heavy_alcohol = _rand_bool(rng, 0.14)
		any_healthcare = True
		no_docbc_cost = _rand_bool(rng, 0.22)
		general_health = rng.choices(
			[
				PatientClinicalRecord.GeneralHealth.GOOD,
				PatientClinicalRecord.GeneralHealth.FAIR,
				PatientClinicalRecord.GeneralHealth.POOR,
			],
			weights=[0.24, 0.44, 0.32],
			k=1,
		)[0]
		mental_days = rng.randint(4, 24)
		physical_days = rng.randint(3, 28)
		diff_walk = _rand_bool(rng, 0.34)
		education = rng.choices(
			list(PatientClinicalRecord.Education.values),
			weights=[4, 3, 2, 1, 1, 1],
			k=1,
		)[0]
		income = rng.choices(
			list(PatientClinicalRecord.Income.values),
			weights=[4, 3, 2, 2, 1, 1, 1, 1],
			k=1,
		)[0]
	else:
		age_years = rng.randint(40, 66)
		hypertension = _rand_bool(rng, 0.30)
		heart_disease = _rand_bool(rng, 0.07)
		smoking_history = rng.choices(
			[
				PatientClinicalRecord.SmokingHistory.NEVER,
				PatientClinicalRecord.SmokingHistory.NO_INFO,
				PatientClinicalRecord.SmokingHistory.FORMER,
				PatientClinicalRecord.SmokingHistory.CURRENT,
			],
			weights=[0.42, 0.10, 0.28, 0.20],
			k=1,
		)[0]
		bmi = round(rng.uniform(24.0, 33.0), 2)
		hba1c = round(rng.uniform(5.4, 6.6), 2)
		blood_glucose = rng.randint(110, 180)
		high_bp = hypertension or _rand_bool(rng, 0.12)
		high_chol = _rand_bool(rng, 0.28)
		chol_check = _rand_bool(rng, 0.84)
		stroke = _rand_bool(rng, 0.03)
		phys_activity = _rand_bool(rng, 0.58)
		fruits = _rand_bool(rng, 0.62)
		veggies = _rand_bool(rng, 0.67)
		heavy_alcohol = _rand_bool(rng, 0.10)
		any_healthcare = True
		no_docbc_cost = _rand_bool(rng, 0.15)
		general_health = rng.choices(
			[
				PatientClinicalRecord.GeneralHealth.VERY_GOOD,
				PatientClinicalRecord.GeneralHealth.GOOD,
				PatientClinicalRecord.GeneralHealth.FAIR,
				PatientClinicalRecord.GeneralHealth.POOR,
			],
			weights=[0.18, 0.34, 0.34, 0.14],
			k=1,
		)[0]
		mental_days = rng.randint(1, 12)
		physical_days = rng.randint(0, 14)
		diff_walk = _rand_bool(rng, 0.18)
		education = rng.choices(
			list(PatientClinicalRecord.Education.values),
			weights=[2, 2, 3, 3, 2, 1],
			k=1,
		)[0]
		income = rng.choices(
			list(PatientClinicalRecord.Income.values),
			weights=[2, 2, 2, 2, 2, 2, 1, 1],
			k=1,
		)[0]

	smoker = smoking_history in {
		PatientClinicalRecord.SmokingHistory.FORMER,
		PatientClinicalRecord.SmokingHistory.CURRENT,
	}

	risk_score = 0
	if risk_band == "high":
		risk_score += 3
	elif risk_band == "moderate":
		risk_score += 1

	if hypertension:
		risk_score += 2
	if heart_disease:
		risk_score += 2
	if smoker:
		risk_score += 1
	if bmi >= 30:
		risk_score += 2
	if hba1c >= 6.5:
		risk_score += 3
	elif hba1c >= 5.8:
		risk_score += 1
	if blood_glucose >= 180:
		risk_score += 3
	elif blood_glucose >= 140:
		risk_score += 1
	if age_years >= 60:
		risk_score += 1
	if not phys_activity:
		risk_score += 1
	if general_health in {PatientClinicalRecord.GeneralHealth.FAIR, PatientClinicalRecord.GeneralHealth.POOR}:
		risk_score += 1
	if diff_walk:
		risk_score += 1

	if risk_score >= 8:
		diabetes_status = _sample_diabetes_status(rng, risk_score)
	elif risk_score <= 3:
		diabetes_status = _sample_diabetes_status(rng, risk_score)
	else:
		diabetes_status = _sample_diabetes_status(rng, risk_score)

	return {
		"age_years": age_years,
		"hypertension": hypertension,
		"heart_disease": heart_disease,
		"smoking_history": smoking_history,
		"bmi": bmi,
		"hba1c": hba1c,
		"blood_glucose": blood_glucose,
		"high_bp": high_bp,
		"high_chol": high_chol,
		"chol_check": chol_check,
		"smoker": smoker,
		"stroke": stroke,
		"phys_activity": phys_activity,
		"fruits": fruits,
		"veggies": veggies,
		"heavy_alcohol": heavy_alcohol,
		"any_healthcare": any_healthcare,
		"no_docbc_cost": no_docbc_cost,
		"general_health": general_health,
		"mental_days": mental_days,
		"physical_days": physical_days,
		"diff_walk": diff_walk,
		"education": education,
		"income": income,
		"diabetes_status": diabetes_status,
	}


class Command(BaseCommand):
	help = "Generate realistic synthetic patients and clinical records for training tests."

	def add_arguments(self, parser):
		parser.add_argument("--count", type=int, default=100, help="How many patient records to create.")
		parser.add_argument("--start-index", type=int, default=1, help="Starting index for generated identities.")
		parser.add_argument(
			"--seed",
			type=int,
			default=42,
			help="Random seed for reproducible data generation.",
		)
		parser.add_argument(
			"--clear-existing",
			action="store_true",
			help="Delete existing patients and clinical records before generating new ones.",
		)

	def handle(self, *args, **options):
		count = options["count"]
		start_index = options["start_index"]
		seed = options["seed"]
		clear_existing = options["clear_existing"]

		if count <= 0:
			raise CommandError("--count must be greater than 0")

		rng = random.Random(seed)

		if clear_existing:
			PatientClinicalRecord.objects.all().delete()
			Patient.objects.all().delete()
			self.stdout.write(self.style.WARNING("Cleared existing patients and clinical records."))

		created_patients = 0
		created_records = 0
		skipped_existing = 0

		with transaction.atomic():
			for offset in range(count):
				index = start_index + offset
				first_name = rng.choice(FIRST_NAMES)
				last_name = rng.choice(LAST_NAMES)
				gender = rng.choice([PatientClinicalRecord.Gender.FEMALE, PatientClinicalRecord.Gender.MALE])
				profile = _build_synthetic_profile(rng)
				birth_date = _age_to_birthdate(profile["age_years"], rng)
				cnp = _make_cnp(index)

				if Patient.objects.filter(CNP=cnp).exists():
					skipped_existing += 1
					continue

				patient = Patient.objects.create(
					CNP=cnp,
					nume=last_name,
					prenume=first_name,
					data_nasterii=birth_date,
					serie_ci=chr(65 + (index % 26)) + chr(65 + ((index + 7) % 26)),
					numar_ci=_make_ci_number(index),
					nationalitate="Română",
					telefon=f"+40{7_000_000_00 + index:09d}"[:12],
					email=f"{first_name.lower()}.{last_name.lower()}{index}@example.com",
					oras=rng.choice(ORAS),
					judet=rng.choice(JUDET),
				)
				created_patients += 1

				PatientClinicalRecord.objects.create(
					patient=patient,
					gender=gender,
					age_years=profile["age_years"],
					hypertension=profile["hypertension"],
					heart_disease=profile["heart_disease"],
					smoking_history=profile["smoking_history"],
					bmi=Decimal(str(profile["bmi"])),
					hba1c_level=Decimal(str(profile["hba1c"])),
					blood_glucose_level=profile["blood_glucose"],
					high_bp=profile["high_bp"],
					high_chol=profile["high_chol"],
					chol_check=profile["chol_check"],
					smoker=profile["smoker"],
					stroke=profile["stroke"],
					phys_activity=profile["phys_activity"],
					fruits=profile["fruits"],
					veggies=profile["veggies"],
					heavy_alcohol_consumption=profile["heavy_alcohol"],
					any_healthcare=profile["any_healthcare"],
					no_docbc_cost=profile["no_docbc_cost"],
					general_health=profile["general_health"],
					mental_health_days=profile["mental_days"],
					physical_health_days=profile["physical_days"],
					diff_walk=profile["diff_walk"],
					education=profile["education"],
					income=profile["income"],
					diabetes_status=profile["diabetes_status"],
					data_consent_for_training=True,
					notes="Synthetic training record generated for federated learning testing.",
				)
				created_records += 1

		self.stdout.write(self.style.SUCCESS(
			f"Created {created_patients} patients and {created_records} clinical records. "
			f"Skipped {skipped_existing} existing CNPs."
		))