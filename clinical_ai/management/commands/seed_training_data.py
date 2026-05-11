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
				age_years = rng.randint(28, 82)
				birth_date = _age_to_birthdate(age_years, rng)
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

				risk_score = 0
				bmi = round(rng.uniform(20.0, 39.5), 2)
				hba1c = round(rng.uniform(4.6, 10.8), 2)
				blood_glucose = rng.randint(78, 310)

				hypertension = _rand_bool(rng, 0.35)
				heart_disease = _rand_bool(rng, 0.14)
				smoking_history = rng.choices(
					[
						PatientClinicalRecord.SmokingHistory.NEVER,
						PatientClinicalRecord.SmokingHistory.NO_INFO,
						PatientClinicalRecord.SmokingHistory.FORMER,
						PatientClinicalRecord.SmokingHistory.CURRENT,
					],
					weights=[0.45, 0.15, 0.25, 0.15],
					k=1,
				)[0]

				high_bp = hypertension or _rand_bool(rng, 0.18)
				high_chol = _rand_bool(rng, 0.34)
				chol_check = _rand_bool(rng, 0.83)
				smoker = smoking_history in {
					PatientClinicalRecord.SmokingHistory.FORMER,
					PatientClinicalRecord.SmokingHistory.CURRENT,
				}
				stroke = _rand_bool(rng, 0.05)
				phys_activity = _rand_bool(rng, 0.62)
				fruits = _rand_bool(rng, 0.55)
				veggies = _rand_bool(rng, 0.64)
				heavy_alcohol = _rand_bool(rng, 0.12)
				any_healthcare = True
				no_docbc_cost = _rand_bool(rng, 0.18)
				general_health = rng.choices(
					[
						PatientClinicalRecord.GeneralHealth.EXCELLENT,
						PatientClinicalRecord.GeneralHealth.VERY_GOOD,
						PatientClinicalRecord.GeneralHealth.GOOD,
						PatientClinicalRecord.GeneralHealth.FAIR,
						PatientClinicalRecord.GeneralHealth.POOR,
					],
					weights=[0.12, 0.24, 0.3, 0.22, 0.12],
					k=1,
				)[0]
				mental_days = rng.randint(0, 24)
				physical_days = rng.randint(0, 25)
				diff_walk = _rand_bool(rng, 0.28)
				education = rng.choice(list(PatientClinicalRecord.Education.values))
				income = rng.choice(list(PatientClinicalRecord.Income.values))

				if hypertension:
					risk_score += 2
				if heart_disease:
					risk_score += 2
				if smoker:
					risk_score += 2
				if bmi >= 30:
					risk_score += 2
				if hba1c >= 6.5:
					risk_score += 3
				if blood_glucose >= 180:
					risk_score += 3
				if age_years >= 55:
					risk_score += 1
				if not phys_activity:
					risk_score += 1

				if risk_score >= 7:
					diabetes_status = PatientClinicalRecord.DiabetesStatus.HAS
				elif risk_score <= 3:
					diabetes_status = PatientClinicalRecord.DiabetesStatus.HAS_NOT
				else:
					diabetes_status = PatientClinicalRecord.DiabetesStatus.NOT_CONFIRMED

				PatientClinicalRecord.objects.create(
					patient=patient,
					gender=gender,
					age_years=age_years,
					hypertension=hypertension,
					heart_disease=heart_disease,
					smoking_history=smoking_history,
					bmi=Decimal(str(bmi)),
					hba1c_level=Decimal(str(hba1c)),
					blood_glucose_level=blood_glucose,
					high_bp=high_bp,
					high_chol=high_chol,
					chol_check=chol_check,
					smoker=smoker,
					stroke=stroke,
					phys_activity=phys_activity,
					fruits=fruits,
					veggies=veggies,
					heavy_alcohol_consumption=heavy_alcohol,
					any_healthcare=any_healthcare,
					no_docbc_cost=no_docbc_cost,
					general_health=general_health,
					mental_health_days=mental_days,
					physical_health_days=physical_days,
					diff_walk=diff_walk,
					education=education,
					income=income,
					diabetes_status=diabetes_status,
					data_consent_for_training=True,
					notes="Synthetic training record generated for federated learning testing.",
				)
				created_records += 1

		self.stdout.write(self.style.SUCCESS(
			f"Created {created_patients} patients and {created_records} clinical records. "
			f"Skipped {skipped_existing} existing CNPs."
		))