from django.test import TestCase

from clinical_ai.management.commands.compare_clinic_quality_reports import (
	build_comparison_report,
	_parse_clinic_spec,
	_summarize_numeric_series,
)
from clinical_ai.management.commands.generate_system_efficiency_report import build_system_efficiency_report


class CompareClinicQualityReportsTests(TestCase):
	def test_parse_clinic_spec_defaults_label_from_filename(self):
		clinic_label, report_path = _parse_clinic_spec("/tmp/clinic1_report.json")

		self.assertEqual(clinic_label, "clinic1_report")
		self.assertEqual(str(report_path), "/tmp/clinic1_report.json")

	def test_summarize_numeric_series_tracks_best_and_worst_clinic(self):
		summary = _summarize_numeric_series({"clinic_a": 0.78, "clinic_b": 0.91})

		self.assertAlmostEqual(summary["mean"], 0.845)
		self.assertAlmostEqual(summary["std"], 0.065)
		self.assertEqual(summary["best_clinic"], "clinic_b")
		self.assertEqual(summary["worst_clinic"], "clinic_a")
		self.assertAlmostEqual(summary["range"], 0.13)

	def test_build_comparison_report_includes_raw_and_calibrated_blocks(self):
		report_a = {
			"generated_at": "2026-06-11T00:00:00+00:00",
			"model": "mustafa",
			"dataset": {"evaluation_records": 20, "usable_records": 25},
			"core_model_quality": {
				"roc_auc": 0.90,
				"pr_auc": 0.88,
				"brier_score": 0.12,
				"expected_calibration_error": 0.05,
				"calibration_slope": 1.1,
				"calibration_intercept": -0.2,
				"threshold_metrics": {
					"default_or_override": {
						"threshold": 0.5,
						"accuracy": 0.8,
						"precision": 0.75,
						"recall": 0.85,
						"f1": 0.79,
					}
				},
			},
			"calibrated_core_model_quality": {
				"roc_auc": 0.91,
				"pr_auc": 0.89,
				"brier_score": 0.11,
				"expected_calibration_error": 0.04,
				"calibration_slope": 1.0,
				"calibration_intercept": -0.1,
				"threshold_metrics": {
					"default_or_override": {
						"threshold": 0.5,
						"accuracy": 0.81,
						"precision": 0.76,
						"recall": 0.86,
						"f1": 0.80,
					}
				},
			},
			"report_path": "clinic_a.json",
		}
		report_b = {
			"generated_at": "2026-06-11T00:00:00+00:00",
			"model": "mustafa",
			"dataset": {"evaluation_records": 22, "usable_records": 26},
			"core_model_quality": {
				"roc_auc": 0.84,
				"pr_auc": 0.81,
				"brier_score": 0.15,
				"expected_calibration_error": 0.06,
				"calibration_slope": 0.95,
				"calibration_intercept": -0.3,
				"threshold_metrics": {
					"default_or_override": {
						"threshold": 0.5,
						"accuracy": 0.77,
						"precision": 0.70,
						"recall": 0.80,
						"f1": 0.74,
					}
				},
			},
			"calibrated_core_model_quality": {
				"roc_auc": 0.85,
				"pr_auc": 0.82,
				"brier_score": 0.14,
				"expected_calibration_error": 0.05,
				"calibration_slope": 0.96,
				"calibration_intercept": -0.25,
				"threshold_metrics": {
					"default_or_override": {
						"threshold": 0.5,
						"accuracy": 0.78,
						"precision": 0.71,
						"recall": 0.81,
						"f1": 0.75,
					}
				},
			},
			"report_path": "clinic_b.json",
		}

		comparison = build_comparison_report({"clinic_a": report_a, "clinic_b": report_b})

		raw_summary = comparison["comparison"]["raw"]
		calibrated_summary = comparison["comparison"]["calibrated"]

		self.assertEqual(raw_summary["clinic_count"], 2)
		self.assertAlmostEqual(raw_summary["summary"]["roc_auc"]["mean"], 0.87)
		self.assertEqual(raw_summary["summary"]["roc_auc"]["best_clinic"], "clinic_a")
		self.assertEqual(raw_summary["summary"]["roc_auc"]["worst_clinic"], "clinic_b")
		self.assertAlmostEqual(calibrated_summary["summary"]["brier_score"]["mean"], 0.125)

	def test_build_system_efficiency_report_aggregates_prometheus_samples(self):
		def fetch(query: str):
			if "healthcheck_client_fit_rounds_total" in query:
				return [
					{"metric": {"clinic": "clinic_a", "model": "mustafa", "status": "success"}, "value": [0, "3"]},
					{"metric": {"clinic": "clinic_a", "model": "mustafa", "status": "timeout"}, "value": [0, "1"]},
				]
			if "healthcheck_client_evaluate_rounds_total" in query:
				return [{"metric": {"clinic": "clinic_a", "model": "mustafa", "status": "success"}, "value": [0, "2"]}]
			if "healthcheck_client_fit_duration_seconds_sum" in query:
				return [{"metric": {"clinic": "clinic_a", "model": "mustafa"}, "value": [0, "12.0"]}]
			if "healthcheck_client_fit_duration_seconds_count" in query:
				return [{"metric": {"clinic": "clinic_a", "model": "mustafa"}, "value": [0, "3"]}]
			if "healthcheck_client_evaluate_duration_seconds_sum" in query:
				return [{"metric": {"clinic": "clinic_a", "model": "mustafa"}, "value": [0, "6.0"]}]
			if "healthcheck_client_evaluate_duration_seconds_count" in query:
				return [{"metric": {"clinic": "clinic_a", "model": "mustafa"}, "value": [0, "2"]}]
			if "healthcheck_client_parameters_sent_bytes_total" in query:
				return [{"metric": {"clinic": "clinic_a", "model": "mustafa"}, "value": [0, "4096"]}]
			if "healthcheck_client_parameters_received_bytes_total" in query:
				return [{"metric": {"clinic": "clinic_a", "model": "mustafa"}, "value": [0, "2048"]}]
			if "healthcheck_client_training_loss" in query:
				return [{"metric": {"clinic": "clinic_a", "model": "mustafa"}, "value": [0, "0.21"]}]
			if "healthcheck_client_evaluation_accuracy" in query:
				return [{"metric": {"clinic": "clinic_a", "model": "mustafa"}, "value": [0, "0.89"]}]
			if "healthcheck_client_evaluation_loss" in query:
				return [{"metric": {"clinic": "clinic_a", "model": "mustafa"}, "value": [0, "0.34"]}]
			if "healthcheck_flower_rounds_total" in query:
				return [{"metric": {"model": "mustafa", "status": "success"}, "value": [0, "5"]}]
			if "healthcheck_flower_round_duration_seconds_sum" in query:
				return [{"metric": {"model": "mustafa"}, "value": [0, "25.0"]}]
			if "healthcheck_flower_round_duration_seconds_count" in query:
				return [{"metric": {"model": "mustafa"}, "value": [0, "5"]}]
			if "healthcheck_flower_clients_per_round_sum" in query:
				return [{"metric": {"model": "mustafa"}, "value": [0, "15"]}]
			if "healthcheck_flower_clients_per_round_count" in query:
				return [{"metric": {"model": "mustafa"}, "value": [0, "5"]}]
			if "healthcheck_flower_current_round" in query:
				return [{"metric": {"model": "mustafa"}, "value": [0, "7"]}]
			if "healthcheck_flower_last_checkpoint_round" in query:
				return [{"metric": {"model": "mustafa"}, "value": [0, "5"]}]
			if 'up{job="healthcheck-flower"}' in query:
				return [{"metric": {"instance": "flower", "job": "healthcheck-flower"}, "value": [0, "1"]}]
			if 'up{job="healthcheck-pushgateway"}' in query:
				return [{"metric": {"instance": "pushgateway", "job": "healthcheck-pushgateway"}, "value": [0, "1"]}]
			return []

		report = build_system_efficiency_report(fetch, window="24h")

		self.assertEqual(report["window"], "24h")
		self.assertEqual(report["summary"]["clinic_count"], 1)
		self.assertEqual(report["summary"]["flower_model_count"], 1)
		self.assertEqual(len(report["clinics"]), 1)
		self.assertEqual(report["clinics"][0]["clinic"], "clinic_a")
		self.assertEqual(report["clinics"][0]["fit_rounds"]["total"], 4.0)
		self.assertEqual(report["clinics"][0]["fit_duration_seconds"]["average"], 4.0)
		self.assertEqual(report["flower"][0]["rounds"]["total"], 5.0)
		self.assertEqual(report["summary"]["clinics"]["bytes_sent"], 4096.0)
		self.assertTrue(report["summary"]["flower"]["flower_up"])
