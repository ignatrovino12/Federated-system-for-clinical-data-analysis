from django.test import TestCase

from clinical_ai.management.commands.compare_clinic_quality_reports import (
	build_comparison_report,
	_parse_clinic_spec,
	_summarize_numeric_series,
)


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
