import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from django.core.management.base import BaseCommand, CommandError


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def _load_report(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_clinic_spec(spec: str) -> Tuple[str, Path]:
    raw_spec = spec.strip()
    if not raw_spec:
        raise CommandError("--clinic-report cannot be empty")

    if "=" in raw_spec:
        clinic_label, report_path = raw_spec.split("=", 1)
        clinic_label = clinic_label.strip()
        report_path = report_path.strip()
        if not clinic_label or not report_path:
            raise CommandError(f"Invalid --clinic-report value: {spec!r}. Expected label=path")
    else:
        report_path = raw_spec
        clinic_label = Path(report_path).stem

    return clinic_label, Path(report_path)


def _extract_block(report: Dict[str, object], block_name: str) -> Dict[str, object]:
    block = report.get(block_name)
    if not isinstance(block, dict):
        return {}
    return block


def _extract_threshold_metric(block: Dict[str, object], metric_name: str) -> Optional[float]:
    threshold_metrics = block.get("threshold_metrics")
    if not isinstance(threshold_metrics, dict):
        return None

    default_block = threshold_metrics.get("default_or_override")
    if not isinstance(default_block, dict):
        return None

    return _coerce_float(default_block.get(metric_name))


def _summarize_numeric_series(values_by_clinic: Dict[str, Optional[float]]) -> Dict[str, object]:
    available: List[Tuple[str, float]] = [
        (clinic, value)
        for clinic, value in values_by_clinic.items()
        if value is not None
    ]

    if not available:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "range": None,
            "best_clinic": None,
            "worst_clinic": None,
            "by_clinic": values_by_clinic,
        }

    values = np.asarray([value for _, value in available], dtype=np.float64)
    best_clinic, best_value = max(available, key=lambda item: item[1])
    worst_clinic, worst_value = min(available, key=lambda item: item[1])

    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
        "best_clinic": best_clinic,
        "best_value": float(best_value),
        "worst_clinic": worst_clinic,
        "worst_value": float(worst_value),
        "by_clinic": values_by_clinic,
    }


def _collect_block_summary(reports: Dict[str, Dict[str, object]], block_name: str) -> Dict[str, object]:
    clinic_rows: Dict[str, Dict[str, object]] = {}

    for clinic_label, report in reports.items():
        block = _extract_block(report, block_name)
        dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
        threshold_block = block.get("threshold_metrics") if isinstance(block, dict) else {}
        default_threshold = None
        if isinstance(threshold_block, dict):
            default_metrics = threshold_block.get("default_or_override")
            if isinstance(default_metrics, dict):
                default_threshold = _coerce_float(default_metrics.get("threshold"))

        clinic_rows[clinic_label] = {
            "report_path": report.get("report_path"),
            "generated_at": report.get("generated_at"),
            "model": report.get("model"),
            "evaluation_records": dataset.get("evaluation_records") if isinstance(dataset, dict) else None,
            "usable_records": dataset.get("usable_records") if isinstance(dataset, dict) else None,
            "roc_auc": _coerce_float(block.get("roc_auc")),
            "pr_auc": _coerce_float(block.get("pr_auc")),
            "brier_score": _coerce_float(block.get("brier_score")),
            "expected_calibration_error": _coerce_float(block.get("expected_calibration_error")),
            "calibration_slope": _coerce_float(block.get("calibration_slope")),
            "calibration_intercept": _coerce_float(block.get("calibration_intercept")),
            "threshold": default_threshold,
            "default_accuracy": _extract_threshold_metric(block, "accuracy"),
            "default_f1": _extract_threshold_metric(block, "f1"),
            "default_precision": _extract_threshold_metric(block, "precision"),
            "default_recall": _extract_threshold_metric(block, "recall"),
        }

    summary_metrics = {}
    for metric_name in [
        "roc_auc",
        "pr_auc",
        "brier_score",
        "expected_calibration_error",
        "calibration_slope",
        "calibration_intercept",
        "default_accuracy",
        "default_f1",
        "default_precision",
        "default_recall",
    ]:
        summary_metrics[metric_name] = _summarize_numeric_series(
            {clinic_label: row.get(metric_name) for clinic_label, row in clinic_rows.items()}
        )

    return {
        "clinics": clinic_rows,
        "summary": summary_metrics,
        "clinic_count": len(clinic_rows),
    }


def build_comparison_report(clinic_reports: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    if len(clinic_reports) < 2:
        raise CommandError("Provide at least two clinic reports for comparison")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison": {
            "raw": _collect_block_summary(clinic_reports, "core_model_quality"),
            "calibrated": _collect_block_summary(clinic_reports, "calibrated_core_model_quality"),
        },
    }


class Command(BaseCommand):
    help = "Compare clinic-specific quality reports and summarize cross-clinic variability."

    def add_arguments(self, parser):
        parser.add_argument(
            "--clinic-report",
            action="append",
            required=True,
            help="Clinic report in the form label=path. You can repeat this option for each clinic.",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Optional output JSON path. Defaults to reports/model_quality/clinic_quality_comparison_<timestamp>.json",
        )

    def handle(self, *args, **options):
        clinic_reports: Dict[str, Dict[str, object]] = {}

        for spec in options["clinic_report"]:
            clinic_label, report_path = _parse_clinic_spec(spec)
            if clinic_label in clinic_reports:
                raise CommandError(f"Duplicate clinic label: {clinic_label}")
            if not report_path.exists():
                raise CommandError(f"Report file not found for {clinic_label}: {report_path}")

            report = _load_report(report_path)
            report["report_path"] = str(report_path)
            clinic_reports[clinic_label] = report

        comparison_report = build_comparison_report(clinic_reports)

        if options["output"]:
            destination = Path(options["output"])
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            destination = Path("reports") / "model_quality" / f"clinic_quality_comparison_{timestamp}.json"

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(comparison_report, indent=2), encoding="utf-8")

        raw_summary = comparison_report["comparison"]["raw"]
        self.stdout.write(self.style.SUCCESS("Clinic quality comparison generated"))
        self.stdout.write(f"Clinics compared: {raw_summary['clinic_count']}")
        self.stdout.write(f"ROC AUC mean: {raw_summary['summary']['roc_auc']['mean']}")
        self.stdout.write(f"ROC AUC worst clinic: {raw_summary['summary']['roc_auc']['worst_clinic']}")
        self.stdout.write(f"Output: {destination}")