import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from django.core.management.base import BaseCommand, CommandError


def _load_report(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: object) -> Optional[float]:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_value):
        return None
    return float(numeric_value)


def _seed_from_report(report: Dict[str, object], fallback: int) -> int:
    dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
    seed_value = dataset.get("seed") if isinstance(dataset, dict) else None
    seed = _safe_float(seed_value)
    return int(seed) if seed is not None else fallback


def _metric_value(report: Dict[str, object], block_name: str, metric_name: str) -> Optional[float]:
    block = report.get(block_name)
    if not isinstance(block, dict):
        return None
    return _safe_float(block.get(metric_name))


def _threshold_value(report: Dict[str, object], block_name: str, metric_name: str) -> Optional[float]:
    block = report.get(block_name)
    if not isinstance(block, dict):
        return None
    threshold_metrics = block.get("threshold_metrics")
    if not isinstance(threshold_metrics, dict):
        return None
    default_block = threshold_metrics.get("default_or_override")
    if not isinstance(default_block, dict):
        return None
    return _safe_float(default_block.get(metric_name))


def _bootstrap_summary(values: Sequence[float], seed: int, n_bootstrap: int = 2000) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None, "lower": None, "upper": None, "min": None, "max": None}

    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(array, size=len(array), replace=True)
        bootstrap_means.append(float(np.mean(sample)))

    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "lower": float(lower),
        "upper": float(upper),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _paired_bootstrap_delta(
    left_values: Sequence[float],
    right_values: Sequence[float],
    seed: int,
    n_bootstrap: int = 2000,
) -> Dict[str, Optional[float]]:
    if not left_values or not right_values or len(left_values) != len(right_values):
        return {"mean_delta": None, "std_delta": None, "lower": None, "upper": None}

    left = np.asarray(left_values, dtype=np.float64)
    right = np.asarray(right_values, dtype=np.float64)
    deltas = left - right
    rng = np.random.default_rng(seed)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(deltas), len(deltas))
        bootstrap_means.append(float(np.mean(deltas[indices])))

    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas, ddof=0)),
        "lower": float(lower),
        "upper": float(upper),
    }


def _mcnemar_pvalue(b: int, c: int) -> float:
    total = b + c
    if total == 0:
        return 1.0
    chi2 = ((abs(b - c) - 1) ** 2) / total
    return float(math.erfc(math.sqrt(max(chi2, 0.0) / 2.0)))


def _mcnemar_from_predictions(y_true: Sequence[int], pred_left: Sequence[int], pred_right: Sequence[int]) -> Dict[str, Optional[float]]:
    if not y_true or not pred_left or not pred_right:
        return {"b": None, "c": None, "chi2": None, "p_value": None}

    y = np.asarray(y_true, dtype=np.int32)
    left = np.asarray(pred_left, dtype=np.int32)
    right = np.asarray(pred_right, dtype=np.int32)

    left_correct = left == y
    right_correct = right == y
    b = int(np.sum(left_correct & ~right_correct))
    c = int(np.sum(~left_correct & right_correct))
    chi2 = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0.0
    p_value = _mcnemar_pvalue(b, c)
    return {"b": b, "c": c, "chi2": float(chi2), "p_value": float(p_value)}


def _compute_midrank(values: np.ndarray) -> np.ndarray:
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    midranks = np.empty(len(values), dtype=np.float64)

    index = 0
    while index < len(values):
        end = index
        while end < len(values) and sorted_values[end] == sorted_values[index]:
            end += 1
        midrank = 0.5 * (index + end - 1) + 1.0
        midranks[index:end] = midrank
        index = end

    result = np.empty(len(values), dtype=np.float64)
    result[sorted_indices] = midranks
    return result


def _delong_covariance(y_true: Sequence[int], predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=np.int32)
    if len(np.unique(labels)) != 2:
        raise ValueError("DeLong requires both positive and negative examples")

    order = np.argsort(-labels)
    label_1_count = int(np.sum(labels))
    predictions_sorted = predictions[:, order]

    positive_predictions = predictions_sorted[:, :label_1_count]
    negative_predictions = predictions_sorted[:, label_1_count:]

    tx = np.vstack([_compute_midrank(row) for row in positive_predictions])
    ty = np.vstack([_compute_midrank(row) for row in negative_predictions])
    tz = np.vstack([_compute_midrank(row) for row in predictions_sorted])

    m = float(label_1_count)
    n = float(predictions_sorted.shape[1] - label_1_count)

    aucs = tz[:, :label_1_count].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :label_1_count] - tx) / n
    v10 = 1.0 - (tz[:, label_1_count:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    if np.ndim(sx) == 0:
        sx = np.asarray([[float(sx)]], dtype=np.float64)
    if np.ndim(sy) == 0:
        sy = np.asarray([[float(sy)]], dtype=np.float64)

    covariance = sx / m + sy / n
    return aucs, covariance


def _delong_pairwise_test(
    y_true: Sequence[int],
    scores_left: Sequence[float],
    scores_right: Sequence[float],
) -> Dict[str, Optional[float]]:
    if not y_true or not scores_left or not scores_right:
        return {"auc_left": None, "auc_right": None, "auc_diff": None, "z_score": None, "p_value": None, "ci_low": None, "ci_high": None}

    y = np.asarray(y_true, dtype=np.int32)
    left = np.asarray(scores_left, dtype=np.float64)
    right = np.asarray(scores_right, dtype=np.float64)
    if len(y) != len(left) or len(y) != len(right):
        return {"auc_left": None, "auc_right": None, "auc_diff": None, "z_score": None, "p_value": None, "ci_low": None, "ci_high": None}

    predictions = np.vstack([left, right])
    try:
        aucs, covariance = _delong_covariance(y, predictions)
    except Exception:
        return {"auc_left": None, "auc_right": None, "auc_diff": None, "z_score": None, "p_value": None, "ci_low": None, "ci_high": None}

    auc_left = float(aucs[0])
    auc_right = float(aucs[1])
    auc_diff = auc_left - auc_right
    diff_var = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    diff_var = max(diff_var, 0.0)
    diff_se = math.sqrt(diff_var)
    z_score = auc_diff / diff_se if diff_se > 0 else None
    p_value = float(math.erfc(abs(z_score) / math.sqrt(2.0))) if z_score is not None else None
    ci_low = auc_diff - 1.96 * diff_se if diff_se > 0 else auc_diff
    ci_high = auc_diff + 1.96 * diff_se if diff_se > 0 else auc_diff

    return {
        "auc_left": auc_left,
        "auc_right": auc_right,
        "auc_diff": float(auc_diff),
        "z_score": float(z_score) if z_score is not None else None,
        "p_value": p_value,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def _collect_reports(report_paths: Sequence[str]) -> List[Dict[str, object]]:
    reports: List[Dict[str, object]] = []
    for raw_path in report_paths:
        path = Path(raw_path)
        if not path.exists():
            raise CommandError(f"Report file not found: {path}")
        reports.append(_load_report(path))
    reports_with_seeds = sorted(
        enumerate(reports),
        key=lambda item: _seed_from_report(item[1], item[0]),
    )
    return [report for _, report in reports_with_seeds]


def _metric_summary_for_reports(
    reports: Sequence[Dict[str, object]],
    block_name: str,
    metric_name: str,
    paired_with: Optional[Sequence[Dict[str, object]]] = None,
    paired_block_name: Optional[str] = None,
    threshold_metric: bool = False,
) -> Dict[str, object]:
    values = []
    for report in reports:
        value = _threshold_value(report, block_name, metric_name) if threshold_metric else _metric_value(report, block_name, metric_name)
        if value is not None:
            values.append(value)

    stable_seed = 1234 + sum(ord(char) for char in f"{block_name}:{metric_name}")
    summary = _bootstrap_summary(values, seed=stable_seed)
    result: Dict[str, object] = {"summary": summary}

    if paired_with is not None:
        paired_values_left = []
        paired_values_right = []
        compare_block = paired_block_name or block_name
        for left_report, right_report in zip(reports, paired_with):
            left_value = _threshold_value(left_report, block_name, metric_name) if threshold_metric else _metric_value(left_report, block_name, metric_name)
            right_value = _threshold_value(right_report, compare_block, metric_name) if threshold_metric else _metric_value(right_report, compare_block, metric_name)
            if left_value is not None and right_value is not None:
                paired_values_left.append(left_value)
                paired_values_right.append(right_value)
        result["paired_delta"] = _paired_bootstrap_delta(paired_values_left, paired_values_right, seed=4321 + hash((block_name, metric_name)) % 1000)

    return result


def _extract_score_arrays(report: Dict[str, object], block_name: str) -> Tuple[Optional[List[int]], Optional[List[float]]]:
    per_sample = report.get("per_sample")
    if not isinstance(per_sample, dict):
        return None, None

    if block_name == "calibrated_core_model_quality":
        y_pred = per_sample.get("y_pred_calibrated") or per_sample.get("y_pred_raw")
    else:
        y_pred = per_sample.get("y_pred_raw") or per_sample.get("y_pred_calibrated")

    y_true = per_sample.get("y_true")
    if not isinstance(y_true, list) or not isinstance(y_pred, list):
        return None, None

    try:
        return [int(value) for value in y_true], [float(value) for value in y_pred]
    except Exception:
        return None, None


def _other_block_name(block_name: str) -> str:
    return "calibrated_core_model_quality" if block_name == "core_model_quality" else "core_model_quality"


class Command(BaseCommand):
    help = "Summarize statistical validation across multiple seed-based model-quality reports."

    def add_arguments(self, parser):
        parser.add_argument(
            "--baseline-report",
            action="append",
            required=True,
            help="Path to a compute_core_model_quality JSON report. Repeat for multiple seeds.",
        )
        parser.add_argument(
            "--variant-report",
            action="append",
            default=None,
            help="Optional second set of reports for paired comparisons. Repeat in the same seed order.",
        )
        parser.add_argument(
            "--block",
            default="core_model_quality",
            choices=("core_model_quality", "calibrated_core_model_quality"),
            help="Report block to summarize.",
        )
        parser.add_argument(
            "--paired-block",
            default=None,
            choices=("core_model_quality", "calibrated_core_model_quality"),
            help="Optional paired block to compare against. Defaults to the opposite block when no variant reports are supplied.",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Optional output JSON path.",
        )

    def handle(self, *args, **options):
        baseline_reports = _collect_reports(options["baseline_report"])
        variant_report_paths = options.get("variant_report") or []
        variant_reports = _collect_reports(variant_report_paths) if variant_report_paths else None

        paired_reports = variant_reports if variant_reports is not None else baseline_reports
        paired_block_name = options.get("paired_block") or (
            options["block"] if variant_reports is not None else _other_block_name(options["block"])
        )

        if variant_reports is not None and len(variant_reports) != len(baseline_reports):
            raise CommandError("--variant-report must contain the same number of reports as --baseline-report")

        baseline_seeds = [_seed_from_report(report, index) for index, report in enumerate(baseline_reports)]
        variant_seeds = [_seed_from_report(report, index) for index, report in enumerate(variant_reports)] if variant_reports is not None else []

        if variant_reports is not None and baseline_seeds != variant_seeds:
            raise CommandError("Baseline and variant reports must correspond to the same seed order for paired comparison")

        metrics = [
            ("roc_auc", False),
            ("pr_auc", False),
            ("brier_score", False),
            ("expected_calibration_error", False),
            ("accuracy", True),
            ("f1", True),
            ("precision", True),
            ("recall", True),
        ]

        summary: Dict[str, object] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "block": options["block"],
            "seed_count": len(baseline_reports),
            "baseline": {},
        }

        summary["baseline"]["seeds"] = baseline_seeds

        for metric_name, threshold_metric in metrics:
            summary["baseline"][metric_name] = _metric_summary_for_reports(
                baseline_reports,
                block_name=options["block"],
                metric_name=metric_name,
                threshold_metric=threshold_metric,
            )["summary"]

        summary["paired_comparison"] = {
            "seeds": variant_seeds if variant_reports is not None else baseline_seeds,
            "paired_block": paired_block_name,
        }
        for metric_name, threshold_metric in metrics:
            summary["paired_comparison"][metric_name] = _metric_summary_for_reports(
                baseline_reports,
                block_name=options["block"],
                metric_name=metric_name,
                paired_with=paired_reports,
                paired_block_name=paired_block_name,
                threshold_metric=threshold_metric,
            )["paired_delta"]

        summary["paired_comparison"]["delong_auc"] = []
        for seed, baseline_report, variant_report in zip(baseline_seeds, baseline_reports, paired_reports):
            baseline_y, baseline_scores = _extract_score_arrays(baseline_report, options["block"])
            variant_y, variant_scores = _extract_score_arrays(variant_report, paired_block_name)
            if baseline_y is None or baseline_scores is None or variant_y is None or variant_scores is None:
                continue
            if baseline_y != variant_y:
                continue
            delong_result = _delong_pairwise_test(baseline_y, baseline_scores, variant_scores)
            delong_result["seed"] = int(seed)
            summary["paired_comparison"]["delong_auc"].append(delong_result)

        baseline_first = baseline_reports[0]
        variant_first = paired_reports[0]
        if isinstance(baseline_first.get("per_sample"), dict) and isinstance(variant_first.get("per_sample"), dict):
            baseline_per = baseline_first["per_sample"]
            variant_per = variant_first["per_sample"]
            y_true = baseline_per.get("y_true", [])
            baseline_pred = baseline_per.get("y_pred_raw", []) if options["block"] == "core_model_quality" else baseline_per.get("y_pred_calibrated", [])
            variant_pred = variant_per.get("y_pred_raw", []) if paired_block_name == "core_model_quality" else variant_per.get("y_pred_calibrated", [])
            if len(y_true) == len(baseline_pred) == len(variant_pred):
                summary["paired_comparison"]["mcnemar_raw_threshold_0_5"] = _mcnemar_from_predictions(
                    y_true=y_true,
                    pred_left=[int(float(x) >= 0.5) for x in baseline_pred],
                    pred_right=[int(float(x) >= 0.5) for x in variant_pred],
                )

        if options["output"]:
            destination = Path(options["output"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        self.stdout.write(self.style.SUCCESS("Statistical validation summary generated"))
        self.stdout.write(json.dumps(summary, indent=2))